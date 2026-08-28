"""Offline engineering verification only; never produces formal task results."""
import json
import subprocess
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/project')
from gate_a.import_preflight import sandbox_launcher

workspace = Path('/workspace')
workspace.joinpath('working').mkdir(exist_ok=True)
common = '''import torch
import numpy as np
from pathlib import Path
torch.set_num_threads(2)
torch.manual_seed(7)
device = torch.device('cuda')
model = torch.nn.Sequential(
    torch.nn.Conv2d(3,8,3,padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(8,8,3,padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(8,8,3,padding=1), torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8,9)
).to(device)
def prepare(array):
    x=torch.from_numpy(array.copy()).permute(0,3,1,2).float().to(device)/255
    return torch.nn.functional.interpolate(x,size=(28,28),mode='bilinear',align_corners=False)
'''
train = common + '''
with np.load('/dataset/dataset.npz',allow_pickle=False) as data:
    x=prepare(data['train_images'][:16])
    y=torch.from_numpy(data['train_labels'][:16].copy()).long().to(device)
    validation=prepare(data['validation_images'][:4])
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
loss=torch.nn.functional.cross_entropy(model(x),y)
loss.backward()
optimizer.step()
torch.save(model.state_dict(),'working/model_checkpoint.pt')
model.eval()
with torch.no_grad():
    torch.save({'x':validation.cpu(),'logits':model(validation).cpu()},'working/reference.pt')
print('SHORT_TRAIN_COMPLETE samples=16 validation=4')
'''
infer = common + '''
model.load_state_dict(torch.load('/workspace/model_checkpoint.pt',map_location=device,weights_only=True))
model.eval()
reference=torch.load('working/reference.pt',map_location=device,weights_only=True)
with torch.no_grad():
    logits=model(reference['x'])
assert torch.allclose(logits,reference['logits'],atol=1e-6)
print('INDEPENDENT_RELOAD_PREDICTIONS_MATCH')
'''
records = []
for label, code, inference in [('train',train,False),('inference',infer,True)]:
    if inference:
        (workspace/'model_checkpoint.pt').write_bytes((workspace/'working/model_checkpoint.pt').read_bytes())
    file = workspace/f'{label}.py'
    file.write_text(code)
    result=subprocess.run([sys.executable,'-c',sandbox_launcher(code,str(file),
        {'input_sizes':[[28,28]],'inference_only':inference})],cwd=workspace,
        capture_output=True,text=True,timeout=120)
    print(result.stdout,flush=True)
    if result.returncode:
        print(result.stderr,flush=True)
        raise SystemExit(result.returncode)
    records.append({'phase':label,'passed':True,'output':result.stdout})
checkpoint = workspace/'model_checkpoint.pt'
backup = checkpoint.read_bytes()
def negative(label, body, inference, expected):
    file=workspace/f'negative_{label}.py'
    code=common+body
    file.write_text(code)
    result=subprocess.run([sys.executable,'-c',sandbox_launcher(code,str(file),
        {'input_sizes':[[28,28]],'inference_only':inference})],cwd=workspace,
        capture_output=True,text=True,timeout=40)
    assert result.returncode != 0 and expected in result.stderr, (label,result.stdout,result.stderr)
    records.append({'phase':label,'passed':True,'expected_rejection':expected})
    print('EXPECTED_REJECTION_PASSED '+label,flush=True)
negative('wrong_size',"\ninput_resolutions=[28]\nmodel(torch.randn(1,3,64,64,device=device))",False,'actual model input (64, 64)')
negative('never_loaded',"\nmodel(torch.randn(1,3,28,28,device=device))",True,'before successful checkpoint')
negative('load_without_restore',"\ntorch.load('/workspace/model_checkpoint.pt',weights_only=True)\nmodel(torch.randn(1,3,28,28,device=device))",True,'before successful checkpoint')
negative('random_state',"\nmodel.load_state_dict(model.state_dict())",True,'weights from the required checkpoint')
checkpoint.unlink()
negative('missing_checkpoint',"\nmodel(torch.randn(1,3,28,28,device=device))",True,'inference requires')
checkpoint.write_bytes(b'corrupt')
negative('corrupt_checkpoint',"\ntry:\n    torch.load('/workspace/model_checkpoint.pt',weights_only=True)\nexcept Exception:\n    pass\nmodel(torch.randn(1,3,28,28,device=device))",True,'before successful checkpoint')
torch.save({'0.weight':torch.ones(8,3,3,3)},checkpoint)
negative('partial_checkpoint',"\nmodel.load_state_dict(torch.load('/workspace/model_checkpoint.pt',weights_only=True),strict=False)",True,'partial loading is forbidden')
checkpoint.write_bytes(backup)
negative('caught_shape_error',"\ntry:\n    model(torch.randn(1,3,64,64,device=device))\nexcept RuntimeError:\n    pass\nmodel(torch.randn(1,3,28,28,device=device))",False,'MODEL_CONTRACT_FAILED')
negative('early_success_exit',"\nraise SystemExit(0)",False,'no instrumented image-model forward')
report={'engineering_only':True,'train_samples':16,'validation_samples':4,
        'sealed_test_accessed':False,'paid_api_calls':0,'records':records}
(workspace/'verification.json').write_text(json.dumps(report,indent=2))
print('ALL_MODEL_RUNTIME_CHECKS_PASSED',flush=True)
