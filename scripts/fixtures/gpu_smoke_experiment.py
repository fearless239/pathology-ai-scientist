"""Fixed integration fixture, not an autonomous scientific result."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

seed = 0
num_epochs = 2
torch.manual_seed(seed)
torch.set_num_threads(2)
assert torch.cuda.is_available(), "Real GPU required"
assert os.getuid() != 0, "Non-root sandbox required"
assert not any(key.endswith("API_KEY") for key in os.environ), "Unexpected credential"
routes = Path('/proc/net/route').read_text()
assert not any(line.split()[1] == '00000000' for line in routes.splitlines()[1:] if line.split()), "Unexpected default network route"
data = np.load('/dataset/dataset.npz', allow_pickle=False)
HAS_TRAIN_SPLIT = 'train_images' in data.files
assert not any(key.startswith('test') for key in data.files)
working = Path('working')
working.mkdir(exist_ok=True)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # AUGMENTATIONS
    transforms.ToTensor(),
])
eval_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


class Images(Dataset):
    def __init__(self, split, transform):
        self.images = data[split + '_images']
        self.labels = data[split + '_labels'].reshape(-1)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), int(self.labels[index])


model = resnet18(weights=None, num_classes=9).cuda()
if HAS_TRAIN_SPLIT:
    train_dataset = Images('train', train_transform)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images.cuda()), labels.cuda())
            loss.backward()
            optimizer.step()
        print(f'seed={seed} epoch={epoch + 1}/{num_epochs} loss={loss.item():.5f}', flush=True)
        (working / 'epoch_progress.json').write_text(json.dumps({'seed': seed, 'epoch': epoch + 1}))
    torch.save(model.state_dict(), working / 'model_checkpoint.pt')
else:
    model.load_state_dict(torch.load(working / 'model_checkpoint.pt', weights_only=True))

model.eval()
predictions, targets, probabilities = [], [], []
with torch.no_grad():
    for images, labels in DataLoader(Images('validation', eval_transform), batch_size=32):
        probs = model(images.cuda()).softmax(dim=1).cpu()
        probabilities.extend(probs.tolist())
        predictions.extend(probs.argmax(dim=1).tolist())
        targets.extend(labels.tolist())
accuracy = sum(a == b for a, b in zip(predictions, targets)) / len(targets)
(working / 'experiment_result.json').write_text(json.dumps({
    'seed': seed, 'metrics': {'accuracy': accuracy}, 'predictions': predictions,
    'targets': targets, 'sample_ids': data['validation_sample_ids'].tolist(),
    'probabilities': probabilities, 'test_data_accessed': False,
}))
(working / 'experiment_manifest.json').write_text(json.dumps({
    'schema_version': 1, 'dataset': 'PathMNIST-smoke-subset', 'model': 'ResNet-18',
    'optimizer': 'Adam', 'learning_rate': 0.001, 'epochs': num_epochs, 'batch_size': 32,
    'seed': seed, 'input_resolutions': [64], 'selection_metric': 'accuracy', 'hardware': 'cuda',
}))
(working / 'contract_execution.json').write_text(json.dumps({'contract_role': 'ROLE', 'training_seed': seed}))
(working / 'isolation.json').write_text(json.dumps({'uid': os.getuid(), 'cuda': True, 'no_api_keys': True, 'no_default_route': True}))
print(f'validation accuracy={accuracy:.6f}', flush=True)
