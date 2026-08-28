# PATH_AI_METHOD_SPEC: {"schema_version":1,"hypothesis":"Replacing standard cross-entropy with label smoothing cross-entropy on the same lightweight 3-conv-layer CNN improves PathMNIST validation accuracy by at least 1.0pp under identical splits, seeds, and training controls.","components":[{"id":"data_pipeline","category":"data_loading","implementation_symbols":["npz_path","train_images","validation_images","validation_sample_ids","NHWC","NCHW","resize_28","normalize"]},{"id":"model","category":"cnn_architecture","implementation_symbols":["Conv2d","BatchNorm2d","ReLU","MaxPool2d","Linear","Dropout","from_scratch","num_classes"]},{"id":"loss","category":"cross_entropy","implementation_symbols":["CrossEntropyLoss","label_smoothing","smoothing_factor"]},{"id":"optimizer","category":"sgd","implementation_symbols":["SGD","momentum","weight_decay"]},{"id":"training","category":"supervised_training","implementation_symbols":["DataLoader","epochs","early_stopping","seed","train_subset_fraction"]},{"id":"evaluation","category":"classification_metrics","implementation_symbols":["accuracy","macro_f1","weighted_f1","confusion_matrix"]},{"id":"param_search","category":"label_smoothing_tuning","implementation_symbols":["candidate_smoothing_factors","tuning_progress","tuning_evidence","checkpoint_selection","selected_parameters"]}],"changes":["Switch loss to torch.nn.CrossEntropyLoss(label_smoothing=smoothing_factor)","Search smoothing factor over 3 candidates on 20% train subset with 5 epochs each (15 total)","Final fit on full training set with selected smoothing factor for 12 epochs (total 27 <= 30)","Select candidate by highest validation accuracy; restore best checkpoint by validation_loss/min","Write contract_role=proposed_method and method_components including label_smoothing"],"preserved":["3-conv-layer CNN architecture","SGD optimizer with momentum=0.9 and weight_decay=5e-4","Early stopping on validation loss (patience=5, min_delta=0)","batch_size=128","seed-injectable training","validation-only candidate selection","28x28 input resolution","PathMNIST dataset","baseline learning rate 1e-2"]}
import os
import sys
import json
import time
import random
import numpy as np

working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Seed handling - seed-injectable
if 'seed' not in globals():
    seed = 0
effective_seed = int(seed)
split_seed = 7

random.seed(effective_seed)
np.random.seed(effective_seed)
torch.manual_seed(effective_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(effective_seed)

# Hyperparameters - preserved from baseline
baseline_learning_rate = 1e-2
batch_size = 128
num_classes = 9
input_resolution = [28, 28]
weight_decay = 5e-4
momentum = 0.9
early_stopping_patience = 5
early_stopping_min_delta = 0.0

# Label smoothing tuning config
candidate_smoothing_factors = [0.05, 0.1, 0.15]
tuning_max_epochs = 5
tuning_subset_fraction = 0.20
final_max_epochs = 12  # 3*5 + 12 = 27 <= 30

# Dataset path
npz_path = "/dataset/dataset.npz"

def load_data(path):
    data = np.load(path, allow_pickle=True)
    print(f"Available keys: {list(data.files)}")
    return data

def preprocess_images(images):
    """Convert images to NCHW float32 and resize to 28x28."""
    images = images.astype(np.float32) / 255.0
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        images = np.transpose(images, (0, 3, 1, 2))
    elif images.ndim == 3:
        images = images[:, np.newaxis, :, :]
    img_tensor = torch.from_numpy(np.ascontiguousarray(images))
    if img_tensor.shape[-1] != 28 or img_tensor.shape[-2] != 28:
        img_tensor = F.interpolate(img_tensor, size=(28, 28), mode='bilinear', align_corners=False)
    return img_tensor.numpy()

def to_tensor(arr, dtype):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)

# Load data
data = load_data(npz_path)
HAS_TRAIN_SPLIT = "train_images" in data.files
print(f"HAS_TRAIN_SPLIT: {HAS_TRAIN_SPLIT}")

if HAS_TRAIN_SPLIT:
    train_images = preprocess_images(data["train_images"])
    train_labels = data["train_labels"].astype(np.int64).reshape(-1)
    val_images = preprocess_images(data["validation_images"])
    val_labels = data["validation_labels"].astype(np.int64).reshape(-1)
    print(f"Train images shape: {train_images.shape}, labels: {train_labels.shape}")
    print(f"Val images shape: {val_images.shape}, labels: {val_labels.shape}")
    print(f"Train label range: {train_labels.min()} - {train_labels.max()}")
    print(f"Val label range: {val_labels.min()} - {val_labels.max()}")
else:
    val_images = preprocess_images(data["validation_images"])
    val_labels = data["validation_labels"].astype(np.int64).reshape(-1)
    print(f"Inference-only mode. Val images shape: {val_images.shape}")

val_sample_ids = data["validation_sample_ids"]
if val_sample_ids.ndim > 1:
    val_sample_ids = val_sample_ids.reshape(-1)

# Model definition - 3 conv layers (preserved from baseline)
class LightweightCNN(nn.Module):
    def __init__(self, num_classes=9, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Experiment data tracking
experiment_data = {
    'label_smoothing_tuning': {
        'pathmnist': {
            'metrics': {'train': [], 'val': []},
            'losses': {'train': [], 'val': []},
            'predictions': [],
            'ground_truth': [],
            'epochs': [],
            'candidate_histories': {},
        }
    }
}

training_runs = []

def atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def train_candidate(smoothing_factor, train_x, train_y, val_x, val_y, seed_val, max_ep, lr=baseline_learning_rate):
    """Train a single candidate with given label smoothing factor."""
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    
    in_channels = val_x.shape[1]
    model = LightweightCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Label smoothing cross-entropy loss
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing_factor)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0
    completed_epochs = 0
    history = []
    
    print(f"\n--- Training candidate smoothing={smoothing_factor} for up to {max_ep} epochs (lr={lr}) ---")
    start_time = time.time()
    
    for epoch in range(max_ep):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'validation_loss': float(val_loss),
            'validation_metric': float(val_acc)
        })
        
        print(f'Epoch {epoch+1}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, validation_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}', flush=True)
        
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        completed_epochs = epoch + 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (patience={early_stopping_patience})", flush=True)
            break
    
    elapsed = time.time() - start_time
    print(f"Candidate smoothing={smoothing_factor} completed in {elapsed:.1f}s. Completed epochs: {completed_epochs}", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}, Best validation accuracy: {best_val_acc:.4f}", flush=True)
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'best_state_dict': best_state_dict,
        'completed_epochs': completed_epochs,
        'model': model
    }

# Determine candidate smoothing factors
if globals().get('PATH_AI_REPEAT') is not None:
    repeat_config = PATH_AI_REPEAT
    candidate_smoothing_factors = [float(repeat_config.get('parameters', {}).get('smoothing_factor', repeat_config.get('smoothing_factor', 0.1)))]
    print(f"PATH_AI_REPEAT detected: training once with smoothing={candidate_smoothing_factors[0]}")

if HAS_TRAIN_SPLIT:
    train_x_full = to_tensor(train_images, torch.float32)
    train_y_full = to_tensor(train_labels, torch.long)
    val_x = to_tensor(val_images, torch.float32)
    val_y = to_tensor(val_labels, torch.long)
    
    # Create 20% subset for tuning
    n_train = len(train_labels)
    rng = np.random.RandomState(effective_seed)
    subset_size = int(n_train * tuning_subset_fraction)
    subset_indices = rng.choice(n_train, size=subset_size, replace=False)
    train_x_subset = train_x_full[subset_indices]
    train_y_subset = train_y_full[subset_indices]
    print(f"Tuning subset: {subset_size} samples ({tuning_subset_fraction*100:.0f}% of {n_train})")
    
    candidates_results = []
    tuning_progress = {
        'schema_version': 1,
        'complete': False,
        'seed': effective_seed,
        'selection_metric': 'accuracy',
        'primary_metric': 'accuracy',
        'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'},
        'candidates_completed': [],
        'next_candidate_index': 0
    }
    atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)
    
    # Phase 1: Tune smoothing factor on 20% subset
    for idx, sf in enumerate(candidate_smoothing_factors):
        result = train_candidate(sf, train_x_subset, train_y_subset, val_x, val_y, effective_seed, max_ep=tuning_max_epochs)
        training_runs.append({'max_epochs': tuning_max_epochs, 'epochs': result['completed_epochs']})
        
        candidate_entry = {
            'smoothing_factor': float(sf),
            'validation_metric': float(result['best_val_acc']),
            'selected_epoch': result['best_epoch'],
            'completed_epochs': result['completed_epochs'],
            'history': result['history']
        }
        candidates_results.append((sf, result, candidate_entry))
        
        tuning_progress['candidates_completed'].append(candidate_entry)
        tuning_progress['next_candidate_index'] = idx + 1
        atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)
        print(f"Atomically saved tuning_progress.json after candidate {idx+1}", flush=True)
    
    # Select best candidate by highest validation accuracy
    best_candidate = max(candidates_results, key=lambda x: x[1]['best_val_acc'])
    selected_smoothing = best_candidate[0]
    print(f"\nSelected smoothing factor: {selected_smoothing} (val_acc={best_candidate[1]['best_val_acc']:.4f})")
    
    # Phase 2: Final fit on full training set with selected smoothing factor
    print(f"\n=== Final training on full dataset with smoothing={selected_smoothing} ===")
    final_result = train_candidate(selected_smoothing, train_x_full, train_y_full, val_x, val_y, effective_seed, max_ep=final_max_epochs)
    training_runs.append({'max_epochs': final_max_epochs, 'epochs': final_result['completed_epochs']})
    
    selected_model = final_result['model']
    if final_result['best_state_dict'] is not None:
        selected_model.load_state_dict(final_result['best_state_dict'])
    
    # Save selected model
    torch.save(selected_model.state_dict(), os.path.join(working_dir, 'model_checkpoint.pt'))
    print(f"\nSaved model checkpoint to working/model_checkpoint.pt (smoothing={selected_smoothing})")
    
    # Write tuning_evidence.json
    tuning_evidence = {
        'schema_version': 1,
        'complete': True,
        'seed': effective_seed,
        'selection_metric': 'accuracy',
        'primary_metric': 'accuracy',
        'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'},
        'selected_smoothing_factor': float(selected_smoothing),
        'train_subset_fraction': tuning_subset_fraction,
        'candidates': [cr[2] for cr in candidates_results],
        'final_training': {
            'smoothing_factor': float(selected_smoothing),
            'max_epochs': final_max_epochs,
            'completed_epochs': final_result['completed_epochs'],
            'best_epoch': final_result['best_epoch'],
            'best_val_loss': float(final_result['best_val_loss']),
            'best_val_acc': float(final_result['best_val_acc']),
            'history': final_result['history']
        }
    }
    atomic_write_json(os.path.join(working_dir, 'tuning_evidence.json'), tuning_evidence)
    print("Saved tuning_evidence.json")
    
    # Populate experiment_data with final training history
    for h in final_result['history']:
        experiment_data['label_smoothing_tuning']['pathmnist']['losses']['train'].append(h['train_loss'])
        experiment_data['label_smoothing_tuning']['pathmnist']['losses']['val'].append(h['validation_loss'])
        experiment_data['label_smoothing_tuning']['pathmnist']['metrics']['val'].append(h['validation_metric'])
        experiment_data['label_smoothing_tuning']['pathmnist']['epochs'].append(h['epoch'])
    
    # Store all candidate histories
    for sf, result, entry in candidates_results:
        experiment_data['label_smoothing_tuning']['pathmnist']['candidate_histories'][str(sf)] = result['history']
    experiment_data['label_smoothing_tuning']['pathmnist']['candidate_histories']['final'] = final_result['history']
    
    model = selected_model
    selected_lr = baseline_learning_rate
    selected_epochs = final_result['completed_epochs']

else:
    # Inference-only mode
    checkpoint_path = "/workspace/model_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Required checkpoint not found at {checkpoint_path}")
    in_channels = val_images.shape[1]
    model = LightweightCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Loaded checkpoint from /workspace/model_checkpoint.pt for inference-only evaluation.")
    selected_lr = baseline_learning_rate
    selected_smoothing = 0.1
    selected_epochs = 0

# Final evaluation on validation set
model.eval()
val_x_tensor = to_tensor(val_images, torch.float32)
val_y_tensor = to_tensor(val_labels, torch.long)

with torch.no_grad():
    outputs = model(val_x_tensor.to(device))
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)

predicted_np = predicted.cpu().numpy()
val_labels_np = val_labels
probabilities_np = probabilities.cpu().numpy()

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

val_accuracy = accuracy_score(val_labels_np, predicted_np)
macro_f1 = f1_score(val_labels_np, predicted_np, average='macro')
weighted_f1 = f1_score(val_labels_np, predicted_np, average='weighted')
conf_matrix = confusion_matrix(val_labels_np, predicted_np, labels=list(range(num_classes)))

print(f"\n=== Final Validation Results (Label Smoothing) ===")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

experiment_data['label_smoothing_tuning']['pathmnist']['predictions'] = predicted_np.tolist()
experiment_data['label_smoothing_tuning']['pathmnist']['ground_truth'] = val_labels_np.tolist()

np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)
print("Saved experiment_data.npy")

# Save confusion matrix plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('PathMNIST Label Smoothing - Validation Confusion Matrix')
fig.colorbar(im)
classes = [str(i) for i in range(num_classes)]
ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label', xlabel='Predicted label')
thresh = conf_matrix.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig(os.path.join(working_dir, 'pathmnist_label_smoothing_confusion_matrix.png'), dpi=150)
plt.close(fig)
print("Saved confusion matrix plot")

# Training curves
if HAS_TRAIN_SPLIT and len(experiment_data['label_smoothing_tuning']['pathmnist']['losses']['train']) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_list = experiment_data['label_smoothing_tuning']['pathmnist']['epochs']
    
    axes[0].plot(epochs_list, experiment_data['label_smoothing_tuning']['pathmnist']['losses']['train'], label='Train Loss')
    axes[0].plot(epochs_list, experiment_data['label_smoothing_tuning']['pathmnist']['losses']['val'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('PathMNIST Label Smoothing (Final) - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs_list, experiment_data['label_smoothing_tuning']['pathmnist']['metrics']['val'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('PathMNIST Label Smoothing (Final) - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, 'pathmnist_label_smoothing_training_curves.png'), dpi=150)
    plt.close(fig)
    print("Saved training curves plot")

# Write experiment_result.json
result_json = {
    'metrics': {
        'accuracy': float(val_accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': conf_matrix.tolist()
    },
    'predictions': predicted_np.tolist(),
    'targets': val_labels_np.tolist(),
    'sample_ids': val_sample_ids.tolist() if hasattr(val_sample_ids, 'tolist') else list(val_sample_ids),
    'probabilities': probabilities_np.tolist(),
    'test_data_accessed': False,
    'seed': effective_seed,
    'split_seed': split_seed,
    'selected_parameters': {
        'learning_rate': float(selected_lr),
        'smoothing_factor': float(selected_smoothing),
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum
    },
    'training_runs': training_runs,
    'early_stopping': {
        'enabled': True,
        'monitor': 'validation_loss',
        'mode': 'min',
        'patience': early_stopping_patience,
        'min_delta': early_stopping_min_delta
    },
    'max_epochs': final_max_epochs,
    'epochs_completed': selected_epochs
}

with open(os.path.join(working_dir, 'experiment_result.json'), 'w') as f:
    json.dump(result_json, f, indent=2)
print("Saved experiment_result.json")

# Write experiment_manifest.json
manifest = {
    'schema_version': 1,
    'dataset': 'PathMNIST',
    'model': 'LightweightCNN_3conv',
    'optimizer': 'SGD',
    'learning_rate': float(selected_lr),
    'epochs': final_max_epochs,
    'batch_size': batch_size,
    'seed': effective_seed,
    'split_seed': split_seed,
    'input_resolutions': [input_resolution],
    'primary_metric': 'accuracy',
    'selection_metric': 'accuracy',
    'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'},
    'hardware': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': num_classes,
    'weight_decay': weight_decay,
    'momentum': momentum,
    'early_stopping': {
        'enabled': True,
        'monitor': 'validation_loss',
        'mode': 'min',
        'patience': early_stopping_patience,
        'min_delta': early_stopping_min_delta
    },
    'training_runs': training_runs,
    'max_epochs': final_max_epochs,
    'epochs': selected_epochs,
    'selected_parameters': {
        'learning_rate': float(selected_lr),
        'smoothing_factor': float(selected_smoothing),
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum
    }
}

with open(os.path.join(working_dir, 'experiment_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
print("Saved experiment_manifest.json")

# Write contract_execution.json
contract_execution = {
    'contract_role': 'proposed_method',
    'comparison_id': 'baseline_vs_proposed',
    'seed': effective_seed,
    'split_seed': split_seed,
    'method_components': ['label_smoothing', 'cross_entropy', 'smoothing_factor', 'conv_layers', 'from_scratch', 'num_classes', 'train_subset_fraction'],
    'test_data_accessed': False,
    'has_train_split': HAS_TRAIN_SPLIT,
    'epochs_completed': selected_epochs,
    'max_epochs': final_max_epochs,
    'selected_learning_rate': float(selected_lr),
    'selected_smoothing_factor': float(selected_smoothing),
    'train_subset_fraction': tuning_subset_fraction
}

with open(os.path.join(working_dir, 'contract_execution.json'), 'w') as f:
    json.dump(contract_execution, f, indent=2)
print("Saved contract_execution.json")

print(f"\n=== LABEL SMOOTHING INTERVENTION COMPLETE ===")
print(f"Selected Smoothing Factor: {selected_smoothing}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")
print(f"Validation Weighted F1: {weighted_f1:.4f}")
print(f"Training runs: {training_runs}")
