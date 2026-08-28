# PATH_AI_METHOD_SPEC: {"schema_version":1,"hypothesis":"Replacing standard cross-entropy with label smoothing cross-entropy (smoothing factor selected on 20% train subset) on the same lightweight 3-conv-layer CNN improves PathMNIST validation accuracy by at least 1.0 percentage points over the baseline.","components":[{"id":"data_pipeline","category":"color_perturbation","implementation_symbols":["npz_path","train_images","validation_images","validation_sample_ids","NHWC","NCHW","normalize","train_subset_fraction"]},{"id":"model","category":"rotation","implementation_symbols":["Conv2d","BatchNorm2d","ReLU","MaxPool2d","Linear","Dropout","from_scratch","conv_layers","num_classes"]},{"id":"loss","category":"label_smoothing","implementation_symbols":["label_smoothing","smoothing_factor","cross_entropy","uniform_distribution","num_classes"]},{"id":"optimizer","category":"flip","implementation_symbols":["SGD","momentum","weight_decay"]},{"id":"training","category":"flip","implementation_symbols":["DataLoader","epochs","early_stopping","seed","paired_seeds"]},{"id":"evaluation","category":"rotation","implementation_symbols":["accuracy","macro_f1","weighted_f1","confusion_matrix","test_accuracy"]},{"id":"hyperparam_search","category":"rotation","implementation_symbols":["candidate_smoothing_factors","tuning_progress","tuning_evidence","checkpoint_selection","selected_parameters","train_subset_fraction"]}],"changes":["Implement label smoothing cross-entropy loss replacing standard CE","Search smoothing factor over candidates [0.05, 0.1, 0.15] on 20% train subset","Select best smoothing factor by validation accuracy","Final full-training-set fit with selected smoothing factor","Set contract_role to proposed_method","Record training_runs for all launches (3 candidates + 1 final)","Ensure all required intervention signals are present in code symbols and outputs"],"preserved":["3-conv-layer CNN architecture","SGD optimizer with momentum=0.9 and weight_decay=5e-4","batch_size=128","max_epochs=12","early stopping on validation loss (patience=5, min_delta=0)","seed-injectable training","validation-only selection","28x28 input resolution","PathMNIST dataset","learning_rate=1e-2 from baseline"]}
import os
working_dir = os.path.join(os.getcwd(), 'working')
os.makedirs(working_dir, exist_ok=True)

import sys
import json
import time
import random
import numpy as np

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
max_epochs = 12
num_classes = 9
input_resolution = [28, 28]
weight_decay = 5e-4
momentum = 0.9
early_stopping_patience = 5
early_stopping_min_delta = 0.0
train_subset_fraction = 0.20

# Explicit intervention signal symbols (required for detection)
conv_layers = 3
from_scratch = True
label_smoothing = True
smoothing_factor = 0.1  # default, will be selected during tuning
test_accuracy = None  # test split is sealed; not accessed

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

# Model definition - 3 conv layers (expects 28x28 input -> 3x3 after 3 max-pools)
# from_scratch=True: no external weights, trained from scratch
class LightweightCNN(nn.Module):
    """Lightweight CNN with conv_layers=3 convolutional layers, from_scratch training."""
    def __init__(self, num_classes=9, in_channels=3, conv_layers=3):
        super().__init__()
        self.conv_layers = conv_layers
        self.from_scratch = True
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

# Label smoothing cross-entropy loss
# label_smoothing=True; mixes hard target with uniform_distribution over num_classes
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy: mixes hard target with uniform_distribution."""
    def __init__(self, smoothing=0.1, num_classes=9):
        super().__init__()
        self.label_smoothing = True
        self.smoothing = smoothing
        self.smoothing_factor = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        # uniform_distribution over num_classes
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

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

def train_candidate(smoothing_factor, train_x, train_y, val_x, val_y, seed_val, lr=baseline_learning_rate, max_ep=max_epochs, use_label_smoothing=True):
    """Train a single candidate with given smoothing_factor. Returns history and best state."""
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    
    in_channels = val_x.shape[1]
    model = LightweightCNN(num_classes=num_classes, in_channels=in_channels, conv_layers=conv_layers).to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing_factor, num_classes=num_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0
    completed_epochs = 0
    history = []
    
    label_desc = f"smoothing_factor={smoothing_factor}" if use_label_smoothing else "standard_ce"
    print(f"\n--- Training candidate {label_desc} for up to {max_ep} epochs ---")
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
    print(f"Candidate {label_desc} completed in {elapsed:.1f}s. Completed epochs: {completed_epochs}", flush=True)
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
candidate_smoothing_factors = [0.05, 0.1, 0.15]

if globals().get('PATH_AI_REPEAT') is not None:
    repeat_config = PATH_AI_REPEAT
    repeat_params = repeat_config.get('parameters', {})
    selected_smoothing = float(repeat_params.get('smoothing_factor', 0.1))
    candidate_smoothing_factors = [selected_smoothing]
    print(f"PATH_AI_REPEAT detected: training once with smoothing_factor={selected_smoothing}")
    skip_search = True
else:
    selected_smoothing = 0.1
    skip_search = False

if HAS_TRAIN_SPLIT:
    train_x_full = to_tensor(train_images, torch.float32)
    train_y_full = to_tensor(train_labels, torch.long)
    val_x = to_tensor(val_images, torch.float32)
    val_y = to_tensor(val_labels, torch.long)
    
    # Create 20% subset for parameter selection using train_subset_fraction
    n_train = len(train_images)
    n_subset = int(n_train * train_subset_fraction)
    rng = np.random.RandomState(effective_seed)
    subset_indices = rng.choice(n_train, size=n_subset, replace=False)
    
    train_x_subset = train_x_full[subset_indices]
    train_y_subset = train_y_full[subset_indices]
    
    print(f"Full train set: {n_train}, Subset ({train_subset_fraction*100:.0f}%): {n_subset}")
    
    candidates_results = []
    tuning_progress = {
        'schema_version': 1,
        'complete': False,
        'seed': effective_seed,
        'selection_metric': 'accuracy',
        'primary_metric': 'accuracy',
        'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'},
        'candidates_completed': [],
        'next_candidate_index': 0,
        'train_subset_fraction': train_subset_fraction,
        'candidate_smoothing_factors': candidate_smoothing_factors,
        'label_smoothing': label_smoothing,
        'smoothing_factor': smoothing_factor,
        'conv_layers': conv_layers,
        'from_scratch': from_scratch,
        'num_classes': num_classes,
        'test_accuracy': test_accuracy
    }
    atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)
    
    if not skip_search:
        # Phase 1: Parameter selection on 20% subset
        for idx, sf in enumerate(candidate_smoothing_factors):
            result = train_candidate(sf, train_x_subset, train_y_subset, val_x, val_y, effective_seed, 
                                     lr=baseline_learning_rate, max_ep=6, use_label_smoothing=True)
            training_runs.append({'max_epochs': 6, 'epochs': result['completed_epochs']})
            
            candidate_entry = {
                'smoothing_factor': float(sf),
                'label_smoothing': True,
                'validation_metric': float(result['best_val_acc']),
                'selected_epoch': result['best_epoch'],
                'completed_epochs': result['completed_epochs'],
                'history': result['history'],
                'train_subset_fraction': train_subset_fraction
            }
            candidates_results.append((sf, result, candidate_entry))
            
            tuning_progress['candidates_completed'].append(candidate_entry)
            tuning_progress['next_candidate_index'] = idx + 1
            atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)
            print(f"Atomically saved tuning_progress.json after candidate {idx+1}", flush=True)
        
        # Select best candidate by highest validation accuracy
        best_candidate = max(candidates_results, key=lambda x: x[1]['best_val_acc'])
        selected_smoothing = best_candidate[0]
        smoothing_factor = selected_smoothing
        print(f"\nSelected smoothing_factor: {selected_smoothing} (val_acc={best_candidate[1]['best_val_acc']:.4f})")
    else:
        # Skip search, train directly with PATH_AI_REPEAT params
        result = train_candidate(selected_smoothing, train_x_subset, train_y_subset, val_x, val_y, effective_seed,
                                 lr=baseline_learning_rate, max_ep=6, use_label_smoothing=True)
        training_runs.append({'max_epochs': 6, 'epochs': result['completed_epochs']})
        candidates_results.append((selected_smoothing, result, {
            'smoothing_factor': float(selected_smoothing),
            'label_smoothing': True,
            'validation_metric': float(result['best_val_acc']),
            'selected_epoch': result['best_epoch'],
            'completed_epochs': result['completed_epochs'],
            'history': result['history'],
            'train_subset_fraction': train_subset_fraction
        }))
        tuning_progress['candidates_completed'] = [candidates_results[0][2]]
        tuning_progress['next_candidate_index'] = 1
        atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)
        smoothing_factor = selected_smoothing
    
    # Phase 2: Final full-training-set fit with selected smoothing factor
    print(f"\n=== Final full-training-set fit with smoothing_factor={selected_smoothing} ===")
    final_result = train_candidate(selected_smoothing, train_x_full, train_y_full, val_x, val_y, effective_seed,
                                   lr=baseline_learning_rate, max_ep=max_epochs, use_label_smoothing=True)
    training_runs.append({'max_epochs': max_epochs, 'epochs': final_result['completed_epochs']})
    
    selected_model = final_result['model']
    if final_result['best_state_dict'] is not None:
        selected_model.load_state_dict(final_result['best_state_dict'])
    
    # Save selected model
    torch.save(selected_model.state_dict(), os.path.join(working_dir, 'model_checkpoint.pt'))
    print(f"\nSaved model checkpoint to working/model_checkpoint.pt (selected smoothing factor={selected_smoothing})")
    
    # Write tuning_evidence.json
    tuning_evidence = {
        'schema_version': 1,
        'complete': True,
        'seed': effective_seed,
        'selection_metric': 'accuracy',
        'primary_metric': 'accuracy',
        'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'},
        'selected_smoothing_factor': float(selected_smoothing),
        'label_smoothing': True,
        'train_subset_fraction': train_subset_fraction,
        'candidate_smoothing_factors': candidate_smoothing_factors,
        'candidates': [cr[2] for cr in candidates_results],
        'conv_layers': conv_layers,
        'from_scratch': from_scratch,
        'num_classes': num_classes,
        'test_accuracy': test_accuracy
    }
    atomic_write_json(os.path.join(working_dir, 'tuning_evidence.json'), tuning_evidence)
    print("Saved tuning_evidence.json")
    
    # Populate experiment_data with final fit's history
    sel_history = final_result['history']
    for h in sel_history:
        experiment_data['label_smoothing_tuning']['pathmnist']['losses']['train'].append(h['train_loss'])
        experiment_data['label_smoothing_tuning']['pathmnist']['losses']['val'].append(h['validation_loss'])
        experiment_data['label_smoothing_tuning']['pathmnist']['metrics']['val'].append(h['validation_metric'])
        experiment_data['label_smoothing_tuning']['pathmnist']['epochs'].append(h['epoch'])
    
    # Store all candidate histories
    for sf, result, entry in candidates_results:
        experiment_data['label_smoothing_tuning']['pathmnist']['candidate_histories'][str(sf)] = result['history']
    experiment_data['label_smoothing_tuning']['pathmnist']['candidate_histories']['final'] = final_result['history']
    
    model = selected_model
    learning_rate = baseline_learning_rate
    selected_epochs = final_result['completed_epochs']
    
    # Update tuning progress as complete
    tuning_progress['complete'] = True
    tuning_progress['selected_smoothing_factor'] = float(selected_smoothing)
    atomic_write_json(os.path.join(working_dir, 'tuning_progress.json'), tuning_progress)

else:
    # Inference-only mode
    checkpoint_path = "/workspace/model_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Required checkpoint not found at {checkpoint_path}")
    in_channels = val_images.shape[1]
    model = LightweightCNN(num_classes=num_classes, in_channels=in_channels, conv_layers=conv_layers).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Loaded checkpoint from /workspace/model_checkpoint.pt for inference-only evaluation.")
    learning_rate = baseline_learning_rate
    selected_smoothing = 0.1
    smoothing_factor = selected_smoothing
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

# test_accuracy remains None - test split is sealed and not accessed
print(f"\n=== Final Validation Results ===")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"test_accuracy: {test_accuracy} (test split sealed, not accessed)")

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
        'confusion_matrix': conf_matrix.tolist(),
        'test_accuracy': test_accuracy
    },
    'predictions': predicted_np.tolist(),
    'targets': val_labels_np.tolist(),
    'sample_ids': val_sample_ids.tolist() if hasattr(val_sample_ids, 'tolist') else list(val_sample_ids),
    'probabilities': probabilities_np.tolist(),
    'test_data_accessed': False,
    'seed': effective_seed,
    'split_seed': split_seed,
    'selected_parameters': {
        'label_smoothing': label_smoothing,
        'smoothing_factor': float(selected_smoothing),
        'learning_rate': float(learning_rate),
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'conv_layers': conv_layers,
        'from_scratch': from_scratch,
        'num_classes': num_classes,
        'train_subset_fraction': train_subset_fraction,
        'test_accuracy': test_accuracy
    },
    'training_runs': training_runs,
    'early_stopping': {
        'enabled': True,
        'monitor': 'validation_loss',
        'mode': 'min',
        'patience': early_stopping_patience,
        'min_delta': early_stopping_min_delta
    },
    'max_epochs': max_epochs,
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
    'learning_rate': float(learning_rate),
    'epochs': max_epochs,
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
    'max_epochs': max_epochs,
    'epochs': selected_epochs,
    'selected_parameters': {
        'label_smoothing': label_smoothing,
        'smoothing_factor': float(selected_smoothing),
        'learning_rate': float(learning_rate),
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'conv_layers': conv_layers,
        'from_scratch': from_scratch,
        'num_classes': num_classes,
        'train_subset_fraction': train_subset_fraction,
        'test_accuracy': test_accuracy
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
    'method_components': ['label_smoothing', 'smoothing_factor', 'conv_layers', 'cross_entropy', 'from_scratch', 'num_classes', 'test_accuracy', 'train_subset_fraction'],
    'test_data_accessed': False,
    'has_train_split': HAS_TRAIN_SPLIT,
    'epochs_completed': selected_epochs,
    'max_epochs': max_epochs,
    'label_smoothing': label_smoothing,
    'selected_smoothing_factor': float(selected_smoothing),
    'smoothing_factor': float(selected_smoothing),
    'conv_layers': conv_layers,
    'from_scratch': from_scratch,
    'num_classes': num_classes,
    'test_accuracy': test_accuracy,
    'train_subset_fraction': train_subset_fraction,
    'selected_learning_rate': float(learning_rate)
}

with open(os.path.join(working_dir, 'contract_execution.json'), 'w') as f:
    json.dump(contract_execution, f, indent=2)
print("Saved contract_execution.json")

print(f"\n=== LABEL SMOOTHING EXPERIMENT COMPLETE ===")
print(f"Selected Smoothing Factor: {selected_smoothing}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")
print(f"Validation Weighted F1: {weighted_f1:.4f}")
print(f"Training runs: {training_runs}")