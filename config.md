# Configuration Reference

This document provides comprehensive documentation for all configuration options available in the motorlab package.

## Core Settings

### Task and Experiment
- **`task`** *(str)*: The ML task being performed
  - Examples: `"poses_to_location"`, `"poses_to_spike_count"`, `"spike_count_to_poses"`
  - Default: `"poses_to_location"`

- **`experiment`** *(str)*: Name of the experiment/dataset  
  - Options: `"gbyk"`, `"pg"`
  - This refers to the actual dataset name, not the ML task type

- **`sessions`** *(list[str])*: List of session identifiers to include in the experiment

- **`seed`** *(int)*: Random seed for reproducibility
  - Default: `0`

## Paths Configuration

All paths are organized under the `paths` section:

- **`paths.artifacts_dir`** *(str)*: Base directory for all artifacts
  - Default: `"artifacts"`

- **`paths.data_dir`** *(str)*: Directory containing the raw data
  - Default: `"data"`

**Auto-generated paths** (based on task name):
- `checkpoint_dir`: `{artifacts_dir}/checkpoint/{task}`
- `config_dir`: `{artifacts_dir}/config/{task}`  
- `log_dir`: `{artifacts_dir}/logs/{task}`

## Data Configuration

### Input/Output Modalities
- **`data.input_modalities`** *(str or list[str])*: Input data modalities
  - Options: `"poses"`, `"speed"`, `"acceleration"`, `"spike_count"`, `"location"`
  - Can be single string or list for multi-modal inputs
  - Examples: `"poses"`, `["poses", "spike_count"]`

- **`data.output_modalities`** *(str or list[str])*: Output data modalities
  - Same options as input modalities
  - Examples: `"location"`, `["location", "spike_count"]`

### Dataset Configuration
- **`data.dataset.stride`** *(int)*: Stride for data sampling in milliseconds
  - Default: `1000` (1 second)

- **`data.dataset.concat_input`** *(bool)*: Whether to concatenate multi-modal inputs into single tensor
  - Default: `True`

- **`data.dataset.concat_output`** *(bool)*: Whether to concatenate multi-modal outputs into single tensor
  - Default: `True`

### Dataloader Configuration
- **`data.dataloader.variable_length`** *(bool)*: Whether to use variable length sequences
  - Default: `True`

- **`data.dataloader.length`** *(int)*: Fixed sequence length in milliseconds (when `variable_length=False`)
  - Mutually exclusive with `min_length`/`max_length`

- **`data.dataloader.min_length`** *(int)*: Minimum sequence length in milliseconds (when `variable_length=True`)
  - Default: `100`

- **`data.dataloader.max_length`** *(int)*: Maximum sequence length in milliseconds (when `variable_length=True`)
  - Default: `4000`

- **`data.dataloader.batch_size`** *(int)*: Batch size for training
  - Default: `64`

## Model Architecture

- **`model.architecture`** *(str)*: Neural network architecture type
  - Options: `"fc"` (fully connected), `"gru"`, `"lstm"`, `"linreg"` (linear regression)
  - Default: `"fc"`

- **`model.embedding_dim`** *(int)*: Size of embedding layer
  - Default: `256`

- **`model.hidden_dim`** *(int)*: Size of hidden layers
  - Default: `256`

- **`model.n_layers`** *(int)*: Number of layers in the network
  - Default: `3`

- **`model.readout`** *(str or dict)*: Readout layer configuration
  - Options: `"linear"`, `"softplus"`
  - Default: `"linear"`
  - For multi-modal: `{"location": "linear", "spike_count": "softplus"}`

### Optional Model Parameters
- **`model.n_classes`** *(int)*: Number of classes (for classification tasks only)
- **`model.dropout`** *(float)*: Dropout rate
- **`model.bidirectional`** *(bool)*: Whether to use bidirectional RNNs (for GRU/LSTM)

## Training Configuration

- **`training.max_epochs`** *(int)*: Maximum number of training epochs
  - Default: `500`

- **`training.loss_function`** *(str or dict)*: Loss function specification
  - Single task options: `"mse"`, `"crossentropy"`, `"poisson"`
  - Default: `"mse"`
  - Multi-task example: `{"location": "mse", "spike_count": "poisson"}`

- **`training.metric`** *(str or dict)*: Evaluation metrics
  - Options: `"mse"`, `"accuracy"`, `"correlation"`
  - Default: `"mse"`
  - Multi-modal example: `{"location": "correlation", "spike_count": "mse"}`

### Optimizer Configuration
- **`training.optimizer.type`** *(str)*: Optimizer type
  - Options: `"adam"`, `"sgd"`
  - Default: `"adam"`

- **`training.optimizer.lr`** *(float)*: Learning rate
  - Default: `1e-2`

- **`training.optimizer.weight_decay`** *(float)*: Weight decay (L2 regularization)
  - Default: `1e-4`

- **`training.optimizer.momentum`** *(float)*: Momentum (for SGD only)
  - Default: `0.9`

### Scheduler Configuration
- **`training.scheduler.type`** *(str)*: Learning rate scheduler type
  - Options: `"step_lr"`, `"cosine_annealing"`, `"reduce_on_plateau"`
  - Default: `"step_lr"`

#### StepLR Parameters
- **`training.scheduler.step_size`** *(int)*: Step size for learning rate decay
  - Default: `100`
- **`training.scheduler.gamma`** *(float)*: Multiplicative factor for learning rate decay
  - Default: `0.8`

#### CosineAnnealingLR Parameters
- **`training.scheduler.T_max`** *(int)*: Maximum number of iterations
- **`training.scheduler.eta_min`** *(float)*: Minimum learning rate
  - Default: `1e-4`

#### ReduceLROnPlateau Parameters
- **`training.scheduler.factor`** *(float)*: Factor by which learning rate is reduced
  - Default: `0.9`

### Validation Configuration
- **`training.validation.frequency`** *(int)*: How often to run validation (in epochs)
  - Default: `25`

- **`training.validation.gradient_threshold`** *(float)*: Gradient norm threshold for saving checkpoints
  - Default: `0.5`

### Early Stopping Configuration
- **`training.early_stopping.enabled`** *(bool)*: Whether to enable early stopping
  - Default: `False`

- **`training.early_stopping.patience`** *(int)*: Number of epochs to wait before stopping
  - Default: `6`

- **`training.early_stopping.min_delta`** *(float)*: Minimum change to qualify as improvement
  - Default: `0.0`

- **`training.early_stopping.gradient_threshold`** *(float)*: Gradient threshold for early stopping
  - Default: `0.5`
- **`scheduler.patience`** *(int)*: Number of epochs with no improvement after which learning rate is reduced
  - Default: `10`
- **`scheduler.min_lr`** *(float)*: Lower bound on learning rate
  - Default: `1e-5`

### Early Stopping Configuration
- **`early_stopping.enabled`** *(bool)*: Whether to enable early stopping
  - Default: `False`

- **`early_stopping.patience`** *(int)*: Number of epochs with no improvement after which training is stopped
  - Default: `6`

- **`early_stopping.min_delta`** *(float)*: Minimum change to qualify as an improvement
  - Default: `0.0`

- **`early_stopping.gradient_threshold`** *(float)*: Gradient norm threshold
  - Default: `0.5`

## Tracking and Logging

- **`tracking.stdout`** *(bool)*: Whether to print metrics to console
  - Default: `True`

- **`tracking.wandb`** *(bool)*: Whether to log to Weights & Biases
  - Default: `False`

- **`tracking.checkpoint`** *(bool)*: Whether to save model checkpoints
  - Default: `True`

- **`tracking.logging`** *(bool)*: Whether to enable file logging
  - Default: `True`

## Modality-Specific Configuration

### Location Configuration
- **`modalities.location.representation`** *(str)*: Location representation type
  - Options: `"com"` (center of mass)
  - Default: `"com"`

### Pose Configuration
- **`modalities.poses.representation`** *(str)*: Pose representation type
  - Options: `"egocentric"`, `"allocentric"`, `"centered"`, `"trunk"`
  - Default: `"egocentric"`

- **`modalities.poses.keypoints_to_exclude`** *(list[str])*: Keypoints to exclude from analysis
  - Example: `["head"]`

- **`modalities.poses.project_to_pca`** *(bool)*: Whether to project poses to PCA space
  - Default: `False`

- **`modalities.poses.divide_variance`** *(bool)*: Whether to divide by variance
  - Default: `False`

- **`modalities.poses.pcs_to_exclude`** *(dict)*: PCA components to exclude per session
  - Example: `{"session1": [0, 1, 2], "session2": [1, 2]}`

### Kinematics Configuration
- **`modalities.kinematics.representation`** *(str)*: Kinematics representation type
  - Options: `"com_vec"`, `"kps_vec"`, `"kps_mag"`, `"com_mag"`
  - Default: `"com_vec"`

### Spike Configuration
- **`modalities.spikes.brain_area`** *(str)*: Brain area to include
  - Options: `"all"`, `"m1"`, `"pmd"`, `"dlpfc"`
  - Default: `"all"`

### Intervals Configuration
- **`modalities.intervals.include_trial`** *(bool)*: Whether to include trial intervals
  - Default: `True`

- **`modalities.intervals.include_homing`** *(bool)*: Whether to include homing intervals
  - Default: `True`

- **`modalities.intervals.include_sitting`** *(bool)*: Whether to include sitting intervals
  - Default: `True`

- **`modalities.intervals.balance_intervals`** *(bool)*: Whether to balance sitting vs walking intervals
  - Default: `False`

## Configuration Examples

### Basic Single-Modal Task
```python
import motorlab as ml

config = ml.config.load_default("gbyk", ["session1", "session2"], "poses_to_location")
# Uses default settings for poses → location prediction
```

### Multi-Modal Input Task
```python
config = ml.config.load_default("gbyk", sessions, "multi_modal_prediction")
config["data"]["input_modalities"] = ["poses", "spikes"]
config["data"]["output_modalities"] = "location"
config["model"]["readout"] = "linear"
```

### Custom Training Configuration
```python
config = ml.config.load_default("pg", sessions, "custom_task")
config["training"]["max_epochs"] = 200
config["training"]["optimizer"]["lr"] = 1e-3
config["training"]["early_stopping"]["enabled"] = True
config["training"]["early_stopping"]["patience"] = 10
```

### Fixed-Length Sequences
```python
config = ml.config.load_default("gbyk", sessions, "fixed_length_task")
config["data"]["dataloader"]["variable_length"] = False
config["data"]["dataloader"]["length"] = 2000  # 2 seconds in milliseconds
```

## Advanced Configuration

### Model Loading and Transfer Learning
- **`uid`** *(str)*: Unique identifier for loading specific model
- **`load_epoch`** *(int)*: Specific epoch to load from checkpoint
- **`freeze_core`** *(bool)*: Whether to freeze core model layers during transfer learning

### Directory Overrides
These keys allow overriding the auto-generated paths:
- **`checkpoint_dir`** *(str)*: Override auto-generated checkpoint directory
- **`config_dir`** *(str)*: Override auto-generated config directory
- **`log_dir`** *(str)*: Override auto-generated log directory
- **`checkpoint_load_dir`** *(str)*: Load model from different checkpoint directory
- **`checkpoint_save_dir`** *(str)*: Save model to different checkpoint directory
- **`config_load_dir`** *(str)*: Load config from different directory
- **`config_save_dir`** *(str)*: Save config to different directory

## Notes

- All timing-related parameters (stride, min_length, max_length, length) are specified in **milliseconds**
- String modalities are automatically converted to lists during preprocessing
- Auto-generated paths follow the pattern: `{artifacts_dir}/{type}/{task}`
- Multi-modal configurations require matching readout specifications for each output modality
- **`config_save_dir`** *(str)*: Save config to different directory

## Validation Rules

1. **Mutually Exclusive Options:**
   - `dataloader.length` vs `dataloader.min_length`/`dataloader.max_length`
   - Classification (`model.n_classes`) vs regression tasks

2. **Required Combinations:**
   - When `uid` is provided, model loading parameters are required
   - When `variable_length=True`, both `min_length` and `max_length` must be specified
   - When `variable_length=False`, either `length` or `max_length` must be specified and `max_length` will be used as `length` if provided.

3. **Context Dependencies:**
   - `model.n_classes` only used for classification tasks
   - Multi-modal configurations require list inputs for modalities
   - Transfer learning requires both load and save directory specifications

## Example Configurations

### Basic Regression Task
```python
config = {
    "task": "poses_to_location",
    "experiment": "gbyk", 
    "sessions": ["session1", "session2"],
    "data": {
        "input_modalities": "poses",
        "output_modalities": "location"
    },
    "training": {
        "loss_function": "mse",
        "metric": "correlation"
    }
}
```

### Multi-Modal Classification
```python
config = {
    "task": "poses_to_behavior",
    "data": {
        "input_modalities": ["poses", "spike_count"],
        "output_modalities": "behavior"
    },
    "model": {
        "n_classes": 15,
        "readout": "linear"
    },
    "training": {
        "loss_function": "crossentropy",
        "metric": "accuracy"
    }
}
```

### Transfer Learning
```python
config = {
    "task": "transfer_task",
    "uid": "20241201_120000",  # Load pretrained model
    "freeze_core": True,
    "checkpoint_load_dir": "artifacts/checkpoint/source_task",
    "checkpoint_save_dir": "artifacts/checkpoint/target_task"
}
```
