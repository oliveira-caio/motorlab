# Configuration Reference

This document provides comprehensive documentation for all configuration options available in the motorlab package.

## General

- **`task`** *(str, optional)*: The ML task being performed
  - Used only if `save` is `True`.
  - Examples: `"poses_to_location"`, `"poses_to_spike_count"`, `"spike_count_to_poses"`

- **`experiment`** *(str)*: Name of the experiment/dataset
  - Options: `"old_gbyk"`, `"gbyk"`, `"pg"`

- **`sessions`** *(list[str], optional)*: List of session identifiers to include in the experiment
  - See `docs/gbyk.md` and `docs/pg.md` for more details.
  - Default: it will be set using the `experiment` value by calling the `get` function inside `sessions.py`

- **`seed`** *(int, optional)*: Random seed for reproducibility
  - Default: `0`

- **`artifacts_dir`** *(int, optional)*: Directory where the artifacts should be created.



## Logger

- **`logger.dir`** *(str, optional)*: Directory where the checkpoints and predictions will be saved.
  - Created only if `save` is `True`.
  - The path will be created as `<artifacts_dir>/<task>/<run_id>`

- **`logger.checkpoints`** *(bool, optional)*: Whether the best checkpoint should be logged.
  - Default: `True`

- **`logger.predictions`** *(bool, optional)*: Whether the predictions should be logged.
  - Default: `True`

- **`logger.plots`** *(bool, optional)*: Whether the plots should be logged.
  - Default: `True`



## Data

- **`data.dir`** *(str, optional)*: Directory containing the raw data
  - Default: `"data"`

- **`data.sampling_rate`** *(int, optional)*: The sampling rate (in Hz) all the modalities will be converted to.
  - Default: `20`

### Dataset

- **`data.dataset.input_modalities`** *(list[str])*: Input data modalities
  - Options: `"poses"`, `"speed"`, `"acceleration"`, `"spike_count"`, `"location"`
  - Examples: `["poses"]`, `["poses", "spike_count"]`

- **`data.dataset.output_modalities`** *(list[str])*: Output data modalities
  - Same options as input modalities
  - Examples: `["location"]`, `["location", "spike_count"]`

- **`data.dataset.stride`** *(int, optional)*: Stride for data sampling in milliseconds
  - Default: `1000` (1 second)

- **`data.dataset.concat_input`** *(bool, optional)*: Whether to concatenate multi-modal inputs into single tensor
  - Default: `True`

- **`data.dataset.concat_output`** *(bool, optional)*: Whether to concatenate multi-modal outputs into single tensor
  - Default: `True`

### Dataloader

- **`data.dataloader.min_length`** *(int, optional)*: Minimum sequence length in milliseconds (when `variable_length=True`)
  - Default: `100`

- **`data.dataloader.max_length`** *(int, optional)*: Maximum sequence length in milliseconds (when `variable_length=True`)
  - Default: `4000`

- **`data.dataloader.batch_size`** *(int, optional)*: Batch size for trainer
  - Default: `64`

You can use all the default arguments for dataloaders, such as `num_workers`.

#### Intervals

- **`data.intervals.include_trial`** *(bool, optional)*: Whether to include trial intervals
  - Default: `True`

- **`data.intervals.include_homing`** *(bool, optional)*: Whether to include homing intervals
  - Default: `True`

- **`data.intervals.include_sitting`** *(bool, optional)*: Whether to include sitting intervals
  - Default: `True`

- **`data.intervals.balance_intervals`** *(bool, optional)*: Whether to balance sitting vs walking intervals
  - Default: `False`


### Modalities

#### Location

- **`data.modalities.location.representation`** *(str, optional)*: Location representation type
  - Options: `"com"` (center of mass), `"tiles"` (discretizes the location into tiles)
  - Default: `"com"`

#### Poses

- **`data.modalities.poses.representation`** *(str, optional)*: The name of the representation for the poses
  - Convenient way to keep track of which preprocessing steps we're performing in the poses.
  - Example: `egocentric + PCA` would be a representation where we use egocentric coordinates combined with PCA embedding.
  - Default: `egocentric`.

- **`data.modalities.poses.coordinates`** *(str, optional)*: The coordinate system of the poses
  - Options: `"egocentric"`, `"allocentric"`, `"centered"`, `"trunk"`, `"head"`
  - For more details, check the notebook `investigations/representations.ipynb` [!todo]
  - Default: `egocentric`

- **`data.modalities.poses.residualize`** *(bool, optional)*: Whether to apply residualization from poses to location task.
  - Default: `False`

- **`data.modalities.poses.project_to_pcs`** *(str, optional)*: Whether to project the poses to (a subset of) the principal components space.
  - The subspace of the principal components is obtained by fitting models to predict location and pruning the best components. See `analysis_pca.ipynb` for the details.
  - Options: `"all"`, `"loose"`, `"medium"`, `"strict"`, `None`
  - Default: `None`

- **`data.modalities.poses.keypoints_to_exclude`** *(list[str], optional)*: List of keypoints to exclude.
  - It'll be applied after the change in representation
  - Default: `None`
  - Example: `["head"]`

- **`data.modalities.poses.dims_to_exclude`** *(dict, optional)*: Which dimensions to exclude per session.
  - Used when one wants to exclude some principal components, for example
  - It'll be applied last in the preprocessing stage
  - Default: `None`
  - Example: `{"session1": [0, 1, 2], "session2": [1, 2]}`

- **`data.modalities.poses.skeleton_type`** *(str, optional)*: Which skeleton is being used.
  - The `gbyk` experiment can use the `normal` and the `reduced` skeleton while the `pg` experiment can use all
  - Options: `normal`, `reduced`, `extended`
  - Default: `normal`

#### Kinematics

- **`data.modalities.kinematics.representation`** *(str, optional)*: Kinematics representation type
  - Whether we use the speed and acceleration of the center of mass or all keypoints
  - Whether we use the magnitude of the speed and acceleration or the vector
  - Options: `"com_vec"`, `"kps_vec"`, `"kps_mag"`, `"com_mag"`
  - Default: `"com_vec"`

#### Spikes

Used for both spikes and spike count

- **`data.modalities.spikes.brain_areas`** *(str or list[str], optional)*: Brain areas that will be used
  - Options: `"all"`, `"m1"`, `"pmd"`, `"dlpfc"`
  - Default: `"all"`
  - Example: `["m1", "pmd"]`



## Model

- **`model.architecture`** *(str)*: Neural network architecture type
  - Options: `"lr"` (Linear Regression), `"cr"`, `"ecr"` (Embedding + Core + Readout)


#### Embedding

- **`model.embedding.architecture`** *(str or dict[str, str], optional)*: Which embedding architecture it will use
  - Options for the architectures: `"linear"`, `"pca"`
  - Example for multiple modalities: `{"spike_count": "linear", "poses": "pca"}`
  - Default: `"linear"`

- **`model.embedding.dim`** *(int, optional)*: Dimension after applying embedding
  - For the `"pca"` architecture it will first project to the components and then apply a linear layer to `dim`
  - Default: `256`

#### Core

- **`model.core.architecture`** *(str)*: The neural network that will be used
  - Options: `"fc"` (fully connected), `"gru"`, `"lstm"`, `"rnn"`

- **`model.core.dims`** *(int or list[int], optional)*: Dimension of core "hidden" layers
  - If an integer is provided, all hidden layers will have the same dimension
  - Important: the list must have the same length as the number of layers
  - Default: `256`

- **`model.core.n_layers`** *(int, optional)*: Number of layers in the network
  - Default: `3`

#### Readout

- **`model.readout.map`** *(str or dict[str, str])*: Readout map configuration
  - Options for the architectures: `"linear"`, `"softplus"`
  - Example for multiple modalities: `{"location": "linear", "spike_count": "softplus"}`
  - Default: `"linear"`


### Optional Model Parameters
- **`model.n_classes`** *(int, optional)*: Number of classes (for classification tasks only)
- **`model.dropout`** *(float, optional)*: Dropout rate
- **`model.bidirectional`** *(bool, optional)*: Whether to use bidirectional RNNs (for RNN/GRU/LSTM)

## Trainer Configuration

- **`trainer.max_epochs`** *(int, optional)*: Maximum number of trainer epochs
  - Default: `1000`

- **`trainer.loss_fns`** *(str or dict)*: Loss function specification
  - Single task options: `"mse"`, `"crossentropy"`, `"poisson"`
  - Multi-task example: `{"location": "mse", "spike_count": "poisson"}`
  - Single-task example: `"mse"`

- **`trainer.metrics`** *(str or dict)*: Evaluation metrics
  - Options: `"mse"`, `"accuracy"`, `"correlation"`, `None`
  - Multi-modal example: `{"location": "correlation", "spike_count": "mse"}`
  - Uni-modal example: `"mse"`

- **`trainer.validation_frequency`** *(int, optional)*: How often to run validation (in epochs)
  - Default: `25`

- **`trainer.log_plots`** *(bool, optional)*: Whether to log the plots during training.
  - Default: `True`

### Optimizer

- **`trainer.optimizer.algorithm`** *(str, optional)*: Optimizer type
  - Options: `"adam"`, `"adamw"`, `"sgd"`
  - Default: `"adam"`

- **`trainer.optimizer.lr`** *(float, optional)*: Learning rate
  - Default: `1e-2`

It is possible to provide all the normal parameters one can pass to an optimizer.

### Scheduler Configuration

- **`trainer.scheduler.method`** *(str, optional)*: Learning rate scheduler method.
  - Options: `"step_lr"`, `"cosine_annealing"`, `"reduce_on_plateau"`, `None`.
  - Default: `None`, which will use a constant learning rate.

#### StepLR Parameters
- **`trainer.scheduler.step_size`** *(int, optional)*: Step size for learning rate decay
  - Default: `100`
- **`trainer.scheduler.gamma`** *(float, optional)*: Multiplicative factor for learning rate decay
  - Default: `0.8`

#### CosineAnnealingLR Parameters
- **`trainer.scheduler.T_max`** *(int, optional)*: Maximum number of iterations
  - Default: `100`
- **`trainer.scheduler.eta_min`** *(float, optional)*: Minimum learning rate
  - Default: `1e-5`

#### ReduceLROnPlateau Parameters
- **`trainer.scheduler.factor`** *(float, optional)*: Factor by which learning rate is reduced
  - Default: `0.9`
- **`trainer.scheduler.patience`** *(int, optional)*: Number of epochs with no improvement after which learning rate is reduced
  - Default: `10`
- **`trainer.scheduler.min_lr`** *(float, optional)*: Lower bound on learning rate
  - Default: `1e-5`

### Early Stopper Configuration

- **`trainer.early_stopper.enabled`** *(bool, optional)*: Whether to enable early stopping
  - Default: `True`

- **`trainer.early_stopper.patience`** *(int, optional)*: Number of validation epochs to wait before stopping
  - Default: `6`(which translates to `150` trainer epochs if you use the default `validation.frequency` value)

- **`trainer.early_stopper.min_delta`** *(float, optional)*: Minimum change to qualify as improvement
  - Default: `0.0`


## Other Configurations

- **`uid`** *(str, optional)*: Unique identifier for loading a model
  - It will download the configuration and the trained model from Weights & Biases
  - Default: `None`

- **`freeze_core`** *(bool, optional)*: Whether to freeze core model layers during transfer learning
  - Default: `False`
