# GBYK Dataset Documentation

This document describes the Go-Before-You-Know (GBYK) dataset structure, data organization, and important design decisions for preprocessing neural and behavioral data.

## Overview

The GBYK dataset contains synchronized neural recordings and behavioral data from primates performing a spatial navigation task. Each session represents a complete experimental recording with multiple data modalities stored in a standardized format.

## Data Organization

### Session Structure

Each session is organized as a separate dataset with the following directory structure:

```
session_name/
├── spikes/                # Raw neural spike timing data
├── spike_count/           # Binned spike count data  
├── poses/                 # 3D body keypoint coordinates
├── target/                # Task-related target signals
└── intervals/             # Trial segmentation data
```

### Data Modalities

#### Spikes (`spikes/`)

Contains raw neural spike timing data at millisecond resolution.

**Files:**
- `data.mem`: Binary memory-mapped file with spike timing data
  - **Shape:** `(n_frames, n_neurons)`
  - **Data Type:** `float32`
  - **Sampling Rate:** 1000 Hz
- `meta.yml`: Metadata dictionary (see [Metadata Format](#metadata-format))
- `meta/areas.npy`: Brain area labels for each neuron (as strings)

#### Spike Count (`spike_count/`)

Contains binned neural activity for efficient processing.

**Files:**
- `data.mem`: Spike counts in 50ms bins
  - **Shape:** `(n_frames, n_neurons)`  
  - **Data Type:** `float32`
  - **Sampling Rate:** 20 Hz (default, configurable)
- `meta.yml`: Metadata dictionary
- `meta/areas.npy`: Brain area labels (copied from spikes)

**Technical Details:**
- Binning is performed by reshaping spike data into `(n_bins, bin_size, n_neurons)` and summing over the `bin_size` dimension
- Default bin size: 50ms (1000Hz ÷ 20Hz sampling rate)
- Data is truncated to ensure complete bins: `duration = len(spikes) - (len(spikes) % period)`

#### Poses (`poses/`)

Contains 3D coordinates of body keypoints and center of mass.

**Files:**
- `data.mem`: Keypoint coordinates 
  - **Shape:** `(n_frames, 3*n_keypoints)`
  - **Data Type:** `float32`
  - **Sampling Rate:** 100 Hz
  - **Format:** Interleaved (x_1, y_1, z_1, x_2, y_2, z_2, ...)
- `meta.yml`: Metadata dictionary
- `meta/com.npy`: Center of mass coordinates `(n_frames, 3)`
- `meta/keypoints.npy`: Keypoint names (list of strings)
- `meta/skeleton.npy`: Skeleton connectivity (adjacency list)
- `meta/skeleton_reduced.npy`: Reduced version of the skeleton with head and eyes removed because of bad tracking

**Coordinate System Transformations:**

⚠️ **Critical Design Decision:** As of 2025-07-15, coordinate transformations differ between data formats:

**New Format (for spike sorted sessions):**
```python
x_axis = -x_axis + room.x_size  # X-axis flip correction
y_axis = y_axis                 # No change
z_axis = z_axis                 # No change  
```

**Old Format (for denoised sessions):**
```python
x_axis = room.y_size * (x_axis + (room.x_size / room.y_size / 2))
y_axis = room.y_size * y_axis
z_axis = room.y_size * z_axis
```

The X-axis flip in the new format corrects for coordinate system conversion errors from the original Vlad format to Irene's processing pipeline.

**Legacy Format Handling:**
- Old format datasets skip the first 5 keypoints (metadata entries)
- Keypoint selection: `coords = coords[5:]` for old format only

#### Target (`target/`)

Contains behavioral target signals indicating reward direction.

**Files:**
- `data.mem`: Target signal array
  - **Shape:** `(n_frames, 1)` 
  - **Data Type:** `float32`
  - **Sampling Rate:** 1000 Hz
  - **Encoding:**
    - `0`: Before cue presentation
    - `1`: Left reward cue active  
    - `2`: Right reward cue active

**Signal Generation:**
Target signals are derived from trial interval data and cue timing:
- Signal remains `0` until cue presentation (`cue_frame_idx`)
- From cue to trial end: `1` for left reward, `2` for right reward
- Based on `reward` field from trial metadata (may differ from `side` if monkey ignores optimal choice)

#### Intervals (`intervals/`)

Contains trial segmentation and behavioral metadata.

**Structure:** Individual YAML files for each trial (`000.yml`, `001.yml`, ...)

**Trial Types:**
- **precue**: Trials with early cue presentation
- **gbyk**: Go-before-you-know trials (cue during movement)  
- **feedback**: Feedback trials
- **homing**: Inter-trial navigation periods

**Fields per trial:**
- `side`: Movement direction (`"L"`, `"R"`)
- `cue_frame_idx`: Frame index when cue was presented
- `first_frame_idx`: Trial start frame
- `num_frames`: Trial duration in frames
- `reward`: Actual reward direction (`"L"`, `"R"`)
- `type`: Trial type (see above)
- `tier`: Data split (`"train"`, `"test"`, `"validation"`)

### Metadata Format

All `meta.yml` files contain standardized metadata:

```yaml
dtype: "float32"                    # Data type for loading .mem files
end_time: <int>                     # End frame index  
is_mem_mapped: true                 # Always true for .mem files
modality: "sequence"                # Always "sequence" for time series
n_signals: <int>                    # Number of features (neurons, keypoints, etc.)
n_timestamps: <int>                 # Number of time frames
sampling_rate: <int>                # Hz (1000 for spikes/target, 100 for poses, 20 for spike_count)
start_time: 0                       # Always 0
```

## Important Design Decisions & Caveats

### Data Processing Pipeline

1. **Trial Filtering:** Only successful, non-feedback trials are included in interval generation
2. **Homing Intervals:** Inter-trial periods are automatically created between consecutive trials (threshold: 60 seconds)
3. **Data Splits:** Train/test/validation splits are stratified by `(choice, block)` combinations (60%/20%/20%)

### Unresolved Issues:

1. **Time of Commitment (TOC):** Parser for commitment timing is not yet implemented
   - Field exists in trial data but extraction method unclear

2. **Movement Timing Encoding:**
   - How are `walk_start` and `walk_end` represented?
   - Are they relative to trial start or absolute timestamps?
   - Relationship between `walk_start`/`mt_on` and `walk_end`/`mt_off` unclear

### Brain Area Processing

**Area Labels:**
- Extracted from HDF5 references and converted to lowercase strings
- Array codes are 1-indexed in source data, converted to 0-indexed for Python
- Area assignment validation: `assert len(array_code) == spikes.shape[-1]`

**Supported Areas:**
- `"all"`: All recorded neurons
- `"m1"`: Primary motor cortex
- `"pmd"`: Premotor cortex  
- `"dlpfc"`: Dorsolateral prefrontal cortex

## Version History

- **2025-07-15:** X-axis coordinate flip correction implemented for new format
- **Legacy:** Original format with first 5 keypoints as metadata entries


## Sessions

- ### Old GBYK
    - **Bex**
      - bex_20230623_denoised: Before
    - **Ken**
      - ken_20230614_denoised: While, before
      - ken_20230618_denoised: Before condition

- ### GBYK
    - **Bex**
      - bex_20230621_spikes_sorted_SES: Before condition
      - bex_20230624_spikes_sorted_SES: Before condition
      - bex_20230629_spikes_sorted_SES: Before condition
      - bex_20230630_spikes_sorted_SES: Before condition
      - bex_20230701_spikes_sorted_SES: Before condition
      - bex_20230708_spikes_sorted_SES: While condition
    - **Ken**
      - ken_20230614_spikes_sorted_SES: While, before
      - ken_20230618_spikes_sorted_SES: Before condition
      - ken_20230622_spikes_sorted_SES: While, before, free conditions
      - ken_20230629_spikes_sorted_SES: While, before, free conditions
      - ken_20230630_spikes_sorted_SES: While condition
      - ken_20230701_spikes_sorted_SES: Before condition
      - ken_20230703_spikes_sorted_SES: While condition