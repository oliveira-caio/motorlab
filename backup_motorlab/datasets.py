import random
import warnings

from pathlib import Path

import numpy as np
import torch

from motorlab.modalities import poses, spikes, location, intervals


class MultiModalEntireIntervals(torch.utils.data.Dataset):
    """
    A unified dataset for multi-modal neural data that stores entire intervals
    and extracts variable-length sequences on-the-fly.

    Assumptions
    -----------
    - `data`: a dictionary with multiple modalities and modalities are aligned
      with the same frequency.
    - `intervals`: aligned and at the same frequency of the data.

    How it works
    ------------
    The dataset stores complete trials and generates sequences on-demand:

    1. Stores entire intervals as variable-length sequences
    2. Uses extract_sequences() to define sequence length and stride
    3. Generates (start, end) tuples for all possible sequences
    4. Extracts sequences on-the-fly during __getitem__

    Parameters
    ----------
    data : dict
        Multi-modal data where keys are modality names and values are arrays
        with shape (n_frames, n_features)
    intervals : list[LabeledInterval]
        Labeled intervals defining valid data segments
    input_modalities : list[str]
        Input modality names
    output_modalities : list[str]
        Output modality names
    stride : int, optional
        Frames to advance between sequences, by default 20
    concat_input : bool, optional
        Concatenate input modalities into single tensor, by default True
    concat_output : bool, optional
        Concatenate output modalities into single tensor, by default True
    """

    def __init__(
        self,
        session_dir: str,
        query: dict,
        input_modalities: list[str],
        output_modalities: list[str],
        sampling_rate: int,
        stride: int = 20,
        concat_input: bool = True,
        concat_output: bool = True,
    ) -> None:
        self.data = load_dict(
            session_dir=session_dir,
            modalities_list=input_modalities + output_modalities,
            query=query,
            sampling_rate=sampling_rate,
        )
        self.intervals = intervals
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.stride = stride
        self.concat_input = concat_input
        self.concat_output = concat_output
        self.sequences = [
            (interval.start, interval.end) for interval in intervals
        ]

    def extract_sequences(self, seq_len: int) -> None:
        """Prepare sequence extraction with given length."""
        self.sequences = [
            (i, i + seq_len)
            for interval in self.intervals
            for i in range(
                interval.start, interval.end - seq_len + 1, self.stride
            )
        ]

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get sequence pair (input, output) at given index."""
        start, end = self.sequences[idx]
        return self._prepare_inputs(start, end), self._prepare_outputs(
            start, end
        )

    def _prepare_inputs(self, start: int, end: int) -> dict:
        """Extract and optionally concatenate input tensors."""
        if self.concat_input:
            return {
                "_".join(self.input_modalities): torch.cat(
                    [
                        torch.tensor(self.data[m][start:end])
                        for m in self.input_modalities
                    ],
                    dim=-1,
                )
            }
        return {
            m: torch.tensor(self.data[m][start:end])
            for m in self.input_modalities
        }

    def _prepare_outputs(self, start: int, end: int) -> dict:
        """Extract and optionally concatenate output tensors."""
        if self.concat_output:
            return {
                "_".join(self.output_modalities): torch.cat(
                    [
                        torch.tensor(self.data[m][start:end])
                        for m in self.output_modalities
                    ],
                    dim=-1,
                )
            }
        return {
            m: torch.tensor(self.data[m][start:end])
            for m in self.output_modalities
        }


class VariableLengthDataLoader:
    """
    DataLoader that samples new sequence length each epoch.

    Parameters
    ----------
    dataset : MultiModalEntireTrials
        Dataset that supports extract_sequences()
    min_length : int
        Minimum sequence length
    max_length : int
        Maximum sequence length
    variable_length : bool
        Whether to use variable-length sequences
    batch_size : int, optional
        Batch size, by default 64
    shuffle : bool, optional
        Whether to shuffle data, by default True
    **kwargs
        Additional DataLoader arguments
    """

    def __init__(
        self,
        dataset: MultiModalEntireTrials,
        min_length: int,
        max_length: int,
        variable_length: bool,
        batch_size: int = 64,
        shuffle: bool = True,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.variable_length = variable_length
        self.min_length = min_length
        self.max_length = max_length

    def __iter__(self):
        length = (
            random.randint(self.min_length, self.max_length)
            if self.variable_length
            else self.min_length
        )
        self.dataset.extract_sequences(length)

        return iter(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                **self.kwargs,
            )
        )


def load_datasets(
    data_dict: dict,
    intervals: dict,
    input_modalities: list[str],
    output_modalities: list[str],
    stride: int = 20,
    concat_input: bool = True,
    concat_output: bool = True,
) -> dict:
    """
    Create Dataset instances for multiple sessions.

    Parameters
    ----------
    data_dict : dict
        Session names to session data mappings
    intervals : dict
        Session names to interval lists mappings
    input_modalities : list[str]
        Input modality names
    output_modalities : list[str]
        Output modality names
    stride : int, optional
        Sequence extraction stride, by default 20
    concat_input : bool, optional
        Concatenate input modalities, by default True
    concat_output : bool, optional
        Concatenate output modalities, by default True

    Returns
    -------
    dict
        Session names to dataset objects mappings
    """
    return {
        session: MultiModalEntireTrials(
            data_dict[session],
            intervals[session],
            input_modalities,
            output_modalities,
            stride,
            concat_input,
            concat_output,
        )
        for session in data_dict
    }


def load_dataloaders(
    datasets: dict,
    min_length: int,
    max_length: int,
    variable_length: bool,
    batch_size: int = 64,
    shuffle: bool = True,
    test_mode: bool = False,
    **kwargs,
) -> dict:
    """
    Create DataLoaders for multiple session datasets.

    Parameters
    ----------
    datasets : dict
        Session names to dataset objects mappings
    dataloader_config : dict
        Variable length configuration
    batch_size : int, optional
        Batch size, by default 64
    shuffle : bool, optional
        Shuffle data, by default True
    is_test : bool, optional
        Test mode (entire trials, batch_size=1), by default False
    **kwargs
        Additional DataLoader arguments

    Returns
    -------
    dict
        Session names to DataLoader objects mappings
    """
    if test_mode:
        return {
            session: torch.utils.data.DataLoader(
                datasets[session], batch_size=1, shuffle=False
            )
            for session in datasets
        }

    return {
        session: VariableLengthDataLoader(
            datasets[session],
            min_length,
            max_length,
            variable_length,
            batch_size,
            shuffle,
            **kwargs,
        )
        for session in datasets
    }


def load_dict(
    session_dir: Path | str,
    modalities_list: list[str],
    query: dict[str, str],
    sampling_rate: int,
) -> dict[str, list[np.ndarray]]:
    """
    Load data from a session directory.

    Parameters
    ----------
    session_dir : Path | str
        Path to the session directory
    modalities_list : list[str]
        List of modalities to load
    query : dict[str, str]
        Query parameters for loading data
    sampling_rate : int
        Sampling rate for the loaded data

    Returns
    -------
    dict[str, list[np.ndarray]]
        Loaded data organized by modality
    """
    session_dir = Path(session_dir)
    data = {}

    intervals_data = intervals.load(session_dir, query, sampling_rate)

    for modality in modalities_list:
        modality_dir = session_dir / modality
        if not modality_dir.exists():
            warnings.warn(
                f"Modality {modality_dir} does not exist, skipping it."
            )
            continue

        if modality == "poses":
            poses_data = poses.load(
                session_dir=session_dir,
                sampling_rate=sampling_rate,
            )
            data["poses"] = poses_data[intervals_data]
        elif modality == "spikes" or modality == "spike_count":
            spikes_data = spikes.load(
                session_dir=session_dir,
                modality=modality,
                sampling_rate=sampling_rate,
            )
            data["spikes"] = spikes_data[intervals_data]
        elif modality == "location":
            location_data = location.load(
                session_dir=session_dir,
                sampling_rate=sampling_rate,
            )
            data["location"] = location_data[intervals_data]
        else:
            warnings.warn(f"Unknown modality {modality}, skipping it.")

    return data


def load_all_dicts(
    sessions: list[str],
    modalities_dict: dict,
    query: dict[str, str],
    sampling_rate: int,
) -> dict[str, dict[str, list[np.ndarray]]]:
    return {
        session: load_dict(
            session_dir=session,
            modalities_dict=modalities_dict,
            query=query,
            sampling_rate=sampling_rate,
        )
        for session in sessions
    }
