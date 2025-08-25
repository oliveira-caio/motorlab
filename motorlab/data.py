import copy
import pathlib
import random
import warnings


import numpy as np
import torch

from motorlab import intervals, poses, spikes, utils


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
        Multi-modal data where keys are modality names and values are lists of ArrayLike with shape (n_frames, n_features)
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
        data: dict[str, list[np.ndarray]],
        input_modalities: list[str],
        output_modalities: list[str],
        stride: int,
        concat_input: bool = True,
        concat_output: bool = True,
    ) -> None:
        self.data = {
            modality: [
                torch.from_numpy(trial).to(utils.get_device())
                for trial in trials
            ]
            for modality, trials in data.items()
        }
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.stride = stride
        self.concat_input = concat_input
        self.concat_output = concat_output
        self._intervals = [
            (idx, 0, len(trial))
            for idx, trial in enumerate(self.data[self.input_modalities[0]])
        ]
        self.sequences = self._intervals.copy()

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def _prepare_data(
        self,
        trial_idx: int,
        start: int,
        end: int,
        mode: str,
    ) -> dict:
        """Extract and optionally concatenate output tensors."""
        if mode == "input":
            modalities = self.input_modalities
            concat = self.concat_input
        else:
            modalities = self.output_modalities
            concat = self.concat_output

        if concat:
            return {
                "_".join(modalities): torch.cat(
                    [self.data[m][trial_idx][start:end] for m in modalities],
                    dim=-1,
                )
            }
        else:
            return {m: self.data[m][trial_idx][start:end] for m in modalities}

    def __getitem__(self, idx):
        """Get sequence pair (input, output) at given index."""
        trial_idx, start, end = self.sequences[idx]
        _input = self._prepare_data(trial_idx, start, end, "input")
        _output = self._prepare_data(trial_idx, start, end, "output")
        return _input, _output

    def compute_sequences(self, length: int) -> None:
        """Prepare sequence extraction with given length."""
        self.sequences = [
            (trial_idx, i, i + length)
            for trial_idx, start, end in self._intervals
            for i in range(start, end - length + 1, self.stride)
        ]

    def compute_input_dims(self):
        return {
            modality: self.data[modality][0].shape[-1]
            for modality in self.input_modalities
        }

    def compute_output_dims(self):
        return {
            modality: self.data[modality][0].shape[-1]
            for modality in self.output_modalities
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
        dataset: MultiModalEntireIntervals,
        min_length: int,
        max_length: int,
        entire_trials: bool,
        shuffle: bool,
        batch_size: int,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_length = min_length
        self.max_length = max_length
        self.entire_trials = entire_trials
        self.kwargs = kwargs

    def __iter__(self):
        if not self.entire_trials:
            length = random.randint(self.min_length, self.max_length)
            self.dataset.compute_sequences(length)

        return iter(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                **self.kwargs,
            )
        )


def load_raw(
    session_dir: pathlib.Path | str,
    modalities: list[str],
    sampling_rate: int,
) -> dict[str, np.ndarray]:
    """Load raw data (dict of np.ndarrays) from a session directory."""
    session_dir = pathlib.Path(session_dir)
    data = {}

    for modality in modalities:
        if modality == "poses":
            loaded_data = poses.load(
                session_dir=session_dir,
                sampling_rate=sampling_rate,
            )
        elif modality == "spikes" or modality == "spike_count":
            loaded_data = spikes.load(
                session_dir=session_dir,
                modality=modality,
                sampling_rate=sampling_rate,
            )
        elif modality == "location":
            loaded_data = poses.load_com(
                session_dir=session_dir,
                sampling_rate=sampling_rate,
            )
        else:
            warnings.warn(f"Unknown modality {modality}, skipping it.")
            continue

        data[modality] = loaded_data

    return data


def load(
    session_dir: pathlib.Path | str,
    modalities: list[str],
    query: dict,
    sampling_rate: int,
) -> dict[str, list[np.ndarray]]:
    """
    Load queried data from a session directory.

    Parameters
    ----------
    session_dir : Path | str
        Path to the session directory
    modalities : list[str]
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
    session_dir = pathlib.Path(session_dir)
    intervals_data = intervals.load(session_dir, query, sampling_rate)
    raw_data = load_raw(session_dir, modalities, sampling_rate)
    data = {
        modality: [
            raw_data[modality][start:end] for start, end in intervals_data
        ]
        for modality in raw_data
    }
    return data


def load_all_sessions(
    data_dir: str | pathlib.Path,
    sessions: list[str],
    modalities: list[str],
    query: dict[str, str],
    sampling_rate: int,
) -> dict[str, dict[str, list[np.ndarray]]]:
    data_dir = pathlib.Path(data_dir)
    data_dict = {
        session: load(
            session_dir=data_dir / session,
            modalities=modalities,
            query=query,
            sampling_rate=sampling_rate,
        )
        for session in sessions
    }
    return data_dict


def preprocess(
    data: dict[str, list],
    cfgs: dict,
    session_dir: str | pathlib.Path,
) -> dict[str, list[np.ndarray]]:
    preprocessed_data = dict()

    for modality in data:
        if modality == "poses":
            preprocessed_data[modality] = [
                poses.preprocess(trial, cfgs.get("poses", {}), session_dir)
                for trial in data[modality]
            ]
        elif modality == "spikes" or modality == "spike_count":
            preprocessed_data[modality] = [
                spikes.preprocess(
                    trial,
                    cfgs.get("spikes", {}),
                    session_dir,
                )
                for trial in data[modality]
            ]
        elif modality == "location":
            preprocessed_data[modality] = [
                poses.preprocess_com(
                    trial,
                    cfgs.get("location", {}),
                )
                for trial in data[modality]
            ]
        else:
            warnings.warn(f"Unknown modality {modality}, skipping it.")

    return preprocessed_data


def preprocess_all_sessions(
    data: dict[str, dict[str, list[np.ndarray]]],
    cfgs: dict,
    data_dir: str | pathlib.Path,
) -> dict[str, dict[str, list[np.ndarray]]]:
    data_dir = pathlib.Path(data_dir)
    preprocessed_data = {
        session: preprocess(
            data=modalities,
            cfgs=cfgs,
            session_dir=data_dir / session,
        )
        for session, modalities in data.items()
    }
    return preprocessed_data


def preprocess_config(cfg: dict):
    stride = cfg["dataset"]["stride"]
    stride = stride * cfg["sampling_rate"] // 1000
    cfg["dataset"]["stride"] = stride

    max_length = cfg["dataloader"]["max_length"]
    max_length = max_length * cfg["sampling_rate"] // 1000
    cfg["dataloader"]["max_length"] = max_length

    min_length = cfg["dataloader"]["min_length"]
    min_length = min_length * cfg["sampling_rate"] // 1000
    cfg["dataloader"]["min_length"] = min_length


def create_dataloaders_sessions(
    sessions: list[str],
    experiment: str,
    cfg: dict,
    query: dict,
    preprocess_cfg: bool = True,
):
    if preprocess_cfg:
        preprocess_config(cfg)

    modalities = (
        cfg["dataset"]["input_modalities"] + cfg["dataset"]["output_modalities"]
    )
    dataloaders = dict()
    dl_kwargs = {
        k: v
        for k, v in cfg["dataloader"].items()
        if k not in {"batch_size", "shuffle"}
    }

    if "tier" in query and query["tier"] == "train":
        dl_kwargs["batch_size"] = cfg["dataloader"].get("batch_size", 64)
        dl_kwargs["entire_trials"] = False
        dl_kwargs["shuffle"] = cfg["dataloader"].get("shuffle", True)
    else:
        dl_kwargs["batch_size"] = 1
        dl_kwargs["entire_trials"] = True
        dl_kwargs["min_length"] = dl_kwargs["min_length"]
        dl_kwargs["shuffle"] = False

    for session in sessions:
        session_dir = pathlib.Path(cfg["dir"]) / experiment / session

        data = load(
            session_dir=session_dir,
            modalities=modalities,
            query=query,
            sampling_rate=cfg["sampling_rate"],
        )
        data = preprocess(data, cfg["modalities"], session_dir)

        dataset = MultiModalEntireIntervals(
            data,
            **{
                k: v
                for k, v in cfg["dataset"].items()
                if k not in {"output_dims", "input_dims"}
            },
        )
        dataloaders[session] = VariableLengthDataLoader(
            dataset,
            **dl_kwargs,
        )

    if "input_dims" not in cfg["dataset"]:
        cfg["dataset"]["input_dims"] = {
            session: dataloaders[session].dataset.compute_input_dims()
            for session in sessions
        }

    if "output_dims" not in cfg["dataset"]:
        cfg["dataset"]["output_dims"] = {
            session: dataloaders[session].dataset.compute_output_dims()
            for session in sessions
        }

    return dataloaders


def create_dataloaders_tiers(
    sessions: list[str],
    experiment: str,
    cfg: dict,
) -> dict:
    """
    Create VariableLengthDataLoader instances for multiple sessions.
    """
    preprocess_config(cfg)
    dataloaders = {
        tier: create_dataloaders_sessions(
            sessions=sessions,
            experiment=experiment,
            cfg=cfg,
            query={"tier": tier},
            preprocess_cfg=False,
        )
        for tier in ["train", "val", "test"]
    }
    return dataloaders
