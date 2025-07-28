import torch


class Readout(torch.nn.Module):
    """
    Flexible module for multiple session-specific readouts with configurable activation types.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    hidden_dim : int
        Hidden dimension size.
    out_dim : dict[str, dict[str, int]]
        Output dimensions for each session and each readout name.
        Example: { 'session1': { 'linear': 2, 'softplus': 3 }, ... }
    readout_map : dict[str, str]
        Mapping from readout name to activation type ('linear', 'softplus', etc).
        Example: { 'linear': 'linear', 'softplus': 'softplus' }
    n_directions : int, optional
        Number of directions (for bidirectional models). Default is 1.
    """

    def __init__(
        self,
        sessions: list[str],
        hidden_dim: int,
        out_dim: dict[str, dict[str, int]],
        readout_map: dict[str, str],
        n_directions: int = 1,
    ) -> None:
        super().__init__()

        self.readouts = torch.nn.ModuleDict({})
        for session in sessions:
            session_dict = {}
            for modality, readout in readout_map.items():
                if readout == "linear":
                    session_dict[modality] = torch.nn.Linear(
                        n_directions * hidden_dim,
                        out_dim[session][modality],
                    )
                elif readout == "softplus":
                    session_dict[modality] = torch.nn.Sequential(
                        torch.nn.Linear(
                            n_directions * hidden_dim,
                            out_dim[session][modality],
                        ),
                        torch.nn.Softplus(),
                    )
                else:
                    raise ValueError(f"unknown readout type: {readout}.")
            self.readouts[session] = torch.nn.ModuleDict(session_dict)

    def forward(self, x: torch.Tensor, session: str) -> dict:
        return {
            modality: readout(x)
            for modality, readout in self.readouts[session].items()
        }


class LinearEmbedding(torch.nn.Module):
    """
    Module for a session-specific linear embedding layer.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    in_dim : dict[str, int]
        Input dimension for each session.
    embedding_dim : int
        Embedding dimension size.
    """

    def __init__(
        self, sessions: list[str], in_dim: dict[str, int], embedding_dim: int
    ) -> None:
        super().__init__()
        self.linear = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    in_dim[session],
                    embedding_dim,
                )
                for session in sessions
            }
        )

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        return self.linear[session](x)


class LinRegModel(torch.nn.Module):
    """
    Linear regression model with session-specific readout layers.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    in_dim : dict[str, int]
        Input dimension for each session.
    out_dim : dict[str, int]
        Output dimension for each session.
    """

    def __init__(
        self,
        sessions: list[str],
        in_dim: dict[str, int],
        out_dim: dict[str, int],
    ) -> None:
        super().__init__()
        self.readout = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    in_dim[session],
                    out_dim[session],
                )
                for session in sessions
            }
        )

    def forward(
        self, x: torch.Tensor, session: str, modality: str = "output"
    ) -> dict:
        return {modality: self.readout[session](x)}


class FCModel(torch.nn.Module):
    """
    Fully connected feedforward model with configurable layers and readout.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    in_dim : dict[str, int]
        Input dimension for each session.
    embedding_dim : int
        Embedding dimension size.
    hidden_dim : int
        Hidden dimension size.
    out_dim : dict[str, int]
        Output dimension for each session.
    n_layers : int, optional
        Number of hidden layers. Default is 1.
    readout_type : str, optional
        Type of readout ('softplus' or 'linear'). Default is 'linear'.
    """

    def __init__(
        self,
        sessions: list[str],
        in_dim: dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        out_dim: dict,
        n_layers: int,
        readout_map: dict,
    ) -> None:
        super().__init__()
        self.embedding = LinearEmbedding(sessions, in_dim, embedding_dim)

        self.core = torch.nn.ModuleList()
        if n_layers > 0:
            self.core.append(
                torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, hidden_dim),
                    torch.nn.ReLU(),
                )
            )
            self.core.extend(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                    )
                    for _ in range(n_layers - 1)
                ]
            )

        self.readout = Readout(
            sessions,
            hidden_dim,
            out_dim,
            readout_map,
            n_directions=1,
        )

    def forward(self, x: torch.Tensor, session: str) -> dict:
        y = self.embedding(x, session)
        for layer in self.core:
            y = layer(y)
        out = self.readout(y, session)
        return out


class GRUModel(torch.nn.Module):
    """
    GRU-based model with session-specific embedding and readout layers.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    in_dim : dict[str, int]
        Input dimension for each session.
    embedding_dim : int
        Embedding dimension size.
    hidden_dim : int
        Hidden dimension size.
    out_dim : dict[str, int]
        Output dimension for each session.
    n_layers : int, optional
        Number of GRU layers. Default is 1.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    bidirectional : bool, optional
        Whether to use bidirectional GRU. Default is True.
    readout_type : str, optional
        Type of readout ('softplus' or 'linear'). Default is 'linear'.
    """

    def __init__(
        self,
        sessions: list[str],
        in_dim: dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        out_dim: dict,
        n_layers: int,
        readout_map: dict,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = LinearEmbedding(sessions, in_dim, embedding_dim)
        self.core = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        n_directions = 2 if bidirectional else 1

        self.readout = Readout(
            sessions,
            hidden_dim,
            out_dim,
            readout_map,
            n_directions=n_directions,
        )

    def forward(self, x: torch.Tensor, session: str) -> dict:
        y = self.embedding(x, session)
        y = self.core(y)[0]
        out = self.readout(y, session)
        return out


class SequentialCrossEntropyLoss(torch.nn.Module):
    """
    Cross-entropy loss for sequential data (flattens batch and sequence dimensions).
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.shape[-1]
        pred_flat = pred.view(-1, n_classes)
        target_flat = target.view(-1).long()
        return self.loss_fn(pred_flat, target_flat)


class DictLoss(torch.nn.Module):
    """
    Computes the total loss for Readout outputs using a modality-to-loss mapping.

    Parameters
    ----------
    loss_map : dict[str, str]
        Dictionary mapping modality names to loss function names.
        Example: {"spike_count": "poisson", "position": "mse"}
    """

    def __init__(self, loss_map: dict[str, str]) -> None:
        super().__init__()

        loss_dict = {
            "poisson": torch.nn.PoissonNLLLoss(log_input=False, full=True),
            "crossentropy": SequentialCrossEntropyLoss(),
            "mse": torch.nn.MSELoss(),
        }

        self.loss_fns = torch.nn.ModuleDict({})
        for modality, loss_name in loss_map.items():
            if loss_name not in loss_dict:
                raise ValueError(f"loss function {loss_name} not implemented.")
            self.loss_fns[modality] = loss_dict[loss_name]

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        """
        Compute total loss across all modalities.

        Parameters
        ----------
        outputs : dict
            Dictionary of modality -> predicted tensor from MultiReadout
        targets : dict
            Dictionary of modality -> target tensor

        Returns
        -------
        torch.Tensor
            Total loss (sum of all modality losses)
        """
        losses = [
            self.loss_fns[modality](outputs[modality], targets[modality])
            for modality in outputs
        ]
        total_loss = torch.stack(losses).sum()
        return total_loss
