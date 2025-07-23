import torch


class SoftplusReadout(torch.nn.Module):
    """
    Module for a session-specific linear layer followed by Softplus activation.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    hidden_dim : int
        Hidden dimension size.
    out_dim : dict[str, int]
        Output dimension for each session.
    n_directions : int, optional
        Number of directions (for bidirectional models). Default is 1.
    """

    def __init__(
        self,
        sessions: list[str],
        hidden_dim: int,
        out_dim: dict[str, int],
        n_directions: int = 1,
    ) -> None:
        super().__init__()
        self.softplus = torch.nn.ModuleDict(
            {
                session: torch.nn.Sequential(
                    torch.nn.Linear(
                        n_directions * hidden_dim,
                        out_dim[session],
                    ),
                    torch.nn.Softplus(),
                )
                for session in sessions
            }
        )

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        return self.softplus[session](x)


class LinearReadout(torch.nn.Module):
    """
    Module for a session-specific linear output layer.

    Parameters
    ----------
    sessions : list[str]
        List of session names.
    hidden_dim : int
        Hidden dimension size.
    out_dim : dict[str, int]
        Output dimension for each session.
    n_directions : int, optional
        Number of directions (for bidirectional models). Default is 1.
    """

    def __init__(
        self,
        sessions: list[str],
        hidden_dim: int,
        out_dim: dict[str, int],
        n_directions: int = 1,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    n_directions * hidden_dim,
                    out_dim[session],
                )
                for session in sessions
            }
        )

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        return self.linear[session](x)


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

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        return self.readout[session](x)


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
        out_dim: dict[str, int],
        n_layers: int = 1,
        readout_type: str = "linear",
    ) -> None:
        super().__init__()
        self.embedding = LinearEmbedding(sessions, in_dim, embedding_dim)

        self.core = torch.nn.ModuleList()
        if n_layers > 0:
            if embedding_dim == hidden_dim:
                self.core.extend(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(embedding_dim, embedding_dim),
                            torch.nn.ReLU(),
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
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

        if readout_type == "softplus":
            self.readout = SoftplusReadout(
                sessions, hidden_dim, out_dim, n_directions=1
            )
        elif readout_type == "linear":
            self.readout = LinearReadout(
                sessions, hidden_dim, out_dim, n_directions=1
            )
        else:
            raise ValueError(f"readout {readout_type} not implemented.")

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        y = self.embedding(x, session)
        for layer in self.core:
            y = layer(y)
        y = self.readout(y, session)
        return y


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
        out_dim: dict[str, int],
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        readout_type: str = "linear",
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
        if readout_type == "softplus":
            self.readout = SoftplusReadout(
                sessions, hidden_dim, out_dim, n_directions=n_directions
            )
        elif readout_type == "linear":
            self.readout = LinearReadout(
                sessions, hidden_dim, out_dim, n_directions=n_directions
            )
        else:
            raise ValueError(f"readout {readout_type} not implemented.")

    def forward(self, x: torch.Tensor, session: str) -> torch.Tensor:
        y = self.embedding(x, session)
        y = self.core(y)[0]
        y = self.readout(y, session)
        return y


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


def losses_map(loss_fn: str) -> torch.nn.Module:
    """
    Map a string identifier to a PyTorch loss function.

    Parameters
    ----------
    loss_fn : str
        Name of the loss function ('poisson', 'crossentropy', 'mse').

    Returns
    -------
    torch.nn.Module
        Corresponding loss function module.

    Raises
    ------
    ValueError
        If the loss function name is not recognized.
    """
    if loss_fn == "poisson":
        return torch.nn.PoissonNLLLoss(log_input=False, full=True)
    elif loss_fn == "crossentropy":
        return SequentialCrossEntropyLoss()
    elif loss_fn == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"loss function {loss_fn} not implemented.")
