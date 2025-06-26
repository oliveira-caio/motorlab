import torch


class SoftplusReadout(torch.nn.Module):
    def __init__(self, config, n_directions):
        super().__init__()

        self.softplus = torch.nn.ModuleDict(
            {
                session: torch.nn.Sequential(
                    torch.nn.Linear(
                        n_directions * config["model"]["hidden_dim"],
                        config["model"]["out_dim"][session],
                    ),
                    torch.nn.Softplus(),
                )
                for session in config["sessions"]
            }
        )

    def forward(self, x, session):
        return self.softplus[session](x)


class LinearReadout(torch.nn.Module):
    def __init__(self, config, n_directions):
        super().__init__()

        self.linear = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    n_directions * config["model"]["hidden_dim"],
                    config["model"]["out_dim"][session],
                )
                for session in config["sessions"]
            }
        )

    def forward(self, x, session):
        return self.linear[session](x)


class LinearEmbedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    config["model"]["in_dim"][session],
                    config["model"]["embedding_dim"],
                )
                for session in config["sessions"]
            }
        )

    def forward(self, x, session):
        return self.linear[session](x)


class FCModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = LinearEmbedding(config)

        self.core = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        config["model"]["embedding_dim"],
                        config["model"]["embedding_dim"],
                    ),
                    torch.nn.ReLU(),
                )
                for _ in range(config["model"]["n_layers"])
            ]
        )

        if config["model"]["readout"] == "softplus":
            self.readout = SoftplusReadout(config, n_directions=1)
        elif config["model"]["readout"] == "linear":
            self.readout = LinearReadout(config, n_directions=1)
        else:
            raise ValueError(
                f"readout {config['model']['readout']} not implemented."
            )

    def forward(self, x, session):
        y = self.embedding(x, session)
        for layer in self.core:
            y = layer(y)
        y = self.readout(y, session)
        return y


class GRUModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = LinearEmbedding(config)

        self.core = torch.nn.GRU(
            input_size=config["model"]["embedding_dim"],
            hidden_size=config["model"]["hidden_dim"],
            num_layers=config["model"]["n_layers"],
            batch_first=True,
            dropout=config["model"].get("dropout", 0),
            bidirectional=config["model"].get("bidirectional", True),
        )

        n_directions = 2 if config["model"].get("bidirectional", True) else 1

        if config["model"]["readout"] == "softplus":
            self.readout = SoftplusReadout(config, n_directions)
        elif config["model"]["readout"] == "linear":
            self.readout = LinearReadout(config, n_directions)
        else:
            raise ValueError(
                f"readout {config['model']['readout']} not implemented."
            )

    def forward(self, x, session):
        y = self.embedding(x, session)
        y = self.core(y)[0]
        y = self.readout(y, session)
        return y


class SequentialCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # pred: (batch, seq, n_classes), dtype: float32
        # target: (batch, seq), dtype: float32
        n_classes = pred.shape[-1]
        pred_flat = pred.view(-1, n_classes)
        target_flat = target.view(-1).long()
        return self.loss_fn(pred_flat, target_flat)


def losses_map(loss_fn):
    if loss_fn == "poisson":
        return torch.nn.PoissonNLLLoss(log_input=False, full=True)
    elif loss_fn == "crossentropy":
        return SequentialCrossEntropyLoss()
    elif loss_fn == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"loss function {loss_fn} not implemented.")
