import torch


class SoftplusReadout(torch.nn.Module):
    def __init__(self, config, n_directions):
        super().__init__()

        self.readout = torch.nn.ModuleDict(
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
        return self.readout[session](x)


class LinearReadout(torch.nn.Module):
    def __init__(self, config, n_directions):
        super().__init__()

        self.readout = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    n_directions * config["model"]["hidden_dim"],
                    config["model"]["out_dim"][session],
                )
                for session in config["sessions"]
            }
        )

    def forward(self, x, session):
        return self.readout[session](x)


class LinearEmbedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = torch.nn.ModuleDict(
            {
                session: torch.nn.Linear(
                    config["model"]["in_dim"][session],
                    config["model"]["hidden_dim"],
                )
                for session in config["sessions"]
            }
        )

    def forward(self, x, session):
        return self.embedding[session](x)


class FCModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_layer = LinearEmbedding(config)

        self.core = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        config["model"]["hidden_dim"], config["model"]["hidden_dim"]
                    ),
                    torch.nn.ReLU(),
                )
                for _ in range(config["model"]["n_layers"])
            ]
        )

        if config["model"]["readout"] == "softplus":
            self.out_layer = SoftplusReadout(config, n_directions=1)
        elif config["model"]["readout"] == "linear":
            self.out_layer = LinearReadout(config, n_directions=1)
        else:
            raise ValueError(f"readout {config['model']['readout']} not implemented.")

    def forward(self, x, session):
        y = self.in_layer(x, session)
        for layer in self.core:
            y = layer(y)
        y = self.out_layer(y, session)
        return y


class GRUModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_layer = LinearEmbedding(config)

        self.core = torch.nn.GRU(
            input_size=config["model"]["hidden_dim"],
            hidden_size=config["model"]["hidden_dim"],
            num_layers=config["model"]["n_layers"],
            batch_first=True,
            dropout=config["model"].get("dropout", 0),
            bidirectional=config["model"].get("bidirectional", True),
        )

        n_directions = 2 if config["model"].get("bidirectional", True) else 1

        if config["model"]["readout"] == "softplus":
            self.out_layer = SoftplusReadout(config, n_directions)
        elif config["model"]["readout"] == "linear":
            self.out_layer = LinearReadout(config, n_directions)
        else:
            raise ValueError(f"readout {config['model']['readout']} not implemented.")

    def forward(self, x, session):
        y = self.in_layer(x, session)
        y = self.core(y)[0]
        y = self.out_layer(y, session)
        return y


class SequenceCrossEntropyLoss(torch.nn.Module):
    """
    Applies CrossEntropyLoss to (batch, seq, n_classes) predictions and (batch, seq) targets.
    Optionally supports class weights.
    """

    def __init__(self, weights=None):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, pred, target):
        # pred: (batch, seq, n_classes), dtype: float32
        # target: (batch, seq), dtype: float32
        n_classes = pred.shape[-1]
        pred_flat = pred.view(-1, n_classes)
        target_flat = target.view(-1).long()
        return self.loss_fn(pred_flat, target_flat)
