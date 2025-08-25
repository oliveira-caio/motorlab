import torch

from motorlab import utils


class RNNModule(torch.nn.Module):
    def __init__(
        self,
        architecture: str,
        in_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        rnns = {
            "gru": torch.nn.GRU,
            "lstm": torch.nn.LSTM,
            "rnn": torch.nn.RNN,
        }
        self.rnn = rnns[architecture](
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return y


class FCModule(torch.nn.Module):
    def __init__(
        self,
        dims: int | list[int],
        n_layers: int,
    ):
        super().__init__()
        self.fc = torch.nn.ModuleList()
        if n_layers > 0:
            if isinstance(dims, int):
                dims = [dims] * n_layers

            self.fc.append(
                torch.nn.Sequential(
                    torch.nn.LazyLinear(dims[0]),
                    torch.nn.GELU(),
                )
            )
            self.fc.extend(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(dims[i - 1], dims[i]),
                        torch.nn.GELU(),
                    )
                    for i in range(1, n_layers)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.fc:
            y = layer(y)
        return y


class EmbeddingLayer(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        architecture: dict[str, dict[str, str]],
    ) -> None:
        super().__init__()
        self.embeddings = torch.nn.ModuleDict({})
        for session, modalities in architecture.items():
            session_dict = torch.nn.ModuleDict({})
            for modality, module in modalities.items():
                if module == "linear":
                    session_dict[modality] = torch.nn.LazyLinear(out_dim)
                else:
                    raise NotImplementedError(f"Unknown module: {module}")
            self.embeddings[session] = session_dict

    def forward(self, x: dict[str, torch.Tensor], session: str) -> torch.Tensor:
        y = torch.sum(
            torch.stack(
                [
                    self.embeddings[session][modality](x[modality])
                    for modality in self.embeddings[session]
                ],
                dim=0,
            ),
            dim=0,
        )
        return y


class ReadoutLayer(torch.nn.Module):
    """Readout layer for session-specific output modalities.

    readout_config: dict[str, str]
        {session: {modality: {readout: <architecture>, output_dimension: <output_dimension>}}}
    """

    def __init__(
        self,
        output_map: dict[str, dict[str, str]],
        output_dims: dict[str, dict[str, int]],
    ) -> None:
        super().__init__()
        self.readouts = torch.nn.ModuleDict({})
        for session in output_map.keys():
            session_dict = torch.nn.ModuleDict({})
            for modality in output_map[session].keys():
                architecture = output_map[session][modality]
                output_dim = output_dims[session][modality]
                if architecture == "linear":
                    session_dict[modality] = torch.nn.LazyLinear(output_dim)
                elif architecture == "softplus":
                    session_dict[modality] = torch.nn.Sequential(
                        torch.nn.LazyLinear(output_dim),
                        torch.nn.Softplus(),
                    )
                elif architecture == "identity":
                    session_dict[modality] = torch.nn.Identity()
                else:
                    raise NotImplementedError(
                        f"models.py: Unknown readout architecture: {architecture}"
                    )
            self.readouts[session] = session_dict

    def forward(self, x: torch.Tensor, session: str) -> dict[str, torch.Tensor]:
        return {
            modality: self.readouts[session][modality](x)
            for modality in self.readouts[session]
        }


class CoreLayer(torch.nn.Module):
    def __init__(
        self,
        cfg: dict,
    ) -> None:
        super().__init__()
        if cfg["architecture"] == "fc":
            kwargs = {k: v for k, v in cfg.items() if k != "architecture"}
        else:
            kwargs = cfg

        cores = {
            "fc": FCModule,
            "gru": RNNModule,
            "lstm": RNNModule,
            "rnn": RNNModule,
        }
        self.core = cores[cfg["architecture"]](**kwargs)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.core(x)


class LinearRegressionModel(torch.nn.Module):
    """
    Linear regression model with session-specific layers.

    Input can only be a single modality per session. Concatenate if you want to have multiple modalities. Output can be multiple modalities. It fits one linear regression per output modality then.

    Parameters
    ----------
    in_dims : dict[str, int]
        Input dimensions for each session.
    out_dims : dict[str, dict[str, int]]
        Output dimensions for each session and modality.
    """

    def __init__(
        self,
        output_dims: dict[str, dict[str, int]],
    ) -> None:
        super().__init__()
        self.linear = torch.nn.ModuleDict(
            {
                session: torch.nn.ModuleDict(
                    {
                        modality: torch.nn.LazyLinear(
                            out_dim,
                        )
                        for modality, out_dim in modalities.items()
                    }
                )
                for session, modalities in output_dims.items()
            }
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        session: str,
    ) -> dict[str, torch.Tensor]:
        y = next(iter(x.values()))
        return {
            modality: self.linear[session][modality](y)
            for modality in self.linear[session]
        }


class CoreReadoutModel(torch.nn.Module):
    def __init__(self, core_cfg: dict, readout_cfg: dict):
        super().__init__()
        self.core = CoreLayer(core_cfg)
        self.readout = ReadoutLayer(
            output_dims=readout_cfg,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        session: str,
    ) -> dict[str, torch.Tensor]:
        y = next(iter(x.values()))
        y = self.core(y, session)
        out = self.readout(y, session)
        return out


class EmbeddingCoreReadoutModel(torch.nn.Module):
    def __init__(
        self,
        embedding_cfg: dict,
        core_cfg: dict,
        readout_cfg: dict,
        output_dims: dict,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(
            out_dim=embedding_cfg["dim"],
            architecture=embedding_cfg["architecture"],
        )
        self.core = CoreLayer(core_cfg)
        self.readout = ReadoutLayer(
            output_map=readout_cfg["map"],
            output_dims=output_dims,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        session: str,
    ) -> dict[str, torch.Tensor]:
        y = self.embedding(x, session)
        y = self.core(y)
        out = self.readout(y, session)
        return out


def create(
    cfg: dict,
    output_dims: dict[str, dict[str, int]],
) -> torch.nn.Module:
    if cfg["architecture"] == "linear_regression":
        model = LinearRegressionModel(output_dims).to(utils.get_device())
    elif cfg["architecture"] == "core_readout":
        model = CoreReadoutModel(
            core_cfg=cfg["core"],
            readout_cfg=cfg["readout"],
        ).to(utils.get_device())
    elif cfg["architecture"] == "embedding_core_readout":
        model = EmbeddingCoreReadoutModel(
            embedding_cfg=cfg["embedding"],
            core_cfg=cfg["core"],
            readout_cfg=cfg["readout"],
            output_dims=output_dims,
        ).to(utils.get_device())
    else:
        raise NotImplementedError(
            f"Unknown architecture: {cfg['architecture']}"
        )
    return model
