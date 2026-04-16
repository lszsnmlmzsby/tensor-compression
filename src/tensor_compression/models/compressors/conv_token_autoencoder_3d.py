from __future__ import annotations

from torch import nn

from tensor_compression.models.compressors import MODEL_REGISTRY
from tensor_compression.models.compressors.base import BaseCompressionModel


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(num_channels: int, norm_name: str) -> nn.Module:
    norm_name = norm_name.lower()
    if norm_name == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm_name == "group":
        groups = max(1, min(8, num_channels))
        return nn.GroupNorm(groups, num_channels)
    if norm_name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm_name}")


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float, norm_name: str, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            _make_norm(channels, norm_name),
            _make_activation(activation),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            _make_norm(channels, norm_name),
        )
        self.activation = _make_activation(activation)

    def forward(self, inputs):
        return self.activation(inputs + self.block(inputs))


@MODEL_REGISTRY.register("conv_token_autoencoder_3d")
class ConvTokenAutoencoder3D(BaseCompressionModel):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        input_size = tuple(int(x) for x in model_cfg["input_size"])
        latent_grid = tuple(int(x) for x in model_cfg["latent_grid"])
        multipliers = [int(x) for x in model_cfg["channel_multipliers"]]
        levels = len(multipliers)
        down_factor = 2 ** levels
        if any(input_dim != latent_dim * down_factor for input_dim, latent_dim in zip(input_size, latent_grid)):
            raise ValueError(
                "input_size must equal latent_grid multiplied by total downsampling factor. "
                f"Got input_size={input_size}, latent_grid={latent_grid}, down_factor={down_factor}."
            )

        self.input_size = input_size
        self.latent_grid = latent_grid
        self.in_channels = int(model_cfg["in_channels"])
        self.out_channels = int(model_cfg["out_channels"])
        self.latent_dim = int(model_cfg["latent_dim"])
        base_channels = int(model_cfg["base_channels"])
        num_res_blocks = int(model_cfg["num_res_blocks"])
        dropout = float(model_cfg["dropout"])
        norm_name = str(model_cfg["norm"])
        activation = str(model_cfg["activation"])
        output_activation = str(model_cfg["output_activation"]).lower()

        encoder_layers: list[nn.Module] = [
            nn.Conv3d(self.in_channels, base_channels, kernel_size=3, padding=1),
            _make_norm(base_channels, norm_name),
            _make_activation(activation),
        ]
        current_channels = base_channels
        for mult in multipliers:
            next_channels = base_channels * mult
            encoder_layers.append(
                nn.Conv3d(current_channels, next_channels, kernel_size=3, stride=2, padding=1)
            )
            encoder_layers.append(_make_norm(next_channels, norm_name))
            encoder_layers.append(_make_activation(activation))
            for _ in range(num_res_blocks):
                encoder_layers.append(
                    ResidualBlock3D(
                        channels=next_channels,
                        dropout=dropout,
                        norm_name=norm_name,
                        activation=activation,
                    )
                )
            current_channels = next_channels
        self.encoder = nn.Sequential(*encoder_layers)
        self.to_latent = nn.Conv3d(current_channels, self.latent_dim, kernel_size=1)

        decoder_layers: list[nn.Module] = [
            nn.Conv3d(self.latent_dim, current_channels, kernel_size=3, padding=1),
            _make_activation(activation),
        ]
        reversed_multipliers = list(reversed(multipliers))
        for idx, mult in enumerate(reversed_multipliers):
            out_channels = (
                base_channels * reversed_multipliers[idx + 1]
                if idx + 1 < len(reversed_multipliers)
                else base_channels
            )
            decoder_layers.append(
                nn.ConvTranspose3d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            decoder_layers.append(_make_norm(out_channels, norm_name))
            decoder_layers.append(_make_activation(activation))
            for _ in range(num_res_blocks):
                decoder_layers.append(
                    ResidualBlock3D(
                        channels=out_channels,
                        dropout=dropout,
                        norm_name=norm_name,
                        activation=activation,
                    )
                )
            current_channels = out_channels
        decoder_layers.append(nn.Conv3d(current_channels, self.out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

        if output_activation == "identity":
            self.output_activation = nn.Identity()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

    def encode(self, inputs):
        features = self.encoder(inputs)
        latent_map = self.to_latent(features)
        latent_tokens = latent_map.flatten(2).transpose(1, 2)
        return {
            "latent_map": latent_map,
            "latent_tokens": latent_tokens,
        }

    def decode(self, latent):
        latent_map = latent["latent_map"]
        reconstruction = self.decoder(latent_map)
        return self.output_activation(reconstruction)
