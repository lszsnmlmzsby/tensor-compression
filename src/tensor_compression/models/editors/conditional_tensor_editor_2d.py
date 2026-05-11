from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import nn

from tensor_compression.models.editors import EDITOR_REGISTRY


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max(1, min(max_groups, channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CharacterPromptEncoder(nn.Module):
    """A dependency-free Chinese prompt encoder based on stable character hashing."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if vocab_size < 8:
            raise ValueError("vocab_size must be at least 8.")
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        token_ids, lengths = self._batch_tokenize(prompts, device)
        embedded = self.dropout(self.embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return self.dropout(torch.cat([forward_hidden, backward_hidden], dim=-1))

    def _batch_tokenize(
        self,
        prompts: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prompts)
        token_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        lengths = torch.ones(batch_size, dtype=torch.long, device=device)
        for row, prompt in enumerate(prompts):
            ids = [self._char_to_id(char) for char in str(prompt)[: self.max_length]]
            if not ids:
                ids = [1]
            token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            lengths[row] = len(ids)
        return token_ids, lengths

    def _char_to_id(self, char: str) -> int:
        digest = hashlib.blake2b(char.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return value % (self.vocab_size - 1) + 1


class FiLMResidualBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _make_group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.film = nn.Linear(condition_dim, channels * 4)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.film(condition).chunk(4, dim=-1)
        hidden = self.norm1(inputs)
        hidden = self._apply_film(hidden, gamma1, beta1)
        hidden = self.activation(hidden)
        hidden = self.conv1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.norm2(hidden)
        hidden = self._apply_film(hidden, gamma2, beta2)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        return self.activation(inputs + hidden)

    def _apply_film(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return features * (1.0 + gamma) + beta


class LatentResidualEditor2D(nn.Module):
    def __init__(self, editor_cfg: dict) -> None:
        super().__init__()
        self.latent_grid = tuple(int(x) for x in editor_cfg["latent_grid"])
        self.latent_dim = int(editor_cfg["latent_dim"])
        self.hidden_channels = int(editor_cfg.get("latent_hidden_dim", self.latent_dim))
        self.num_res_blocks = int(editor_cfg.get("num_res_blocks", 1))
        self.condition_dim = int(editor_cfg["condition_dim"])
        self.activation_name = str(editor_cfg.get("activation", "gelu"))
        self.dropout = float(editor_cfg.get("dropout", 0.0))
        self.delta_scale = float(editor_cfg.get("latent_delta_scale", 1.0))
        self.zero_init_delta = bool(editor_cfg.get("zero_init_delta", True))
        if self.hidden_channels <= 0:
            raise ValueError("editor.model.latent_hidden_dim must be positive.")
        if self.latent_grid[0] <= 0 or self.latent_grid[1] <= 0:
            raise ValueError("editor.model.latent_grid must contain positive dimensions.")

        self.input_projection = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.hidden_channels, kernel_size=3, padding=1),
            _make_activation(self.activation_name),
        )
        blocks: list[nn.Module] = []
        for _block_idx in range(self.num_res_blocks):
            blocks.append(
                FiLMResidualBlock2D(
                    channels=self.hidden_channels,
                    condition_dim=self.condition_dim,
                    activation=self.activation_name,
                    dropout=self.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output_projection = nn.Conv2d(self.hidden_channels, self.latent_dim, kernel_size=3, padding=1)
        if self.zero_init_delta:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, latent_map: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(latent_map)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output_projection(hidden) * self.delta_scale


@EDITOR_REGISTRY.register("conditional_tensor_editor_2d")
class ConditionalTensorEditor2D(nn.Module):
    def __init__(self, compressor: nn.Module, config: dict) -> None:
        super().__init__()
        self.compressor = compressor
        editor_cfg = config["editor"]["model"]
        text_cfg = config["editor"].get("text", {})
        self.freeze_compressor = bool(config["editor"]["compressor"].get("freeze", True))
        self.use_base_reconstruction = bool(editor_cfg.get("use_base_reconstruction", True))
        self.residual_latent = bool(editor_cfg.get("residual_latent", True))
        self.detach_latent_target = bool(editor_cfg.get("detach_latent_target", True))

        prompt_encoder = CharacterPromptEncoder(
            vocab_size=int(text_cfg.get("vocab_size", 8192)),
            embed_dim=int(text_cfg.get("embed_dim", 128)),
            hidden_dim=int(text_cfg.get("hidden_dim", 128)),
            max_length=int(text_cfg.get("max_length", 128)),
            dropout=float(text_cfg.get("dropout", 0.0)),
        )
        self.prompt_encoder = prompt_encoder
        self.condition = nn.Sequential(
            nn.Linear(prompt_encoder.output_dim, int(editor_cfg["condition_dim"])),
            _make_activation(str(editor_cfg.get("activation", "gelu"))),
            nn.Dropout(float(editor_cfg.get("dropout", 0.0))),
            nn.Linear(int(editor_cfg["condition_dim"]), int(editor_cfg["condition_dim"])),
        )
        self.latent_editor = LatentResidualEditor2D(editor_cfg)
        if self.freeze_compressor:
            for parameter in self.compressor.parameters():
                parameter.requires_grad = False
            self.compressor.eval()

    def forward(self, inputs: torch.Tensor, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        if self.freeze_compressor:
            with torch.no_grad():
                latent = self.compressor.encode(inputs)
                base_reconstruction = (
                    self.compressor.decode(latent)
                    if self.use_base_reconstruction
                    else torch.zeros_like(inputs)
                )
        else:
            latent = self.compressor.encode(inputs)
            base_reconstruction = (
                self.compressor.decode(latent)
                if self.use_base_reconstruction
                else torch.zeros_like(inputs)
            )

        prompt_features = self.prompt_encoder(prompts, inputs.device)
        condition = self.condition(prompt_features)
        latent_delta = self.latent_editor(latent["latent_map"], condition)
        if self.residual_latent:
            edited_latent_map = latent["latent_map"] + latent_delta
        else:
            edited_latent_map = latent_delta
        edited_latent = {
            **latent,
            "latent_map": edited_latent_map,
            "latent_tokens": edited_latent_map.flatten(2).transpose(1, 2),
        }
        reconstruction = self.compressor.decode(edited_latent)
        return {
            "reconstruction": reconstruction,
            "delta": reconstruction - base_reconstruction,
            "base_reconstruction": base_reconstruction,
            "latent_map": latent["latent_map"],
            "edited_latent_map": edited_latent_map,
            "edited_latent_tokens": edited_latent["latent_tokens"],
            "latent_delta": latent_delta,
            "condition": condition,
        }

    def encode_target(self, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_compressor or self.detach_latent_target:
            with torch.no_grad():
                return self.compressor.encode(targets)
        return self.compressor.encode(targets)


def infer_latent_editor_config_from_compressor(
    compressor: nn.Module,
    base_model_cfg: dict,
    editor_model_cfg: dict,
) -> dict:
    resolved = dict(editor_model_cfg)
    resolved["input_size"] = base_model_cfg["input_size"]
    resolved["latent_grid"] = base_model_cfg["latent_grid"]
    resolved["latent_dim"] = getattr(compressor, "latent_dim", base_model_cfg["latent_dim"])
    resolved["out_channels"] = base_model_cfg.get("out_channels", 1)
    resolved.setdefault("activation", base_model_cfg.get("activation", "gelu"))
    return resolved


infer_decoder_config_from_compressor = infer_latent_editor_config_from_compressor
