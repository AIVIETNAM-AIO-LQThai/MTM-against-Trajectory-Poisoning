from __future__ import annotations

import os
import pathlib

from pathlib import Path
from typing import Union

import torch

from src.methods.dt.model import DecisionTransformer


def load_dt_checkpoint_compat(
    model: DecisionTransformer,
    checkpoint_path: Union[
        str,
        Path,
    ],
    device: Union[
        str,
        torch.device,
    ] = "cpu",
) -> dict:
    """
    Load a DT checkpoint across both:

      modern training runtime
          Torch 2.x / newer Transformers

    and:

      reference evaluation runtime
          Torch 1.8.1 / Transformers 4.5.1

    Transformers 4.5.1 creates fixed GPT-2
    causal-attention buffers that are not present
    in checkpoints produced by the newer runtime.

    Those buffers are NOT learned model parameters.
    """

    # Checkpoints trained on Windows may contain pathlib.WindowsPath
    # objects inside saved runtime arguments. Linux cannot instantiate
    # WindowsPath during unpickling, even though those paths are only
    # metadata and are not required to load the model weights.
    #
    # Temporarily map WindowsPath -> PosixPath while unpickling, then
    # restore pathlib immediately afterward.
    original_windows_path = pathlib.WindowsPath

    try:
        if os.name != "nt":
            pathlib.WindowsPath = pathlib.PosixPath

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
    finally:
        pathlib.WindowsPath = original_windows_path

    # Normalize historical checkpoint progress naming.
    #
    # Group 4C stress checkpoints originally used "step",
    # while the evaluator expects "update".
    if "update" not in checkpoint and "step" in checkpoint:
        checkpoint["update"] = checkpoint["step"]

    if "step" not in checkpoint and "update" in checkpoint:
        checkpoint["step"] = checkpoint["update"]

    if "step" not in checkpoint or "update" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain training progress "
            "under either 'step' or 'update'."
        )

    if int(checkpoint["step"]) != int(checkpoint["update"]):
        raise RuntimeError(
            "Checkpoint has inconsistent 'step' and 'update' values: "
            f"{checkpoint['step']} vs {checkpoint['update']}"
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    checkpoint_state = (
        checkpoint["model_state_dict"]
    )

    result = model.load_state_dict(
        checkpoint_state,
        strict=False,
    )

    # --------------------------------------------------
    # These are the ONLY compatibility differences we
    # permit in the legacy Transformers 4.5.1 runtime.
    # --------------------------------------------------

    allowed_missing = {
        f"transformer.h.{layer}.attn.{buffer}"
        for layer in range(
            model.transformer.config.n_layer
        )
        for buffer in (
            "bias",
            "masked_bias",
        )
    }

    missing = set(
        result.missing_keys
    )

    unexpected = set(
        result.unexpected_keys
    )

    # --------------------------------------------------
    # Strong safety check:
    # no model parameter may be missing.
    # --------------------------------------------------

    model_parameter_names = set(
        dict(
            model.named_parameters()
        ).keys()
    )

    checkpoint_keys = set(
        checkpoint_state.keys()
    )

    missing_parameters = (
        model_parameter_names
        - checkpoint_keys
    )

    if missing_parameters:
        raise RuntimeError(
            "Checkpoint is missing model parameters:\n"
            f"{sorted(missing_parameters)}"
        )

    # --------------------------------------------------
    # Modern runtime:
    #
    #     missing == set()
    #
    # Legacy Transformers 4.5.1:
    #
    #     missing == allowed_missing
    #
    # Anything else is considered incompatible.
    # --------------------------------------------------

    if missing not in (
        set(),
        allowed_missing,
    ):
        raise RuntimeError(
            "Unexpected missing checkpoint keys.\n"
            f"Allowed legacy buffers:\n"
            f"{sorted(allowed_missing)}\n"
            f"Actually missing:\n"
            f"{sorted(missing)}"
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            f"{sorted(unexpected)}"
        )

    model.eval()

    return checkpoint