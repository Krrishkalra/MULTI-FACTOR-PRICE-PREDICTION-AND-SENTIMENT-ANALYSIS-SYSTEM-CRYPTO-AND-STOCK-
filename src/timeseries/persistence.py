"""
Model persistence: save/load the trained model, its fitted scaler, the
configuration used to train it, the exact feature list (order matters!),
and metadata (training timestamp, dataset info, achieved metrics).

Layout written under ``models/price_prediction/<run_name>/``:
    model.pt        - torch state_dict + architecture args needed to rebuild it
    scaler.pkl       - fitted StandardScaler (joblib)
    config.json      - the Config used for this run
    feature_list.json - ordered list of feature column names
    metadata.json     - free-form metadata (metrics, timestamps, dataset info)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from ..config import Config
from .model import build_model

logger = logging.getLogger(__name__)


def save_model_bundle(
    save_dir: str | Path,
    model: torch.nn.Module,
    scaler: StandardScaler,
    config: Config,
    feature_list: List[str],
    n_features: int,
    metadata: dict[str, Any],
) -> None:
    """Persist everything needed to reload and run inference later.

    Args:
        save_dir: target directory, e.g. models/price_prediction/lstm_v1.
        model: the trained model (best-checkpoint weights already loaded).
        scaler: the StandardScaler fitted on the training split.
        config: the Config instance used for this training run.
        feature_list: ordered feature column names the model expects.
        n_features: number of input features (needed to reconstruct the
            architecture before loading the state_dict).
        metadata: arbitrary JSON-serializable metadata (metrics, dataset
            description, git commit, etc).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": config.model.model_type,
            "n_features": n_features,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "bidirectional": config.model.bidirectional,
            "num_heads": config.model.num_heads,
            "ff_dim": config.model.ff_dim,
        },
        save_dir / "model.pt",
    )

    joblib.dump(scaler, save_dir / "scaler.pkl")
    config.save(save_dir / "config.json")

    with open(save_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved model bundle to %s", save_dir)


def load_model_bundle(save_dir: str | Path, device: str = "cpu"):
    """Load a previously saved model bundle.

    Returns:
        (model, scaler, config, feature_list, metadata)
    """
    save_dir = Path(save_dir)
    checkpoint = torch.load(save_dir / "model.pt", map_location=device, weights_only=False)

    model = build_model(
        model_type=checkpoint["model_type"],
        n_features=checkpoint["n_features"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        bidirectional=checkpoint.get("bidirectional", False),
        num_heads=checkpoint.get("num_heads", 4),
        ff_dim=checkpoint.get("ff_dim", 128),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(save_dir / "scaler.pkl")
    config = Config.load(save_dir / "config.json")

    with open(save_dir / "feature_list.json", "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    metadata = {}
    metadata_path = save_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info("Loaded model bundle from %s", save_dir)
    return model, scaler, config, feature_list, metadata
