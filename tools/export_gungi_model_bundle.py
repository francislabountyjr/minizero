#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_RULESETS = {"intro", "beginner", "intermediate", "advanced"}
ACTION_SPACE_SIGNATURE = (
    "gungi:board=9x9:board_delta_planes=76:board_variants=5:"
    "drop_variants=8:piece_types=14:policy_planes=492:policy_size=39852"
)
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_BACKEND_TYPE = "gumbel_alphazero"
RUNTIME_RULES_VERSION = "rules_v1"
RUNTIME_ENCODER_VERSION = "enc_v2"
RUNTIME_ENGINE_COMPAT_VERSION = "engine_api_v1"
RUNTIME_GAME_NAME = "gungi"
RUNTIME_NETWORK_TYPE = "alphazero"
RUNTIME_BOARD_CHANNELS = 116
RUNTIME_BOARD_WIDTH = 9
RUNTIME_BOARD_HEIGHT = 9
RUNTIME_POLICY_SIZE = 39852


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchScriptMetadata:
    network_type_name: str
    game_name: str
    board_channels: int
    board_width: int
    board_height: int
    action_size: int
    discrete_value_size: int


@dataclass(frozen=True)
class SourceCheckpoint:
    config_path: Path
    checkpoint_path: Path
    source_label: str
    source_training_dir_name: str


def fnv1a_64_hex(text: str) -> str:
    hash_value = 14695981039346656037
    for byte in text.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "{:016x}".format(hash_value)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ExportError("Unsupported boolean value in config: {!r}".format(value))


def parse_config(path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def infer_model_type(config: Mapping[str, str]) -> str:
    network_type = config.get("nn_type_name", "alphazero").strip().lower()
    use_gumbel = parse_bool(config.get("actor_use_gumbel", "false"))

    if network_type == "alphazero":
        return "gaz" if use_gumbel else "az"
    if network_type == "muzero":
        return "gmz" if use_gumbel else "mz"

    raise ExportError(
        "Unsupported nn_type_name {!r}; expected alphazero or muzero.".format(network_type)
    )


def resolve_ruleset(config: Mapping[str, str], override: Optional[str]) -> str:
    ruleset = (override or config.get("env_gungi_ruleset", "")).strip().lower()
    if not ruleset or ruleset == "any":
        raise ExportError(
            "A concrete Gungi ruleset is required for app bundles. "
            "Set env_gungi_ruleset to intro/beginner/intermediate/advanced "
            "or pass --ruleset."
        )
    if ruleset not in SUPPORTED_RULESETS:
        raise ExportError(
            "Unsupported ruleset {!r}; expected one of {}.".format(
                ruleset, ", ".join(sorted(SUPPORTED_RULESETS))
            )
        )
    return ruleset


def isoformat_utc(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def call_script_method(module: torch.jit.ScriptModule, name: str) -> int:
    method = getattr(module, name, None)
    if method is None:
        raise ExportError(
            "TorchScript checkpoint {} is missing exported method {}().".format(
                module._c.qualified_name if hasattr(module, "_c") else "<module>",
                name,
            )
        )
    return method()


def inspect_torchscript_checkpoint(checkpoint_path: Path) -> TorchScriptMetadata:
    try:
        module = torch.jit.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:  # pragma: no cover - exercised via tests with valid checkpoints
        raise ExportError(
            "Failed to load TorchScript checkpoint {}: {}".format(checkpoint_path, exc)
        )

    try:
        return TorchScriptMetadata(
            network_type_name=str(call_script_method(module, "get_type_name")),
            game_name=str(call_script_method(module, "get_game_name")),
            board_channels=int(call_script_method(module, "get_num_input_channels")),
            board_width=int(call_script_method(module, "get_input_channel_width")),
            board_height=int(call_script_method(module, "get_input_channel_height")),
            action_size=int(call_script_method(module, "get_action_size")),
            discrete_value_size=int(call_script_method(module, "get_discrete_value_size")),
        )
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(
            "Failed to inspect exported metadata from {}: {}".format(checkpoint_path, exc)
        )


def ensure_app_runtime_compatible(
    metadata: TorchScriptMetadata,
    model_type: str,
    checkpoint_path: Path,
) -> None:
    if metadata.game_name != RUNTIME_GAME_NAME:
        raise ExportError(
            "Checkpoint {} reports game_name={!r}; expected {!r} for the Gungi app.".format(
                checkpoint_path, metadata.game_name, RUNTIME_GAME_NAME
            )
        )

    if metadata.network_type_name != RUNTIME_NETWORK_TYPE:
        raise ExportError(
            "Checkpoint {} is {}-trained (model type {}). "
            "The current desktop app only imports AlphaZero-shaped Gungi checkpoints.".format(
                checkpoint_path, metadata.network_type_name, model_type
            )
        )

    if metadata.board_channels != RUNTIME_BOARD_CHANNELS:
        raise ExportError(
            "Checkpoint {} exposes {} input channels; expected {}.".format(
                checkpoint_path, metadata.board_channels, RUNTIME_BOARD_CHANNELS
            )
        )
    if metadata.board_width != RUNTIME_BOARD_WIDTH or metadata.board_height != RUNTIME_BOARD_HEIGHT:
        raise ExportError(
            "Checkpoint {} exposes input shape {}x{}; expected {}x{}.".format(
                checkpoint_path,
                metadata.board_width,
                metadata.board_height,
                RUNTIME_BOARD_WIDTH,
                RUNTIME_BOARD_HEIGHT,
            )
        )
    if metadata.action_size != RUNTIME_POLICY_SIZE:
        raise ExportError(
            "Checkpoint {} exposes action_size {}; expected {}.".format(
                checkpoint_path, metadata.action_size, RUNTIME_POLICY_SIZE
            )
        )


def choose_local_config(training_dir: Path, override: Optional[Path]) -> Path:
    if override is not None:
        if not override.exists():
            raise ExportError("Config file does not exist: {}".format(override))
        return override.resolve()

    preferred = training_dir / "{}.cfg".format(training_dir.name)
    if preferred.exists():
        return preferred.resolve()

    candidates = sorted(training_dir.glob("*.cfg"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ExportError(
            "Could not find a config file in {}. Expected {} or one *.cfg file.".format(
                training_dir, preferred.name
            )
        )
    raise ExportError(
        "Multiple config files found in {}. Pass --config explicitly.".format(training_dir)
    )


def resolve_local_source(
    training_dir: Path,
    step: int,
    config_override: Optional[Path],
) -> SourceCheckpoint:
    resolved_training_dir = training_dir.resolve()
    checkpoint_path = resolved_training_dir / "model" / "weight_iter_{}.pt".format(step)
    if not checkpoint_path.exists():
        raise ExportError("Checkpoint does not exist: {}".format(checkpoint_path))

    config_path = choose_local_config(resolved_training_dir, config_override)
    return SourceCheckpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path.resolve(),
        source_label=str(resolved_training_dir),
        source_training_dir_name=resolved_training_dir.name,
    )


def select_remote_config_path(repo_files: Sequence[str], remote_training_dir: str) -> str:
    normalized_dir = PurePosixPath(remote_training_dir).as_posix().strip("/")
    expected = "{}/{}.cfg".format(normalized_dir, PurePosixPath(normalized_dir).name)
    if expected in repo_files:
        return expected

    matches = sorted(
        file_name
        for file_name in repo_files
        if file_name.startswith(normalized_dir + "/") and file_name.endswith(".cfg")
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ExportError(
            "Could not find a config file under Hugging Face training dir {!r}.".format(
                normalized_dir
            )
        )
    raise ExportError(
        "Multiple config files found under Hugging Face training dir {!r}; "
        "expected {}.".format(normalized_dir, expected)
    )


def resolve_hf_source(
    repo_id: str,
    remote_training_dir: str,
    step: int,
    revision: Optional[str],
    cache_dir: Optional[Path],
    repo_type: str,
) -> SourceCheckpoint:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ExportError(
            "Hugging Face export mode requires huggingface_hub. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type=repo_type)
    normalized_dir = PurePosixPath(remote_training_dir).as_posix().strip("/")
    checkpoint_rel = "{}/model/weight_iter_{}.pt".format(normalized_dir, step)
    if checkpoint_rel not in repo_files:
        raise ExportError(
            "Checkpoint {} was not found in Hugging Face repo {}.".format(
                checkpoint_rel, repo_id
            )
        )

    config_rel = select_remote_config_path(repo_files, normalized_dir)
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_rel,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            repo_type=repo_type,
        )
    )
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=config_rel,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            repo_type=repo_type,
        )
    )
    return SourceCheckpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        source_label="hf://{}/{}".format(repo_id, normalized_dir),
        source_training_dir_name=PurePosixPath(normalized_dir).name,
    )


def default_output_root(training_dir: Optional[Path]) -> Path:
    if training_dir is not None:
        return training_dir / "exports"
    return REPO_ROOT / "exports"


def build_manifest(
    bundle_name: str,
    display_name: str,
    ruleset: str,
    step: int,
    metadata: TorchScriptMetadata,
    model_type: str,
    checkpoint_path: Path,
    recommended_search_presets: Sequence[str],
) -> Dict[str, object]:
    return {
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "model_id": bundle_name,
        "display_name": display_name,
        "backend_type": RUNTIME_BACKEND_TYPE,
        "rules_version": RUNTIME_RULES_VERSION,
        "ruleset": ruleset,
        "encoder_version": RUNTIME_ENCODER_VERSION,
        "action_space_hash": fnv1a_64_hex(ACTION_SPACE_SIGNATURE),
        "engine_compat_version": RUNTIME_ENGINE_COMPAT_VERSION,
        "training_iteration": step,
        "input_spec": {
            "board_channels": metadata.board_channels,
            "board_width": metadata.board_width,
            "board_height": metadata.board_height,
        },
        "output_spec": {
            "policy_size": metadata.action_size,
            "value_type": "scalar" if metadata.discrete_value_size == 1 else "distribution",
        },
        "recommended_search_presets": list(recommended_search_presets),
        "artifact_files": {
            "torchscript": "model.pt",
        },
        "tags": ["gungi", "minizero", model_type, ruleset],
        "trained_at_utc": isoformat_utc(checkpoint_path.stat().st_mtime),
    }


def export_bundle(
    source: SourceCheckpoint,
    step: int,
    output_root: Path,
    ruleset_override: Optional[str] = None,
    model_id_override: Optional[str] = None,
    display_name_override: Optional[str] = None,
    force: bool = False,
    recommended_search_presets: Optional[Sequence[str]] = None,
) -> Path:
    config = parse_config(source.config_path)
    model_type = infer_model_type(config)
    ruleset = resolve_ruleset(config, ruleset_override)
    metadata = inspect_torchscript_checkpoint(source.checkpoint_path)
    ensure_app_runtime_compatible(metadata, model_type, source.checkpoint_path)

    bundle_name = model_id_override or "gungi_{}_{}_{}".format(model_type, step, ruleset)
    display_name = display_name_override or "Gungi {} step {} ({})".format(
        model_type.upper(), step, ruleset
    )
    presets = list(recommended_search_presets or ["standard_review"])

    bundle_root = output_root.resolve() / bundle_name
    if bundle_root.exists():
        if not force:
            raise ExportError(
                "Bundle output already exists: {}. Pass --force to replace it.".format(
                    bundle_root
                )
            )
        shutil.rmtree(bundle_root)

    bundle_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.checkpoint_path, bundle_root / "model.pt")
    manifest = build_manifest(
        bundle_name=bundle_name,
        display_name=display_name,
        ruleset=ruleset,
        step=step,
        metadata=metadata,
        model_type=model_type,
        checkpoint_path=source.checkpoint_path,
        recommended_search_presets=presets,
    )
    (bundle_root / "model.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return bundle_root


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a MiniZero Gungi training checkpoint into a desktop-app-ready "
            "bundle containing model.pt and model.json."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--training-dir",
        type=Path,
        help="Local MiniZero training folder containing model/weight_iter_<step>.pt.",
    )
    source_group.add_argument(
        "--hf-repo-id",
        help="Hugging Face repo id to download a training folder checkpoint from.",
    )
    parser.add_argument(
        "--hf-training-dir",
        help="Path inside the Hugging Face repo that points to the training folder.",
    )
    parser.add_argument(
        "--hf-revision",
        help="Optional Hugging Face revision, branch, or commit.",
    )
    parser.add_argument(
        "--hf-repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hugging Face repo type when --hf-repo-id is used.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        help="Optional Hugging Face cache dir override.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Explicit local config path. Only valid with --training-dir.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Checkpoint step number to export, e.g. 200 for weight_iter_200.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Root directory to write bundle folders into. Defaults to <training-dir>/exports or <repo>/exports.",
    )
    parser.add_argument(
        "--ruleset",
        help="Override the manifest ruleset when the config is missing or uses env_gungi_ruleset=any.",
    )
    parser.add_argument(
        "--model-id",
        help="Override the default model_id / bundle folder name.",
    )
    parser.add_argument(
        "--display-name",
        help="Override the manifest display_name.",
    )
    parser.add_argument(
        "--recommended-search-preset",
        dest="recommended_search_presets",
        action="append",
        default=None,
        help="Repeat to add recommended_search_presets. Defaults to standard_review.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination bundle directory if it already exists.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.hf_repo_id and not args.hf_training_dir:
        parser.error("--hf-training-dir is required with --hf-repo-id.")
    if args.config is not None and args.training_dir is None:
        parser.error("--config can only be used with --training-dir.")

    try:
        if args.training_dir is not None:
            source = resolve_local_source(args.training_dir, args.step, args.config)
            output_root = args.output_dir or default_output_root(args.training_dir.resolve())
        else:
            source = resolve_hf_source(
                repo_id=args.hf_repo_id,
                remote_training_dir=args.hf_training_dir,
                step=args.step,
                revision=args.hf_revision,
                cache_dir=args.hf_cache_dir,
                repo_type=args.hf_repo_type,
            )
            output_root = args.output_dir or default_output_root(None)

        bundle_root = export_bundle(
            source=source,
            step=args.step,
            output_root=output_root,
            ruleset_override=args.ruleset,
            model_id_override=args.model_id,
            display_name_override=args.display_name,
            force=args.force,
            recommended_search_presets=args.recommended_search_presets,
        )
    except ExportError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    print("Created model bundle at {}".format(bundle_root))
    print("  source: {}".format(source.source_label))
    print("  checkpoint: {}".format(source.checkpoint_path))
    print("  manifest: {}".format(bundle_root / "model.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
