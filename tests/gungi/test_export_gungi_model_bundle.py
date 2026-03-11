import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.export_gungi_model_bundle import (
    ExportError,
    export_bundle,
    infer_model_type,
    resolve_local_source,
)


BOARD_CHANNELS = 116
POLICY_PLANES = 492
POLICY_SIZE = 39852


class SmokeGungiModel(nn.Module):
    def __init__(self, network_type_name="alphazero"):
        super(SmokeGungiModel, self).__init__()
        self._network_type_name = network_type_name
        self._game_name = "gungi"
        self._num_input_channels = BOARD_CHANNELS
        self._input_channel_height = 9
        self._input_channel_width = 9
        self._action_size = POLICY_SIZE
        self._discrete_value_size = 1
        self.stem = nn.Conv2d(self._num_input_channels, 8, kernel_size=1)
        self.policy_head = nn.Conv2d(8, POLICY_PLANES, kernel_size=1)
        self.value_head = nn.Conv2d(8, 1, kernel_size=1)
        self.value_fc = nn.Linear(81, 1)

    @torch.jit.export
    def get_type_name(self):
        return self._network_type_name

    @torch.jit.export
    def get_game_name(self):
        return self._game_name

    @torch.jit.export
    def get_num_input_channels(self):
        return self._num_input_channels

    @torch.jit.export
    def get_input_channel_height(self):
        return self._input_channel_height

    @torch.jit.export
    def get_input_channel_width(self):
        return self._input_channel_width

    @torch.jit.export
    def get_action_size(self):
        return self._action_size

    @torch.jit.export
    def get_discrete_value_size(self):
        return self._discrete_value_size

    def forward(self, state):
        hidden = F.relu(self.stem(state))
        policy_logit = self.policy_head(hidden).flatten(1)
        policy = torch.softmax(policy_logit, dim=1)
        value_hidden = F.relu(self.value_head(hidden)).flatten(1)
        value = torch.tanh(self.value_fc(value_hidden))
        return {
            "policy_logit": policy_logit,
            "policy": policy,
            "value": value,
        }


def write_torchscript_model(path: Path, network_type_name="alphazero"):
    model = SmokeGungiModel(network_type_name=network_type_name).eval()
    torch.jit.script(model).save(str(path))


def create_training_dir(
    tmp_path: Path,
    *,
    training_name: str,
    step: int,
    actor_use_gumbel: bool,
    nn_type_name: str,
    ruleset: str,
) -> Path:
    training_dir = tmp_path / training_name
    (training_dir / "model").mkdir(parents=True)
    write_torchscript_model(
        training_dir / "model" / "weight_iter_{}.pt".format(step),
        network_type_name=nn_type_name,
    )
    (training_dir / "{}.cfg".format(training_name)).write_text(
        "\n".join(
            [
                "actor_use_gumbel={}".format("true" if actor_use_gumbel else "false"),
                "nn_type_name={}".format(nn_type_name),
                "env_gungi_ruleset={}".format(ruleset),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return training_dir


def test_infer_model_type_covers_all_supported_training_modes():
    assert infer_model_type({"nn_type_name": "alphazero", "actor_use_gumbel": "false"}) == "az"
    assert infer_model_type({"nn_type_name": "alphazero", "actor_use_gumbel": "true"}) == "gaz"
    assert infer_model_type({"nn_type_name": "muzero", "actor_use_gumbel": "false"}) == "mz"
    assert infer_model_type({"nn_type_name": "muzero", "actor_use_gumbel": "true"}) == "gmz"


def test_export_bundle_creates_app_ready_layout(tmp_path: Path):
    training_dir = create_training_dir(
        tmp_path,
        training_name="gungi_gaz_main",
        step=200,
        actor_use_gumbel=True,
        nn_type_name="alphazero",
        ruleset="advanced",
    )

    source = resolve_local_source(training_dir, 200, None)
    bundle_dir = export_bundle(source=source, step=200, output_root=tmp_path / "exports")

    assert bundle_dir.name == "gungi_gaz_200_advanced"
    assert (bundle_dir / "model.pt").exists()
    assert not (bundle_dir / "weight_iter_200.pt").exists()

    manifest = json.loads((bundle_dir / "model.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "gungi_gaz_200_advanced"
    assert manifest["display_name"] == "Gungi GAZ step 200 (advanced)"
    assert manifest["backend_type"] == "gumbel_alphazero"
    assert manifest["ruleset"] == "advanced"
    assert manifest["training_iteration"] == 200
    assert manifest["artifact_files"]["torchscript"] == "model.pt"
    assert manifest["recommended_search_presets"] == ["standard_review"]
    assert manifest["input_spec"] == {
        "board_channels": BOARD_CHANNELS,
        "board_width": 9,
        "board_height": 9,
    }
    assert manifest["output_spec"] == {
        "policy_size": POLICY_SIZE,
        "value_type": "scalar",
    }
    assert "notes" not in manifest


def test_export_bundle_rejects_muzero_checkpoints_for_current_app_runtime(tmp_path: Path):
    training_dir = create_training_dir(
        tmp_path,
        training_name="gungi_gmz_main",
        step=200,
        actor_use_gumbel=True,
        nn_type_name="muzero",
        ruleset="advanced",
    )

    source = resolve_local_source(training_dir, 200, None)

    with pytest.raises(ExportError, match="only imports AlphaZero-shaped Gungi checkpoints"):
        export_bundle(source=source, step=200, output_root=tmp_path / "exports")
