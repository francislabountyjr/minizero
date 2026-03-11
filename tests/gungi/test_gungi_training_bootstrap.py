from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TRAINING_RUNTIME_FILES = [
    REPO_ROOT / "minizero" / "actor" / "actor_group.cpp",
    REPO_ROOT / "minizero" / "actor" / "base_actor.cpp",
    REPO_ROOT / "minizero" / "console" / "mode_handler.cpp",
    REPO_ROOT / "minizero" / "learner" / "train.py",
    REPO_ROOT / "minizero" / "zero" / "zero_server.cpp",
    REPO_ROOT / "scripts" / "zero-server.sh",
    REPO_ROOT / "scripts" / "zero-worker.sh",
    REPO_ROOT / "tools" / "quick-run.sh",
]

GUNGI_CONFIG_FILES = [
    REPO_ROOT / "gungi-b176e8-dirty.cfg",
    REPO_ROOT / "gungi_debug" / "gungi_debug.cfg",
    REPO_ROOT / "gungi_gaz" / "gungi_gaz.cfg",
    REPO_ROOT / "gungi_gaz_main" / "gungi_gaz_main.cfg",
    REPO_ROOT / "gungi_gaz_smoke" / "gungi_gaz_smoke.cfg",
    REPO_ROOT / "command.txt",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _existing_gungi_config_files():
    return [path for path in GUNGI_CONFIG_FILES if path.exists()]


def test_training_runtime_has_no_gungi_fen_override_calls():
    forbidden_tokens = ("load_fen", "loadFen(", "setInitialFen(")
    hits = []

    for path in TRAINING_RUNTIME_FILES:
        text = _read_text(path)
        for token in forbidden_tokens:
            if token in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert hits == []


def test_current_gungi_training_configs_do_not_override_start_position():
    forbidden_tokens = ("load_fen", "start_fen", "initial_fen", "opening_fen")
    hits = []

    for path in _existing_gungi_config_files():
        text = _read_text(path)
        for token in forbidden_tokens:
            if token in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {token}")

    assert hits == []


def test_current_gungi_training_configs_pin_advanced_ruleset():
    config_files = _existing_gungi_config_files()
    assert config_files

    missing = []
    for path in config_files:
        if "env_gungi_ruleset=advanced" not in _read_text(path):
            missing.append(str(path.relative_to(REPO_ROOT)))

    assert missing == []
