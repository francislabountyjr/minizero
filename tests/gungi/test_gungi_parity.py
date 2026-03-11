import json
import os
import random
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CPP_BUILD_ROOT = REPO_ROOT / "build" / "gungi"
PY_REF_SRC = Path(os.environ.get("GUNGI_PY_SRC", r"E:\Python\gungi.py\src"))
JS_DIR = Path(os.environ.get("GUNGI_JS_DIR", r"E:\Python\gungi.js"))


def _has_cpp_module(build_dir: Path) -> bool:
    if not build_dir.exists():
        return False
    return any(build_dir.glob("minizero_py*.pyd")) or any(build_dir.glob("minizero_py*.so"))


def _resolve_cpp_build_dir() -> Path:
    configured = os.environ.get("MINIZERO_GUNGI_BUILD_DIR")
    if configured:
        return Path(configured)

    candidates = [
        DEFAULT_CPP_BUILD_ROOT / "Release",
        DEFAULT_CPP_BUILD_ROOT / "RelWithDebInfo",
        DEFAULT_CPP_BUILD_ROOT / "Debug",
        DEFAULT_CPP_BUILD_ROOT / "MinSizeRel",
        DEFAULT_CPP_BUILD_ROOT,
    ]
    for candidate in candidates:
        if _has_cpp_module(candidate):
            return candidate
    return DEFAULT_CPP_BUILD_ROOT


CPP_BUILD_DIR = _resolve_cpp_build_dir()
MODE_RULESETS = {
    "0": "intro",
    "1": "beginner",
    "2": "intermediate",
    "3": "advanced",
}
ADVANCED_START_POSITION = (
    "9/9/9/9/9/9/9/9/9 "
    "M1G1I1J2W2N3R2S2F2D4C1A2K1T1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 wb 1"
)
IN_CHECK_POSITION = (
    "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3 "
    "J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"
)
SOURCE_OF_TRUTH_SELF_CHECK_MOVE = "\u4e2d(1-6-1)(2-6-1)"


def _cpp_import_error() -> Optional[str]:
    if not CPP_BUILD_DIR.exists():
        return f"Gungi build dir not found at {CPP_BUILD_DIR}. Build MiniZero with GAME_TYPE=GUNGI first."
    if not _has_cpp_module(CPP_BUILD_DIR):
        return (
            f"No minizero_py module found under {CPP_BUILD_DIR}. "
            "Build MiniZero with GAME_TYPE=GUNGI first."
        )
    try:
        if str(CPP_BUILD_DIR) not in sys.path:
            sys.path.insert(0, str(CPP_BUILD_DIR))
        import minizero_py  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-specific
        return f"Failed to import minizero_py from {CPP_BUILD_DIR}: {exc}"
    return None


def _python_ref_error() -> Optional[str]:
    if not PY_REF_SRC.exists():
        return f"Python Gungi reference not found at {PY_REF_SRC}"
    return None


def _js_error() -> Optional[str]:
    if not JS_DIR.exists():
        return f"JS Gungi reference not found at {JS_DIR}"
    missing = [tool for tool in ("node", "pnpm") if shutil.which(tool) is None]
    if missing:
        return "JS parity tests require: " + ", ".join(sorted(missing))
    return None


@lru_cache
def _cpp():
    if str(CPP_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(CPP_BUILD_DIR))
    import minizero_py
    assert minizero_py.load_config_string("env_gungi_ruleset=advanced")

    return minizero_py


def _ruleset_from_fen(fen: str) -> str:
    mode = fen.split()[3]
    return MODE_RULESETS[mode]


def _set_cpp_ruleset(ruleset: str) -> None:
    assert _cpp().load_config_string(f"env_gungi_ruleset={ruleset}")


@lru_cache
def _py_ref():
    if str(PY_REF_SRC) not in sys.path:
        sys.path.insert(0, str(PY_REF_SRC))
    import gungi
    from gungi import Gungi
    from gungi.fen import ADVANCED_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION, INTRO_POSITION
    from gungi.move_gen import would_be_in_check_after_move
    from gungi.utils import piece

    return {
        "module": gungi,
        "Gungi": Gungi,
        "ADVANCED_POSITION": ADVANCED_POSITION,
        "BEGINNER_POSITION": BEGINNER_POSITION,
        "INTERMEDIATE_POSITION": INTERMEDIATE_POSITION,
        "INTRO_POSITION": INTRO_POSITION,
        "piece": piece,
        "would_be_in_check_after_move": would_be_in_check_after_move,
    }


@lru_cache
def _ensure_js_build() -> None:
    subprocess.run(
        ["pnpm", "build"],
        cwd=JS_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def _run_js_sequence(moves, start_fen):
    _ensure_js_build()
    script = f"""
    import {{ Gungi }} from './dist/index.js';
    const g = new Gungi({json.dumps(start_fen)});
    const moves = {json.dumps(moves)};
    const snapshots = [];
    for (const move of moves) {{
      g.move(move);
      snapshots.push({{ fen: g.fen(), history: g.history() }});
    }}
    console.log(JSON.stringify({{ fen: g.fen(), history: g.history(), snapshots }}));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=JS_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return json.loads(result.stdout.strip())


def _inspect_js_position(fen, square=None):
    _ensure_js_build()
    script = f"""
    import {{ Gungi }} from './dist/index.js';
    const g = new Gungi({json.dumps(fen)});
    const square = {json.dumps(square)};
    console.log(JSON.stringify({{
      fen: g.fen(),
      turn: g.turn(),
      moveNumber: g.moveNumber(),
      drafting: g.getDraftingRights(),
      moves: square ? g.moves({{ square }}) : g.moves(),
      inCheck: g.inCheck(),
      isCheckmate: g.isCheckmate(),
      isStalemate: g.isStalemate(),
      isInsufficientMaterial: g.isInsufficientMaterial(),
      isFourfoldRepetition: g.isFourfoldRepetition(),
      isDraw: g.isDraw(),
      isGameOver: g.isGameOver(),
    }}));
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=JS_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return json.loads(result.stdout.strip())


def _cpp_game(fen):
    _set_cpp_ruleset(_ruleset_from_fen(fen))
    return _cpp().GungiEnv(fen)


def _captured_summary_from_cpp(game):
    counts = {}
    for move in game.history_verbose():
        for captured in move["captured"]:
            key = (captured["color"], captured["type"])
            counts[key] = counts.get(key, 0) + 1
    return sorted(f"{color}|{ptype}|{count}" for (color, ptype), count in counts.items())


def _compact_cpp_move(move):
    return {
        "san": move["san"],
        "piece": move["piece"],
        "color": move["color"],
        "from": move["from"],
        "to": move["to"],
        "type": move["type"],
        "draftFinished": move["draftFinished"],
        "captured": sorted(
            f"{captured['square']}|{captured['tier']}|{captured['color']}|{captured['type']}"
            for captured in move["captured"]
        ),
        "before": move["before"],
        "after": move["after"],
    }


def _compact_py_move(move):
    return {
        "san": move.san,
        "piece": move.piece,
        "color": move.color,
        "from": move.from_,
        "to": move.to,
        "type": move.type,
        "draftFinished": move.draft_finished,
        "captured": sorted(
            f"{captured.square}|{captured.tier}|{captured.color}|{captured.type}"
            for captured in (move.captured or [])
        ),
        "before": move.before,
        "after": move.after,
    }


def _compact_cpp_state(game):
    return {
        "fen": game.fen(),
        "turn": game.turn(),
        "moveNumber": game.move_number(),
        "drafting": dict(game.drafting()),
        "history": list(game.history()),
        "captured": _captured_summary_from_cpp(game),
        "flags": {
            "inCheck": game.in_check(),
            "isCheckmate": game.is_checkmate(),
            "isStalemate": game.is_stalemate(),
            "isInsufficientMaterial": game.is_insufficient_material(),
            "isFourfoldRepetition": game.is_fourfold_repetition(),
            "isDraw": game.is_draw(),
            "isGameOver": game.is_game_over(),
        },
    }


def _compact_py_state(game):
    return {
        "fen": game.fen(),
        "turn": game.turn(),
        "moveNumber": game.move_number(),
        "drafting": game.get_drafting_rights(),
        "history": game.history(),
        "captured": sorted(f"{hp.color}|{hp.type}|{hp.count}" for hp in game.captured()),
        "flags": {
            "inCheck": game.in_check(),
            "isCheckmate": game.is_checkmate(),
            "isStalemate": game.is_stalemate(),
            "isInsufficientMaterial": game.is_insufficient_material(),
            "isFourfoldRepetition": game.is_fourfold_repetition(),
            "isDraw": game.is_draw(),
            "isGameOver": game.is_game_over(),
        },
    }


def _pick_cpp_draft_end_move(game, color: str) -> str:
    for move in game.legal_moves_verbose():
        if move["color"] == color and move["draftFinished"] is True:
            return move["san"]
    raise AssertionError(f"no draft-ending {color} move found")


cpp_only = pytest.mark.skipif(
    _cpp_import_error() is not None,
    reason=_cpp_import_error() or "",
)

cpp_with_python_ref = pytest.mark.skipif(
    _cpp_import_error() is not None or _python_ref_error() is not None,
    reason=_cpp_import_error() or _python_ref_error() or "",
)


@cpp_only
def test_cpp_default_reset_uses_advanced_deployment_start():
    _set_cpp_ruleset("advanced")
    game = _cpp().GungiEnv()

    assert game.fen() == ADVANCED_START_POSITION
    assert game.in_draft() is True
    assert dict(game.drafting()) == {"b": True, "w": True}
    assert list(game.history()) == []

    first_move = _pick_cpp_draft_end_move(game, "w")
    assert game.move(first_move)
    assert game.fen() != ADVANCED_START_POSITION

    game.reset()

    assert game.fen() == ADVANCED_START_POSITION
    assert game.in_draft() is True
    assert dict(game.drafting()) == {"b": True, "w": True}
    assert list(game.history()) == []


@cpp_only
def test_cpp_default_start_transitions_from_draft_to_live_play():
    _set_cpp_ruleset("advanced")
    game = _cpp().GungiEnv()

    white_end = _pick_cpp_draft_end_move(game, "w")
    assert game.move(white_end)
    assert game.in_draft() is True
    assert dict(game.drafting()) == {"b": True, "w": False}

    black_end = _pick_cpp_draft_end_move(game, "b")
    assert game.move(black_end)
    assert game.in_draft() is False
    assert dict(game.drafting()) == {"b": False, "w": False}

    legal_moves = list(game.legal_moves_verbose())
    assert any(move["from"] for move in legal_moves)
    assert any(not move["from"] for move in legal_moves)


@cpp_only
def test_cpp_accepts_source_of_truth_self_check_move():
    game = _cpp_game(IN_CHECK_POSITION)

    assert SOURCE_OF_TRUTH_SELF_CHECK_MOVE in list(game.moves())
    assert SOURCE_OF_TRUTH_SELF_CHECK_MOVE in list(game.legal_moves())
    assert game.act_san(SOURCE_OF_TRUTH_SELF_CHECK_MOVE)


@cpp_with_python_ref
@pytest.mark.skipif(_js_error() is not None, reason=_js_error() or "")
def test_cpp_matches_js_for_reference_sequence_and_terminal_positions():
    ref = _py_ref()
    Gungi = ref["Gungi"]
    piece = ref["piece"]
    start_fen = ref["BEGINNER_POSITION"]

    py_game = Gungi(start_fen)
    sequence = []
    first_route = sorted(move.san for move in py_game.moves(verbose=True) if move.type == "route")[0]
    sequence.append(first_route)
    py_game.move(first_route)
    second_route = sorted(move.san for move in py_game.moves(verbose=True) if move.type == "route")[0]
    sequence.append(second_route)
    py_game.move(second_route)
    sequence.append(sorted(move.san for move in py_game.moves(verbose=True) if move.type == "arata")[0])

    cpp_game = _cpp_game(start_fen)
    cpp_snapshots = []
    for move in sequence:
        assert cpp_game.move(move)
        cpp_snapshots.append({"fen": cpp_game.fen(), "history": list(cpp_game.history())})

    js_state = _run_js_sequence(sequence, start_fen)
    assert cpp_game.fen() == js_state["fen"]
    assert list(cpp_game.history()) == js_state["history"]
    assert cpp_snapshots == js_state["snapshots"]

    tricky_positions = [
        "3img3/1ra1|n:G|1as1/d1fwdwf2/9/8d/9/D1FWDWF1D/1SA1N1AR1/4MI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
        "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3",
        "8m/9/7DD/7C1/9/9/9/9/M8 -/- b 3 - 1",
        "m8/9/9/9/9/9/9/9/8M -/- w 3 - 1",
        "9/9/9/9/9/9/9/9/4M4 D1/m1g1i1j2w2n3r2s2f2d4c1a2k1t1 w 3 w 1",
    ]
    for fen in tricky_positions:
        cpp = _cpp_game(fen)
        js = _inspect_js_position(fen)
        assert _compact_cpp_state(cpp)["flags"] == {
            "inCheck": js["inCheck"],
            "isCheckmate": js["isCheckmate"],
            "isStalemate": js["isStalemate"],
            "isInsufficientMaterial": js["isInsufficientMaterial"],
            "isFourfoldRepetition": js["isFourfoldRepetition"],
            "isDraw": js["isDraw"],
            "isGameOver": js["isGameOver"],
        }
        assert cpp.fen() == js["fen"]
        assert cpp.turn() == js["turn"]
        assert cpp.move_number() == js["moveNumber"]
        assert dict(cpp.drafting()) == js["drafting"]
        assert list(cpp.moves()) == js["moves"]


@cpp_with_python_ref
def test_cpp_reference_api_and_undo_match_python_reference():
    ref = _py_ref()
    Gungi = ref["Gungi"]
    start_positions = [
        ref["INTRO_POSITION"],
        ref["BEGINNER_POSITION"],
        ref["INTERMEDIATE_POSITION"],
        ref["ADVANCED_POSITION"],
    ]

    rng = random.Random(int(os.environ.get("GUNGI_CPP_PARITY_SEED", "1337")))
    cases_per_mode = int(os.environ.get("GUNGI_CPP_PARITY_CASES_PER_MODE", "1"))
    plies = int(os.environ.get("GUNGI_CPP_PARITY_PLIES", "20"))

    for start_fen in start_positions:
        for _ in range(cases_per_mode):
            py_game = Gungi(start_fen)
            cpp_game = _cpp_game(start_fen)

            for _ in range(plies):
                py_pseudo_verbose = [_compact_py_move(move) for move in py_game.moves(verbose=True)]
                cpp_pseudo_verbose = [_compact_cpp_move(move) for move in cpp_game.moves_verbose()]
                assert cpp_pseudo_verbose == py_pseudo_verbose

                py_legal_verbose = py_pseudo_verbose
                cpp_legal_verbose = [_compact_cpp_move(move) for move in cpp_game.legal_moves_verbose()]
                assert cpp_legal_verbose == py_legal_verbose

                py_pseudo = [move["san"] for move in py_pseudo_verbose]
                py_legal = [move["san"] for move in py_legal_verbose]
                assert list(cpp_game.moves()) == py_pseudo
                assert list(cpp_game.legal_moves()) == py_legal
                if not py_pseudo:
                    break

                chosen = rng.choice(py_pseudo)
                py_game.move(chosen)
                assert cpp_game.move(chosen)
                assert _compact_cpp_state(cpp_game) == _compact_py_state(py_game)

                if py_game.is_game_over():
                    break

            while py_game.history():
                py_game.undo()
                assert cpp_game.undo()
                assert _compact_cpp_state(cpp_game) == _compact_py_state(py_game)


@cpp_with_python_ref
def test_cpp_action_surface_matches_reference_moves_in_check_positions():
    ref = _py_ref()
    Gungi = ref["Gungi"]
    would_be_in_check_after_move = ref["would_be_in_check_after_move"]
    escape_fen = "3img3/1ra1N1as1/d1fw2f1d/4dw3/9/9/D1FWDWF1D/1SA3AR1/3GMI3 J2N2S1R1D1/j2n2s1r1d1 b 1 - 3"

    py_game = Gungi(escape_fen)
    cpp_game = _cpp_game(escape_fen)

    py_moves = [move.san for move in py_game.moves(verbose=True)]
    py_self_check_moves = [
        move.san
        for move in py_game.moves(verbose=True)
        if would_be_in_check_after_move(move)
    ]

    assert len(py_self_check_moves) > 0
    assert list(cpp_game.moves()) == py_moves
    assert list(cpp_game.legal_moves()) == py_moves

    self_check_move = py_self_check_moves[0]
    assert self_check_move in cpp_game.legal_moves()
    assert cpp_game.move(self_check_move)

    cpp_training_game = _cpp_game(escape_fen)
    assert cpp_training_game.act_san(self_check_move)


@cpp_with_python_ref
def test_cpp_action_space_covers_mirrored_archer_routes():
    ref = _py_ref()
    Gungi = ref["Gungi"]

    fen = "3img3/1ra1n1as1/d1fwdwf1d/9/9/9/D1|F:R|WDWF1D/1SA1N1AR1/3GMI3 J2N2S1D1/j2n2s1r1d1 b 1 - 1"
    expected_routes = {
        "弓(2-7-1)(4-6-1)",
        "弓(2-7-1)(4-8-1)",
        "弓(2-3-1)(4-2-1)",
        "弓(2-3-1)(4-4-1)",
    }

    py_game = Gungi(fen)
    cpp_game = _cpp_game(fen)

    py_moves = {move.san for move in py_game.moves(verbose=True)}
    cpp_moves = set(cpp_game.moves())
    cpp_legal = set(cpp_game.legal_moves())

    assert expected_routes <= py_moves
    assert expected_routes <= cpp_moves
    assert expected_routes <= cpp_legal
    assert cpp_moves == py_moves
    assert cpp_legal == py_moves


@cpp_with_python_ref
def test_cpp_feature_tensor_shape_matches_declared_gungi_channels():
    game = _cpp_game(_py_ref()["BEGINNER_POSITION"])
    features = list(game.features())

    assert len(features) == 116 * 81
    assert min(features) >= 0.0
    assert max(features) <= 1.0


@cpp_with_python_ref
def test_gungi_alphazero_network_matches_env_training_surface():
    torch = pytest.importorskip("torch")
    cpp = _cpp()

    from minizero.network.py.alphazero_network import AlphaZeroNetwork

    network = AlphaZeroNetwork(
        cpp.get_game_name(),
        cpp.get_nn_num_input_channels(),
        cpp.get_nn_input_channel_height(),
        cpp.get_nn_input_channel_width(),
        32,
        cpp.get_nn_hidden_channel_height(),
        cpp.get_nn_hidden_channel_width(),
        2,
        cpp.get_nn_action_size(),
        32,
        cpp.get_nn_discrete_value_size(),
    )
    batch = torch.zeros(
        (2, network.num_input_channels, network.input_channel_height, network.input_channel_width),
        dtype=torch.float32,
    )
    output = network(batch)
    scripted = torch.jit.script(network)

    assert network.spatial_policy is True
    assert tuple(output["policy_logit"].shape) == (2, cpp.get_nn_action_size())
    assert tuple(output["policy"].shape) == (2, cpp.get_nn_action_size())
    assert tuple(output["value"].shape) == (2, cpp.get_nn_discrete_value_size())
    assert scripted.get_game_name() == cpp.get_game_name()
