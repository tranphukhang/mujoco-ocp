from __future__ import annotations

from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    """Check whether a path looks like the repository root."""
    return (path / "pyproject.toml").is_file() and (path / "assets").is_dir()


def _find_repo_root(start: Path) -> Path:
    """
    Find the repository root by walking upward from `start`.

    This avoids depending on the current working directory.
    """
    start = start.resolve()

    for candidate in (start, *start.parents):
        if _is_repo_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not find repository root. Expected to find both "
        "'pyproject.toml' and 'assets/' in one of the parent directories."
    )


# ---------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = _find_repo_root(Path(__file__))

ASSETS_DIR = REPO_ROOT / "assets"


# ---------------------------------------------------------------------
# Asset folders
# ---------------------------------------------------------------------

UNITREE_G1_DIR = ASSETS_DIR / "unitree_g1"
UNITREE_G1_MESH_DIR = UNITREE_G1_DIR / "assets"

ROBOTIQ_2F85_DIR = ASSETS_DIR / "robotiq_2f85_v4"
ROBOTIQ_2F85_MESH_DIR = ROBOTIQ_2F85_DIR / "assets"

G1_GRIPPER_DIR = ASSETS_DIR / "g1_gripper"


# ---------------------------------------------------------------------
# Main XML files
# ---------------------------------------------------------------------

G1_GRIPPER_XML = G1_GRIPPER_DIR / "G1_with_gripper.xml"
G1_GRIPPER_SCENE_XML = G1_GRIPPER_DIR / "scene.xml"

ROBOTIQ_2F85_XML = ROBOTIQ_2F85_DIR / "2f85.xml"

G1_FIXED_BASE_XML = UNITREE_G1_DIR / "g1_fixed_base.xml"
G1_FIXED_BASE_SCENE_XML = UNITREE_G1_DIR / "scene_fixed_base.xml"


def require_file(path: Path) -> Path:
    """Return path if it exists, otherwise raise FileNotFoundError."""
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def require_dir(path: Path) -> Path:
    """Return path if it exists, otherwise raise NotADirectoryError."""
    path = path.resolve()

    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    return path


def asset_path(*parts: str, must_exist: bool = False) -> Path:
    """
    Build a path inside the root assets directory.

    Example
    -------
    asset_path("g1_gripper", "G1_with_gripper.xml")
    """
    path = ASSETS_DIR.joinpath(*parts)

    if must_exist:
        return require_file(path)

    return path.resolve()