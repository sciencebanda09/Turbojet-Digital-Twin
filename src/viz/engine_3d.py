from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import yaml
from src.utils.paths import ROOT

BASE_MESH_DIR = ROOT / "models" / "engine_meshes"
TEMPLATE_DIR = Path(__file__).parent
SENSOR_CONFIG = ROOT / "configs" / "sensor_positions.yaml"
VIZ_CONFIG = ROOT / "configs" / "viz_config.yaml"
STAGE_ORDER = ["casing", "compressor", "combustor", "turbine"]


def _active_engine_model() -> str:
    """Return the engine model name from viz_config, falling back to generic_turbine."""
    try:
        cfg = _load_viz_config()
        return cfg.get("active_engine_model", "generic_turbine")
    except Exception:
        return "generic_turbine"


_DEFAULT_HEALTH_THRESHOLDS = [
    {"threshold": 0.85, "color": [0.2, 0.85, 0.2], "label": "Healthy"},
    {"threshold": 0.60, "color": [0.85, 0.80, 0.15], "label": "Warning"},
    {"threshold": 0.35, "color": [0.95, 0.50, 0.10], "label": "Critical"},
    {"threshold": 0.00, "color": [0.85, 0.15, 0.15], "label": "Failed"},
]


def load_engine_meshes(
    model_name: str | None = None, mesh_dir: str | Path | None = None, lite: bool = True
) -> dict[str, pv.PolyData]:
    if mesh_dir is not None:
        mesh_dir = Path(mesh_dir)
    else:
        model = model_name if model_name is not None else _active_engine_model()
        mesh_dir = BASE_MESH_DIR / model
    meshes = {}
    for stage in STAGE_ORDER:
        name = f"{stage}_lite.vtp" if lite else f"{stage}.vtp"
        path = mesh_dir / name
        if not path.exists():
            path = mesh_dir / f"{stage}.vtp"
        if not path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {path}. Run scripts/convert_engine_cad.py first."
            )
        mesh = pv.read(str(path))
        meshes[stage] = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
    return meshes


def _mesh_bounds(mesh: pv.PolyData) -> tuple[float, float, float, float, float, float]:
    return mesh.bounds


def _compute_explode_offsets(meshes: dict[str, pv.PolyData]) -> dict[str, list[float]]:
    """Compute explode direction vectors from mesh bounds.

    Each stage's offset is directed outward from the engine center (x=0)
    along its centroid x-coordinate. Y/Z offsets remain zero.
    """
    offsets = {}
    for stage in STAGE_ORDER:
        mesh = meshes.get(stage)
        if mesh is None:
            offsets[stage] = [0.0, 0.0, 0.0]
            continue
        bounds = _mesh_bounds(mesh)
        cx = (bounds[0] + bounds[1]) / 2
        # Scale factor: roughly 0.4 * half-span for a natural exploded look
        half_span = (bounds[1] - bounds[0]) / 2 if stage != "casing" else 0.0
        x_dir = 1.0 if cx >= 0 else -1.0
        offsets[stage] = [x_dir * half_span * 0.8, 0.0, 0.0]
    return offsets


def _mesh_to_json(mesh: pv.PolyData) -> dict:
    verts = mesh.points.tolist()
    faces = []
    arr = mesh.faces.reshape(-1, 4)
    for row in arr:
        faces.append([int(row[1]), int(row[2]), int(row[3])])
    return {"vertices": verts, "faces": faces}


def _load_sensor_config() -> dict:
    if SENSOR_CONFIG.exists():
        with open(SENSOR_CONFIG, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("sensors", {})
    return {}


def _load_viz_config() -> dict:
    if VIZ_CONFIG.exists():
        with open(VIZ_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _health_key(name: str) -> str:
    return f"{name.capitalize()}Health"


def build_interactive_html(
    health: dict[str, float],
    meshes: dict[str, pv.PolyData] | None = None,
    sensors: dict | None = None,
    replay_data: list[dict] | None = None,
    faults: dict | None = None,
    rpm: float = 12000,
    health_thresholds: list[dict] | None = None,
    model_name: str | None = None,
) -> str:
    if meshes is None:
        meshes = load_engine_meshes(model_name=model_name)
    if sensors is None:
        sensors = _load_sensor_config()
    if health_thresholds is None:
        viz_config = _load_viz_config()
        health_thresholds = viz_config.get("health_thresholds", _DEFAULT_HEALTH_THRESHOLDS)

    template_path = TEMPLATE_DIR / "viewer_template.html"
    if not template_path.exists():
        msg = f"Viewer template not found at {template_path}"
        raise FileNotFoundError(msg)

    stages_json = {}
    for stage in STAGE_ORDER:
        mesh = meshes.get(stage)
        if mesh is not None:
            stages_json[stage] = _mesh_to_json(mesh)

    explode_offsets = _compute_explode_offsets(meshes)

    stage_health = {}
    for stage in STAGE_ORDER:
        if stage == "casing":
            continue
        hk = _health_key(stage)
        stage_health[stage] = health.get(hk, 0.5)

    data_payload = {
        "stages": stages_json,
        "health": stage_health,
        "healthThresholds": health_thresholds,
        "explodeOffsets": explode_offsets,
        "sensors": sensors,
        "rpm": rpm,
    }
    if faults:
        data_payload["faults"] = faults
    if replay_data:
        data_payload["replay"] = replay_data

    data_json = json.dumps(data_payload)

    template_html = template_path.read_text(encoding="utf-8")
    if "{{GEOJSON}}" not in template_html:
        msg = "Template missing {{GEOJSON}} placeholder"
        raise ValueError(msg)
    return template_html.replace("{{GEOJSON}}", data_json)


def render_static_image(
    health: dict[str, float],
    meshes: dict[str, pv.PolyData] | None = None,
    height: int = 500,
    model_name: str | None = None,
) -> Any:
    if meshes is None:
        meshes = load_engine_meshes(model_name=model_name)

    pl = pv.Plotter(off_screen=True, window_size=[int(height * 1.6), height])
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(5000, 3000, 5000), intensity=0.7))
    pl.add_light(pv.Light(position=(-3000, -2000, 2000), intensity=0.3))
    pl.add_light(pv.Light(position=(-2000, 0, -2000), intensity=0.2))

    casing = meshes.get("casing")
    if casing is not None:
        pl.add_mesh(casing, color="lightgray", opacity=0.12, smooth_shading=True, name="casing")

    from pyvista import _vtk  # noqa: F401

    for stage in ["compressor", "combustor", "turbine"]:
        mesh = meshes.get(stage)
        if mesh is None:
            continue
        hk = _health_key(stage)
        hv = health.get(hk, 0.5)
        clipped = np.clip(hv, 0.0, 1.0)
        n_cells = mesh.n_cells
        scalars = np.full(n_cells, clipped)
        m = mesh.copy()
        m.cell_data["health"] = scalars
        pl.add_mesh(
            m,
            scalars="health",
            cmap="RdYlGn",
            clim=[0, 1],
            show_scalar_bar=False,
            smooth_shading=True,
            specular=0.3,
            specular_power=20,
            name=stage,
        )

    pl.camera_position = [(6000, -4000, 2000), (0, 0, 0), (0, 0, 1)]
    pl.camera.zoom(0.7)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img
