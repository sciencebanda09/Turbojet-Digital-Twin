"""STEP -> VTP mesh conversion for engine CAD models.

Groups CAD parts into 4 meshes (compressor/combustor/turbine/casing)
per model.  Supports per-file (one .stp per part) and single-file
(named assembly) CAD sources, selected via --model.

Usage:
    python scripts/convert_engine_cad.py --model generic_turbine
    python scripts/convert_engine_cad.py --model kj66
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cadquery as cq
import numpy as np
import pyvista as pv
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
CONFIG_DIR = _PROJECT_ROOT / "configs" / "cad_models"
BASE_OUT_DIR = _PROJECT_ROOT / "models" / "engine_meshes"


# ── helpers ──────────────────────────────────────────────────────────

def _load_config(model_name: str) -> dict:
    path = CONFIG_DIR / f"{model_name}.yaml"
    if not path.exists():
        print(f"Config not found: {path}")
        print(f"Available models: {[p.stem for p in CONFIG_DIR.glob('*.yaml')]}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _match_name(name: str, patterns: list[str]) -> bool:
    """Return True if *name* matches a pattern.

    An exact match wins over a prefix match.  Prefix matches accept
    dot (``Name.1``) or space (``Name 2``) as a separator after the
    pattern text — this handles XCAF instance naming like ``Bolt 1``.
    """
    name_lower = name.lower()
    sorted_pats = sorted(patterns, key=lambda p: -len(p))
    for pat in sorted_pats:
        pat_lower = pat.lower()
        if name_lower == pat_lower:
            return True
        if name_lower.startswith(pat_lower) and len(name_lower) > len(pat_lower):
            next_ch = name_lower[len(pat_lower)]
            if next_ch in (".", " "):
                return True
    return False


def _tessellate_shape(shape, tolerance: float, label: str) -> pv.PolyData | None:
    """Tessellate a cadquery Shape into a pyvista PolyData mesh."""
    try:
        t0 = time.time()
        verts: list[tuple[float, float, float]] = []
        triangles: list[list[int]] = []
        for face in shape.Faces():
            mesh = face.tessellate(tolerance)
            base = len(verts)
            verts.extend([(float(v.x), float(v.y), float(v.z)) for v in mesh[0]])
            for tri in mesh[1]:
                triangles.append([base + int(tri[0]), base + int(tri[1]), base + int(tri[2])])
        if not triangles:
            print(f"    WARNING: {label} has no faces")
            return None
        faces = np.hstack([[3] + tri for tri in triangles])
        mesh = pv.PolyData(np.array(verts, dtype=np.float64), faces)
        t1 = time.time()
        print(f"    {label}: {mesh.n_points:>6} pts, {mesh.n_faces:>6} faces ({t1 - t0:.1f}s)")
        return mesh
    except Exception as e:
        print(f"    ERROR tessellating {label}: {e}")
        return None


def _merge_meshes(meshes: list[pv.PolyData]) -> pv.PolyData | None:
    """Concatenate meshes by merging point/face arrays — much faster than iterative merge()."""
    valid = [m for m in meshes if m is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    total_pts = sum(m.n_points for m in valid)

    # Pre-allocate arrays
    all_pts = np.empty((total_pts, 3), dtype=np.float64)
    all_faces = []

    offset = 0
    for m in valid:
        all_pts[offset:offset + m.n_points] = m.points
        f = m.faces.reshape(-1, 4).copy()
        f[:, 1:] += offset
        all_faces.append(f)
        offset += m.n_points

    faces_np = np.concatenate(all_faces, axis=0).ravel()
    mesh = pv.PolyData(all_pts, faces_np)
    return mesh


# ── per-file loader (generic_turbine style) ──────────────────────────

def _load_per_file(
    zip_path: Path,
    config: dict,
    tolerance: float,
) -> dict[str, list[pv.PolyData]]:
    """Load parts from individual .stp files in the zip archive."""
    stage_map: dict[str, list[str]] = config["stage_map"]
    casing_parts: list[str] = config.get("casing_parts", [])
    skip_parts: list[str] = config.get("skip_parts", [])

    skip_lower = {s.lower().replace(" ", "_") for s in skip_parts}

    extract_dir = BASE_OUT_DIR / "_step_src"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    files_extracted = list(extract_dir.glob("*.stp"))
    print(f"Extracted {len(files_extracted)} files")

    step_files: dict[str, Path] = {}
    for f in files_extracted:
        stem = f.stem.lower().replace(" ", "_")
        if stem in skip_lower:
            print(f"  SKIP {f.name} (skip_parts)")
            continue
        step_files[stem] = f
    print(f"Found {len(step_files)} STEP files to process")

    results: dict[str, list[pv.PolyData]] = defaultdict(list)

    def load_one(stem: str, tag: str, target: str) -> None:
        if stem not in step_files:
            print(f"    SKIP {stem} ({tag}): not found")
            return
        mesh = _tessellate_shape(
            cq.importers.importStep(str(step_files[stem])).val(),
            tolerance,
            f"{stem} ({tag})",
        )
        if mesh is not None:
            results[target].append(mesh)

    for stage_name, part_stems in stage_map.items():
        print(f"\n--- Stage: {stage_name} ---")
        for stem in part_stems:
            load_one(stem, stage_name, stage_name)

    print("\n--- Stage: casing ---")
    for stem in casing_parts:
        load_one(stem, "casing", "casing")

    assigned = set()
    for stems in stage_map.values():
        assigned.update(stems)
    assigned.update(casing_parts)
    for stem, path in step_files.items():
        if stem not in assigned:
            print(f"  (unassigned -> casing) {stem}")
            mesh = _tessellate_shape(
                cq.importers.importStep(str(path)).val(),
                tolerance,
                f"{stem} (unassigned)",
            )
            if mesh is not None:
                results["casing"].append(mesh)

    return dict(results)


# ── assembly loader (kj66 style) ─────────────────────────────────────

def _load_assembly(
    zip_path: Path,
    step_file: str,
    config: dict,
    tolerance: float,
) -> dict[str, list[pv.PolyData]]:
    """Load named parts from a single STEP assembly via XCAF."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool

    stage_map: dict[str, list[str]] = config["stage_map"]
    casing_parts: list[str] = config.get("casing_parts", [])
    skip_parts: list[str] = config.get("skip_parts", [])

    print(f"Extracting {step_file} from {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = zf.read(step_file)
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".stp")
    os.write(fd, data)
    os.close(fd)
    tmp = Path(tmp_path)

    try:
        print("Loading assembly via XCAF ...")
        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
        app.InitDocument(doc)

        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        status = reader.ReadFile(str(tmp))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed with status {status}")
        ok = reader.Transfer(doc)
        if not ok:
            raise RuntimeError("STEP transfer to XCAF document failed")

        shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

        free_seq = TDF_LabelSequence()
        shape_tool.GetFreeShapes(free_seq)
        print(f"Assembly has {free_seq.Length()} free shape(s)")

        # Collect all components with their names and OCC shapes
        components: list[tuple[str, object]] = []
        for i in range(1, free_seq.Length() + 1):
            lbl = free_seq.Value(i)
            comps = TDF_LabelSequence()
            shape_tool.GetComponents_s(lbl, comps)
            for j in range(1, comps.Length() + 1):
                clbl = comps.Value(j)
                na = TDataStd_Name()
                name = "(unnamed)"
                if clbl.FindAttribute(TDataStd_Name.GetID_s(), na):
                    name = str(na.Get().ToExtString())
                occ_shape = XCAFDoc_ShapeTool.GetShape_s(clbl)
                if occ_shape is not None and not occ_shape.IsNull():
                    components.append((name, occ_shape))

        print(f"Found {len(components)} total components")

        # Match components to stages
        stage_results: dict[str, list[pv.PolyData]] = defaultdict(list)
        assigned_names: set[str] = set()

        total_solids = 0

        # Build a single ordered list of (pattern, target_stage) sorted by pattern length desc
        all_rules = []
        for stage_name, patterns in stage_map.items():
            for p in patterns:
                all_rules.append((p, stage_name))
        for p in casing_parts:
            all_rules.append((p, "casing"))
        all_rules.sort(key=lambda r: -len(r[0]))

        for name, occ_shape in components:
            exp = TopExp_Explorer(occ_shape, TopAbs_SOLID)
            n_solids = 0
            while exp.More():
                n_solids += 1
                exp.Next()
            total_solids += n_solids

            if _match_name(name, skip_parts):
                print(f"  SKIP {name} ({n_solids} solids)")
                assigned_names.add(name)
                continue

            matched_stage = None
            for pat, stage in all_rules:
                if _match_name(name, [pat]):
                    matched_stage = stage
                    break

            if matched_stage is None:
                matched_stage = "casing"
                print(f"  UNMATCHED {name} ({n_solids} solids) -> casing (fallback)")
            else:
                print(f"\n  {name} ({n_solids} solids) -> {matched_stage}")

            mesh = _tessellate_occ_shape(occ_shape, tolerance, f"{name} -> {matched_stage}")
            if mesh is not None:
                stage_results[matched_stage].append(mesh)
            assigned_names.add(name)

        # Report unassigned patterns (for debugging)
        for stage_name, patterns in stage_map.items():
            for pat in patterns:
                found = any(_match_name(n, [pat]) for n, _ in components)
                if not found:
                    print(f"    NOTE: pattern '{pat}' ({stage_name}) matched no components")

        print(f"\nTotal solids in assembly: {total_solids}")

    finally:
        if tmp.exists():
            tmp.unlink()

    return dict(stage_results)


def _tessellate_occ_shape(occ_shape, tolerance: float, label: str) -> pv.PolyData | None:
    """Tessellate an OCC TopoDS_Shape into a pyvista mesh via cadquery."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    try:
        t0 = time.time()
        verts: list[tuple[float, float, float]] = []
        triangles: list[list[int]] = []

        # Wrap OCC shape as a cadquery Shape
        cq_shape = cq.Shape.cast(occ_shape)

        # If the shape is a compound, iterate its solids
        exp = TopExp_Explorer(occ_shape, TopAbs_SOLID)
        solid_shapes = []
        while exp.More():
            solid_shapes.append(cq.Shape.cast(exp.Current()))
            exp.Next()

        if not solid_shapes:
            solid_shapes = [cq_shape]

        for solid in solid_shapes:
            for face in solid.Faces():
                mesh = face.tessellate(tolerance)
                base = len(verts)
                verts.extend([(float(v.x), float(v.y), float(v.z)) for v in mesh[0]])
                for tri in mesh[1]:
                    triangles.append([base + int(tri[0]), base + int(tri[1]), base + int(tri[2])])

        if not triangles:
            print(f"    WARNING: {label} has no faces")
            return None
        faces = np.hstack([[3] + tri for tri in triangles])
        mesh = pv.PolyData(np.array(verts, dtype=np.float64), faces)
        t1 = time.time()
        print(f"    {label}: {mesh.n_points:>6} pts, {mesh.n_faces:>6} faces ({t1 - t0:.1f}s)")
        return mesh
    except Exception as e:
        print(f"    ERROR tessellating {label}: {e}")
        return None


# ── main pipeline ────────────────────────────────────────────────────

def process_model(
    model_name: str,
    tolerance: float = 0.5,
    decimate: int | None = 200_000,
    zip_path_override: str | None = None,
) -> None:
    """Run full conversion for *model_name*."""
    config = _load_config(model_name)
    zip_path = Path(zip_path_override or config["zip_path"])

    # Detect loading mode
    is_assembly = "step_file" in config

    out_dir = BASE_OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Zip:   {zip_path}")
    print(f"Mode:  {'assembly (single STEP file)' if is_assembly else 'per-file (individual .stp parts)'}")
    print(f"Out:   {out_dir}")
    print(f"{'=' * 60}\n")

    if is_assembly:
        stage_results = _load_assembly(zip_path, config["step_file"], config, tolerance)
    else:
        stage_results = _load_per_file(zip_path, config, tolerance)

    if not stage_results:
        print("No meshes loaded — nothing to save.")
        return

    # Merge and save
    print("\n--- Merging and saving ---")
    stages_written = []

    for stage_name in ["compressor", "combustor", "turbine", "casing"]:
        meshes = stage_results.get(stage_name)
        if not meshes:
            print(f"  {stage_name}: no meshes, skipping")
            continue
        merged = _merge_meshes(meshes)
        if merged is None:
            print(f"  {stage_name}: no valid meshes after merge, skipping")
            continue

        if decimate and merged.n_faces > decimate:
            before = merged.n_faces
            try:
                import vtk
                dec = vtk.vtkDecimatePro()
                dec.SetInputData(merged)
                dec.SetTargetReduction(1.0 - decimate / before)
                dec.PreserveTopologyOff()
                dec.Update()
                output = dec.GetOutput()
                merged = pv.wrap(output)
                print(f"  {stage_name}: decimated {before} -> {merged.n_faces} faces")
            except Exception as e:
                print(f"  {stage_name}: decimation failed ({e}), keeping original")

        out_path = out_dir / f"{stage_name}.vtp"
        merged.save(str(out_path))
        print(f"  {stage_name}: {merged.n_points} pts, {merged.n_faces} faces -> {out_path}")
        stages_written.append(stage_name)

    # Assembly metadata
    meta = {
        "model": model_name,
        "stages": stages_written,
        "files": {s: f"{s}.vtp" for s in stages_written},
    }
    meta_path = out_dir / "assembly.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Metadata written to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert STEP CAD to pyVista VTP meshes")
    parser.add_argument("--model", default="generic_turbine", help="Model name (configs/cad_models/<name>.yaml)")
    parser.add_argument("--zip-path", help="Override ZIP path from config")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tessellation tolerance (mm)")
    parser.add_argument("--decimate", type=int, default=200_000, help="Max faces per stage (None = no decimation)")
    args = parser.parse_args()
    process_model(
        model_name=args.model,
        tolerance=args.tolerance,
        decimate=args.decimate,
        zip_path_override=args.zip_path,
    )


if __name__ == "__main__":
    main()
