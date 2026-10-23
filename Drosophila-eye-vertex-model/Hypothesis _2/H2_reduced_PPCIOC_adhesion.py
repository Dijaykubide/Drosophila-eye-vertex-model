"""
WT vertex-model first pass for the user-supplied Drosophila eye geometry.

IMPORTANT
---------
- The original geometry file is NOT edited.
- No junction is added, deleted, or rewired during minimization.
- The only moving quantities are existing vertex coordinates.
- The red dashed lines are treated as cytosolic PPC actomyosin tensile bonds;
  they are NOT cell-cell junctions.
- T1 transitions are intentionally disabled in this WT-validation pass.

Mechanics follow Blackie et al., Development (2021), Eqns 1-4 and Table S1:
    E = sum_cells K/2 (A-A0)^2 + sum_edges Lambda_ij * l_ij
    Lambda_ij = Lambda0 * S
    S = w_myo*c_myo + w_ad/c_ad
    w_myo + w_ad = 1

For the two IOC-IOC rows with laser-ablation-adjusted starred S values in
Table S1, this script interpolates linearly between the reported pure-adhesion
and pure-myosin S endpoints.

The script can compare two choices of ideal area A0:
  geometry : each starting polygon area is its A0. This is the appropriate
             first validation when the hand-built geometry is the intended WT
             shape and its dimensional scaling is not yet calibrated.
  paper    : uses Table S1 values directly (CC side=1, CC top/bottom=1.3,
             PPC=8.7, IOC=1). This is a direct paper-parameter transfer and can
             visibly relax the hand-built geometry because the units/shapes are
             not identical to the paper's mesh.
"""

from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, split
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
GEOMETRY_FILE = HERE / "tiled_sheet_v4.py"
OUTDIR = HERE / "outputs_H2_reduced_PPCIOC_adhesion"
OUTDIR.mkdir(exist_ok=True)

LAMBDA0 = 0.26
K_AREA = 1.0
W_AD_VALUES = [0.7]
MESH_VALUES = [0.25]
AREA_MODES = ["geometry"]

# A representative, article-inspired adhesion-dominant condition used only
# for the overlay image. The full grid is still written to CSV.
OVERLAY_W_AD = 0.7
OVERLAY_MESH = 0.25

# -----------------------------------------------------------------------------
# Load the ORIGINAL geometry without modifying it.
# The original file saves a preview to /mnt/user-data/outputs; redirect only
# that save call to our local output folder. Geometry code itself is unchanged.
# -----------------------------------------------------------------------------
if not GEOMETRY_FILE.exists():
    raise FileNotFoundError(
        f"Put this script in the same folder as: {GEOMETRY_FILE.name}"
    )

_original_savefig = plt.savefig

def _safe_savefig(fname, *args, **kwargs):
    s = str(fname)
    if s.startswith("/mnt/user-data/outputs/"):
        fname = OUTDIR / Path(s).name
    return _original_savefig(fname, *args, **kwargs)

plt.savefig = _safe_savefig
ns = {}
source = GEOMETRY_FILE.read_text()
exec(compile(source, str(GEOMETRY_FILE), "exec"), ns)
plt.savefig = _original_savefig
plt.close("all")

# -----------------------------------------------------------------------------
# USER-REQUESTED WT GEOMETRY CORRECTIONS
# -----------------------------------------------------------------------------
# A. Remove the two artificial gold IOC seams that subdivide a 3-IOC band into
#    5 apparent IOCs, and remove the extra left-side gold seam that breaks
#    left/right symmetry. These are REAL polygon fusions, not just hidden lines.
#
#    Seam family A: vertical seam on an ommatidium's local x=0 axis in the
#    upper/lower IOC band.
#    Seam family B: the extra vertical seam on the local left side
#    (x ~ -3.05, y ~ -0.11). In the neighboring upper/lower band this is the
#    second artificial seam seen ~1.25 units from that neighbor's axis.
#
# No other IOC-IOC contacts are removed.

_original_ioc_polys = [np.asarray(p, float) for p in ns["sheet_polys"]]
_omm_for_cleanup = [np.asarray(o, float) for o in ns["omm_centres"]]

_edge_to_cells = {}
for _ci, _p in enumerate(_original_ioc_polys):
    for _k in range(len(_p)):
        _a = tuple(np.round(_p[_k], 6))
        _b = tuple(np.round(_p[(_k + 1) % len(_p)], 6))
        _key = tuple(sorted([_a, _b]))
        _edge_to_cells.setdefault(_key, []).append(_ci)

_fuse_pairs = []
_fuse_info = []
for _key, _cids in _edge_to_cells.items():
    if len(_cids) != 2:
        continue
    _p1 = np.asarray(_key[0], float)
    _p2 = np.asarray(_key[1], float)
    if abs(_p2[0] - _p1[0]) >= 0.08:  # only the vertical seams requested
        continue
    _mid = (_p1 + _p2) / 2.0
    _kind = None
    for _o in _omm_for_cleanup:
        _d = _mid - _o
        # local central seam in top/bottom IOC band
        if abs(_d[0]) < 0.08 and 2.3 < abs(_d[1]) < 3.95:
            _kind = "central_band_seam"
            break
        # extra left-side seam (also appears as second artificial band seam
        # when viewed relative to the neighboring ommatidium)
        if abs(_d[0] + 3.0505) < 0.10 and abs(_d[1] + 0.112) < 0.25:
            _kind = "left_extra_seam"
            break
    if _kind is not None:
        _fuse_pairs.append(tuple(_cids))
        _fuse_info.append((_kind, tuple(_cids), _mid.copy()))

# The requested seams are disjoint, so each affected IOC polygon is fused once.
_skip_ioc = set()
_pair_from = {}
for _a, _b in _fuse_pairs:
    _lo, _hi = sorted((_a, _b))
    _pair_from[_lo] = _hi
    _skip_ioc.add(_hi)

_corrected_ioc_polys = []
for _i, _p in enumerate(_original_ioc_polys):
    if _i in _skip_ioc:
        continue
    if _i in _pair_from:
        _j = _pair_from[_i]
        _u = Polygon(_original_ioc_polys[_i]).union(Polygon(_original_ioc_polys[_j]))
        if _u.geom_type != "Polygon" or not _u.is_valid:
            raise RuntimeError(f"IOC fusion failed for {_i}, {_j}")
        _corrected_ioc_polys.append(np.asarray(_u.exterior.coords[:-1], float))
    else:
        _corrected_ioc_polys.append(_p.copy())

ns["sheet_polys"] = _corrected_ioc_polys

# B. Make the PPC-PPC junctions geometrically vertical at WT initialization.
# The original TOP/BOTTOM CC apices were only a few thousandths of a unit off
# the ommatidial centerline after _CC_DROP. Move ONLY those two x-coordinates
# onto x=0 locally; all other CC vertices/junctions are unchanged.
_local_center_x = -float(ns["_CC_DROP"][0])
ns["TOP"] = np.asarray(ns["TOP"], float).copy()
ns["BOTTOM"] = np.asarray(ns["BOTTOM"], float).copy()
ns["TOP"][0] = _local_center_x
ns["BOTTOM"][0] = _local_center_x
ns["cc_polys"] = [np.asarray(p, float).copy() for p in ns["cc_polys"]]
ns["cc_polys"][0][0, 0] = _local_center_x
ns["cc_polys"][1][0, 0] = _local_center_x

print("WT cleanup: fused IOC seam pairs:", len(_fuse_pairs))
print("WT cleanup: IOC polygons:", len(_original_ioc_polys), "->", len(ns["sheet_polys"]))

# -----------------------------------------------------------------------------
# Turn the already-existing white PPC spaces into two computational polygons
# per ommatidium. This does NOT add visible boundaries: the split lines are the
# two existing magenta PPC-PPC boundaries already present in the geometry.
# -----------------------------------------------------------------------------
ioc_union = unary_union([Polygon(p) for p in ns["sheet_polys"]])
holes = [Polygon(r) for r in ioc_union.interiors]
omm_centres = [np.asarray(o, float) for o in ns["omm_centres"]]

# match each pocket/hole to an ommatidium center
used = set()
hole_by_center = []
for o in omm_centres:
    idx = min(
        (i for i in range(len(holes)) if i not in used),
        key=lambda i: holes[i].centroid.distance(Point(*o)),
    )
    used.add(idx)
    hole_by_center.append(holes[idx])

cells = []
# IOC cells are exactly the welded polygons from the original geometry
for idx, p in enumerate(ns["sheet_polys"]):
    cells.append({
        "type": "IOC",
        "subtype": "IOC",
        "poly": orient(Polygon(p), sign=1.0),
        "name": f"IOC{idx}",
    })

for oi, o in enumerate(omm_centres):
    ccp = []
    subtypes = ["CC_top", "CC_bottom", "CC_left", "CC_right"]
    for k, p in enumerate(ns["cc_polys"]):
        poly = orient(Polygon(np.asarray(p, float) + ns["_CC_DROP"] + o), sign=1.0)
        ccp.append(poly)
        cells.append({
            "type": "CC",
            "subtype": subtypes[k],
            "poly": poly,
            "name": f"OM{oi}_{subtypes[k]}",
            "omm": oi,
        })

    ring = hole_by_center[oi].difference(unary_union(ccp))
    top_seg = LineString([
        ns["TOP"] + ns["_CC_DROP"] + o,
        np.array([0.0, 2.420]) + o,
    ])
    bottom_seg = LineString([
        ns["BOTTOM"] + ns["_CC_DROP"] + o,
        np.array([0.0, -2.656]) + o,
    ])
    pieces = list(split(ring, MultiLineString([top_seg, bottom_seg])).geoms)
    if len(pieces) != 2:
        raise RuntimeError(f"Could not split PPC pocket for ommatidium {oi}")
    for g in pieces:
        side = "PPC_R" if g.centroid.x > o[0] else "PPC_L"
        cells.append({
            "type": "PPC",
            "subtype": side,
            "poly": orient(g, sign=1.0),
            "name": f"OM{oi}_{side}",
            "omm": oi,
        })

# -----------------------------------------------------------------------------
# Global vertex registry: existing coincident points share one vertex.
# -----------------------------------------------------------------------------
verts = []
vmap = {}
cell_vids = []

def get_vid(pt):
    key = tuple(np.round(np.asarray(pt, float), 6))
    if key not in vmap:
        vmap[key] = len(verts)
        verts.append(np.asarray(pt, float))
    return vmap[key]

for c in cells:
    coords = np.asarray(c["poly"].exterior.coords[:-1], float)
    vids = [get_vid(p) for p in coords]
    cell_vids.append(np.asarray(vids, dtype=int))

V0 = np.asarray(verts, float)

# Edge registry and adjacency
edges = {}
for ci, vids in enumerate(cell_vids):
    for k in range(len(vids)):
        a, b = int(vids[k]), int(vids[(k + 1) % len(vids)])
        key = (a, b) if a < b else (b, a)
        edges.setdefault(key, {"cells": []})["cells"].append(ci)

# -----------------------------------------------------------------------------
# Contact classification matching the color/contact categories in Fig. 3/Table S1
# -----------------------------------------------------------------------------
def nearest_omm(mid):
    return min(omm_centres, key=lambda o: np.linalg.norm(mid - o))

def classify_edge(key, info):
    cids = info["cells"]
    if len(cids) == 1:
        return "BOUNDARY"
    c1, c2 = cells[cids[0]], cells[cids[1]]
    types = {c1["type"], c2["type"]}

    if types == {"CC"}:
        subs = {c1["subtype"], c2["subtype"]}
        return "CC-CC central" if subs == {"CC_top", "CC_bottom"} else "CC-CC side"

    if types == {"CC", "PPC"}:
        cc = c1 if c1["type"] == "CC" else c2
        return "CC-PPC AP" if cc["subtype"] in ("CC_left", "CC_right") else "CC-PPC EqPl"

    if types == {"PPC"}:
        return "PPC-PPC"

    if types == {"PPC", "IOC"}:
        return "PPC-IOC"

    if types == {"IOC"}:
        a, b = key
        mid = (V0[a] + V0[b]) / 2.0
        d = mid - nearest_omm(mid)
        # A/P contacts sit at the lateral (left/right) parts of the ommatidium;
        # Eq/Pl contacts sit at the top/bottom parts.
        return "IOC-IOC AP" if abs(d[0]) >= abs(d[1]) else "IOC-IOC EqPl"

    return "all else"

for key, info in edges.items():
    info["class"] = classify_edge(key, info)

# -----------------------------------------------------------------------------
# Table S1 values
# -----------------------------------------------------------------------------
TABLE = {
    "CC-CC side":    {"cad": 1.00, "cmyo": 1.00},
    "CC-CC central": {"cad": 1.08, "cmyo": 1.06},
    "CC-PPC EqPl":   {"cad": 1.79, "cmyo": 1.52},
    "CC-PPC AP":     {"cad": 2.11, "cmyo": 1.52},
    "PPC-PPC":       {"cad": 3.25, "cmyo": 1.33},
    "PPC-IOC":       {"cad": 4.83, "cmyo": 1.93},
    # Starred, laser-ablation-adjusted S endpoints from Table S1
    "IOC-IOC AP":    {"cad": 2.37, "cmyo": 2.24, "S_ad": 0.19, "S_myo": 1.33},
    "IOC-IOC EqPl":  {"cad": 2.24, "cmyo": 2.30, "S_ad": 0.29, "S_myo": 2.02},
    "all else":      {"cad": 1.00, "cmyo": 1.00},
}

def edge_lambda(edge_class, w_ad):
    if edge_class == "BOUNDARY":
        return 0.0
    w_myo = 1.0 - w_ad
    row = TABLE.get(edge_class, TABLE["all else"])
    if "S_ad" in row:
        # linear interpolation between the two paper-reported starred endpoints
        S = w_ad * row["S_ad"] + w_myo * row["S_myo"]
    else:
        S = w_myo * row["cmyo"] + w_ad / row["cad"]
    return LAMBDA0 * S

# -----------------------------------------------------------------------------
# Existing dashed PPC actomyosin bonds: attach them to the existing boundary
# edges by interpolation, without making them junctions.
# -----------------------------------------------------------------------------
def closest_attachment(point, allowed_classes):
    p = np.asarray(point, float)
    best = None
    for (a, b), info in edges.items():
        if info["class"] not in allowed_classes:
            continue
        va, vb = V0[a], V0[b]
        d = vb - va
        L2 = float(np.dot(d, d))
        if L2 < 1e-12:
            continue
        t = float(np.clip(np.dot(p - va, d) / L2, 0.0, 1.0))
        q = va + t * d
        dist = float(np.linalg.norm(p - q))
        if best is None or dist < best[0]:
            best = (dist, a, b, t)
    return best

mesh_springs = []
for oi, o in enumerate(omm_centres):
    for si, (a, b) in enumerate(ns["_dash"]):
        p_in = np.asarray(a, float) + ns["_CC_DROP"] + o
        p_out = np.asarray(b, float) + o
        ai = closest_attachment(p_in, {"CC-PPC AP", "CC-PPC EqPl"})
        ao = closest_attachment(p_out, {"PPC-IOC"})
        if ai is None or ao is None or ai[0] > 0.05 or ao[0] > 0.05:
            raise RuntimeError(f"Could not attach dashed bond OM{oi} #{si}")
        mesh_springs.append({
            "inner": (ai[1], ai[2], ai[3]),
            "outer": (ao[1], ao[2], ao[3]),
            "omm": oi,
        })

# Outer tissue boundary is fixed, as in the paper.
fixed = set()
for (a, b), info in edges.items():
    if len(info["cells"]) == 1:
        fixed.add(a)
        fixed.add(b)

# Exact PPC-PPC verticality constraint.
# We share ONE x-coordinate variable between the two endpoints of every
# PPC-PPC edge. Their y coordinates remain independent.
_ncoord = 2 * len(V0)
_parent = np.arange(_ncoord, dtype=int)

def _find(x):
    while _parent[x] != x:
        _parent[x] = _parent[_parent[x]]
        x = _parent[x]
    return x

def _union(a, b):
    ra, rb = _find(a), _find(b)
    if ra != rb:
        _parent[rb] = ra

for (_a, _b), _info in edges.items():
    if _info["class"] == "PPC-PPC":
        _union(2 * _a, 2 * _b)   # x only

_fixed_coords = set()
for _v in fixed:
    _fixed_coords.add(2 * _v)
    _fixed_coords.add(2 * _v + 1)

# Build one optimization variable per unconstrained coordinate group.
_root_fixed = {}
for _ci in range(_ncoord):
    _r = _find(_ci)
    if _ci in _fixed_coords:
        _root_fixed[_r] = V0.reshape(-1)[_ci]

_root_to_var = {}
_coord_to_var = np.full(_ncoord, -1, dtype=int)
_z0 = []
_flat0 = V0.reshape(-1)
for _ci in range(_ncoord):
    _r = _find(_ci)
    if _r in _root_fixed:
        continue
    if _r not in _root_to_var:
        _root_to_var[_r] = len(_z0)
        _z0.append(float(_flat0[_ci]))
    _coord_to_var[_ci] = _root_to_var[_r]
x0 = np.asarray(_z0, float)

def _unpack(x):
    flat = _flat0.copy()
    for ci, vi in enumerate(_coord_to_var):
        if vi >= 0:
            flat[ci] = x[vi]
    return flat.reshape((-1, 2))

def _pack_gradient(G):
    gf = np.asarray(G, float).reshape(-1)
    gz = np.zeros(len(x0), float)
    mask = _coord_to_var >= 0
    np.add.at(gz, _coord_to_var[mask], gf[mask])
    return gz

# free vertex ids are retained only for displacement reporting
free_ids = np.asarray([i for i in range(len(V0)) if i not in fixed], dtype=int)

# -----------------------------------------------------------------------------
# Vectorized area/edge bookkeeping
# -----------------------------------------------------------------------------
loc_cell = []
loc_vid = []
loc_prev = []
loc_next = []
area_edge_from = []
area_edge_to = []
area_edge_cell = []
for ci, vids in enumerate(cell_vids):
    n = len(vids)
    for k, v in enumerate(vids):
        loc_cell.append(ci)
        loc_vid.append(v)
        loc_prev.append(vids[(k - 1) % n])
        loc_next.append(vids[(k + 1) % n])
        area_edge_from.append(v)
        area_edge_to.append(vids[(k + 1) % n])
        area_edge_cell.append(ci)

loc_cell = np.asarray(loc_cell, int)
loc_vid = np.asarray(loc_vid, int)
loc_prev = np.asarray(loc_prev, int)
loc_next = np.asarray(loc_next, int)
area_edge_from = np.asarray(area_edge_from, int)
area_edge_to = np.asarray(area_edge_to, int)
area_edge_cell = np.asarray(area_edge_cell, int)

edge_keys = list(edges.keys())
edge_a = np.asarray([a for a, b in edge_keys], int)
edge_b = np.asarray([b for a, b in edge_keys], int)
edge_classes = np.asarray([edges[k]["class"] for k in edge_keys], object)
nonboundary = edge_classes != "BOUNDARY"

mi_a = np.asarray([s["inner"][0] for s in mesh_springs], int)
mi_b = np.asarray([s["inner"][1] for s in mesh_springs], int)
mi_t = np.asarray([s["inner"][2] for s in mesh_springs], float)
mo_a = np.asarray([s["outer"][0] for s in mesh_springs], int)
mo_b = np.asarray([s["outer"][1] for s in mesh_springs], int)
mo_t = np.asarray([s["outer"][2] for s in mesh_springs], float)

# Ideal areas
A0_geometry = np.asarray([c["poly"].area for c in cells], float)
A0_paper = []
for c in cells:
    if c["type"] == "IOC":
        A0_paper.append(1.0)
    elif c["type"] == "PPC":
        A0_paper.append(8.7)
    elif c["subtype"] in ("CC_top", "CC_bottom"):
        A0_paper.append(1.3)
    else:
        A0_paper.append(1.0)
A0_paper = np.asarray(A0_paper, float)

# -----------------------------------------------------------------------------
# Eq. 1 energy + analytic gradient. No topology changes/T1 transitions.
# -----------------------------------------------------------------------------
def make_energy(w_ad, mesh_fraction, area_mode):
    lambdas = np.asarray([edge_lambda(c, w_ad) for c in edge_classes], float)
    A0 = A0_geometry if area_mode == "geometry" else A0_paper
    Tmesh = LAMBDA0 * mesh_fraction

    def energy_grad(x):
        V = _unpack(x)
        G = np.zeros_like(V)

        # Eq. 1(i): area conservation
        vf = V[area_edge_from]
        vt = V[area_edge_to]
        cross = vf[:, 0] * vt[:, 1] - vt[:, 0] * vf[:, 1]
        areas = 0.5 * np.bincount(area_edge_cell, weights=cross, minlength=len(cells))
        dA = areas - A0
        E = 0.5 * K_AREA * float(np.dot(dA, dA))

        coeff = K_AREA * dA[loc_cell]
        gx = 0.5 * (V[loc_next, 1] - V[loc_prev, 1]) * coeff
        gy = 0.5 * (V[loc_prev, 0] - V[loc_next, 0]) * coeff
        np.add.at(G[:, 0], loc_vid, gx)
        np.add.at(G[:, 1], loc_vid, gy)

        # Eq. 1(ii) with Eqns 2-4: line energy
        d = V[edge_a] - V[edge_b]
        lengths = np.linalg.norm(d, axis=1)
        safe = np.maximum(lengths, 1e-12)
        E += float(np.dot(lambdas, lengths))
        ge = (lambdas / safe)[:, None] * d
        np.add.at(G, edge_a, ge)
        np.add.at(G, edge_b, -ge)

        # PPC cytosolic actomyosin bonds, strength = fraction * base line tension
        if Tmesh > 0:
            pin = (1 - mi_t)[:, None] * V[mi_a] + mi_t[:, None] * V[mi_b]
            pout = (1 - mo_t)[:, None] * V[mo_a] + mo_t[:, None] * V[mo_b]
            dm = pin - pout
            lm = np.linalg.norm(dm, axis=1)
            sm = np.maximum(lm, 1e-12)
            E += Tmesh * float(np.sum(lm))
            gm = Tmesh * dm / sm[:, None]
            np.add.at(G, mi_a, (1 - mi_t)[:, None] * gm)
            np.add.at(G, mi_b, mi_t[:, None] * gm)
            np.add.at(G, mo_a, -(1 - mo_t)[:, None] * gm)
            np.add.at(G, mo_b, -mo_t[:, None] * gm)

        return E, _pack_gradient(G)

    return energy_grad


def calculate_areas(V):
    vf = V[area_edge_from]
    vt = V[area_edge_to]
    cross = vf[:, 0] * vt[:, 1] - vt[:, 0] * vf[:, 1]
    return 0.5 * np.bincount(area_edge_cell, weights=cross, minlength=len(cells))


def run_one(w_ad, mesh, area_mode):
    fg = make_energy(w_ad, mesh, area_mode)
    E0, g0 = fg(x0)
    result = minimize(
        fg,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
    )
    V = _unpack(result.x)

    disp = np.linalg.norm(V - V0, axis=1)[free_ids]
    edge_lengths = np.linalg.norm(V[edge_a] - V[edge_b], axis=1)
    active_lengths = edge_lengths[nonboundary]
    Afinal = calculate_areas(V)
    Astart = calculate_areas(V0)

    summary = {
        "area_mode": area_mode,
        "w_ad": w_ad,
        "w_myo": 1.0 - w_ad,
        "mesh_fraction": mesh,
        "optimizer_success": bool(result.success),
        "iterations": int(result.nit),
        "energy_start": E0,
        "energy_final": float(result.fun),
        "initial_force_norm": float(np.linalg.norm(g0)),
        "rms_vertex_displacement": float(np.sqrt(np.mean(disp ** 2))),
        "mean_vertex_displacement": float(np.mean(disp)),
        "max_vertex_displacement": float(np.max(disp)),
        "minimum_internal_edge": float(np.min(active_lengths)),
        "edges_below_0.02": int(np.sum(active_lengths < 0.02)),
        "mean_abs_area_change_percent": float(np.mean(np.abs((Afinal - Astart) / Astart)) * 100),
        "max_ppc_ppc_dx": float(max(
            [abs(V[a,0] - V[b,0]) for (a,b), info in edges.items() if info["class"] == "PPC-PPC"]
            or [0.0]
        )),
    }
    return result, V, summary



# -----------------------------------------------------------------------------
# Plot/output helpers used by the hypothesis sweep
# -----------------------------------------------------------------------------
COLORS = {
    "CC-CC central": "#B30000",
    "CC-CC side": "#508C46",
    "CC-PPC AP": "#0056B6",
    "CC-PPC EqPl": "#54B9D1",
    "PPC-PPC": "#E040E0",
    "PPC-IOC": "#8E2C8E",
    "IOC-IOC AP": "#B8860B",
    "IOC-IOC EqPl": "#D5A72A",
    "BOUNDARY": "#222222",
}

def draw_condition(V, label, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    # WT starting geometry in gray
    for (a, b), info in edges.items():
        p, q = V0[a], V0[b]
        if max(p[0], q[0]) < -6 or min(p[0], q[0]) > 6 or max(p[1], q[1]) < -6 or min(p[1], q[1]) > 6:
            continue
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#D0D0D0", lw=0.8, zorder=0)

    # Relaxed geometry in junction colors
    for (a, b), info in edges.items():
        p, q = V[a], V[b]
        if max(p[0], q[0]) < -6 or min(p[0], q[0]) > 6 or max(p[1], q[1]) < -6 or min(p[1], q[1]) > 6:
            continue
        ax.plot([p[0], q[0]], [p[1], q[1]], color=COLORS[info["class"]], lw=1.5, zorder=2)

    # Existing PPC cytosolic actomyosin mesh (dashed red)
    for s in mesh_springs:
        o = omm_centres[s["omm"]]
        if not (-7 < o[0] < 7 and -7 < o[1] < 7):
            continue
        ai, bi, ti = s["inner"]
        ao, bo, to = s["outer"]
        pin = (1 - ti) * V[ai] + ti * V[bi]
        pout = (1 - to) * V[ao] + to * V[bo]
        ax.plot([pin[0], pout[0]], [pin[1], pout[1]],
                color="#E8452C", lw=1.2, ls="--", dashes=(4, 3), zorder=3)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.axis("off")
    ax.set_title(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / filename, dpi=220, facecolor="white")
    plt.close(fig)


def add_edge_collapse_diagnostics(V, row):
    lengths = np.linalg.norm(V[edge_a] - V[edge_b], axis=1)
    ppcppc_mask = edge_classes == "PPC-PPC"
    ppcioc_mask = edge_classes == "PPC-IOC"

    row["min_ppc_ppc_edge"] = float(np.min(lengths[ppcppc_mask]))
    row["ppc_ppc_edges_below_0.02"] = int(np.sum(lengths[ppcppc_mask] < 0.02))
    row["min_ppc_ioc_edge"] = float(np.min(lengths[ppcioc_mask]))
    row["ppc_ioc_edges_below_0.02"] = int(np.sum(lengths[ppcioc_mask] < 0.02))
    return row


# -----------------------------------------------------------------------------
# H2: Reduced PPC-IOC adhesion
# -----------------------------------------------------------------------------
# Exact 100% adhesion loss is not allowed because c_ad = 0 makes 1/c_ad undefined.
# 99% is therefore used as the near-100% condition.
REDUCTIONS = [20, 40, 60, 80, 99]
WT_CAD = 4.83
WT_CMYO = 1.93
W_AD = 0.7
PPC_MESH = 0.25

rows = []
for reduction in REDUCTIONS:
    fraction_left = 1.0 - reduction / 100.0
    TABLE["PPC-IOC"]["cad"] = WT_CAD * fraction_left
    TABLE["PPC-IOC"]["cmyo"] = WT_CMYO

    result, V, row = run_one(W_AD, PPC_MESH, "geometry")
    row = add_edge_collapse_diagnostics(V, row)
    row["hypothesis"] = "Reduced PPC-IOC adhesion"
    row["reduction_percent"] = reduction
    row["PPC-IOC_cad"] = TABLE["PPC-IOC"]["cad"]
    rows.append(row)

    label = ("~100%" if reduction == 99 else f"{reduction}%") + " PPC-IOC adhesion reduction"
    draw_condition(V, label, f"H2_PPCIOC_adhesion_reduction_{reduction:02d}.png")

pd.DataFrame(rows).to_csv(OUTDIR / "H2_results.csv", index=False)
print("Done: H2 Reduced PPC-IOC adhesion")
print(pd.DataFrame(rows)[[
    "reduction_percent","PPC-IOC_cad","rms_vertex_displacement",
    "ppc_ppc_edges_below_0.02","ppc_ioc_edges_below_0.02"
]].to_string(index=False))
print("Outputs:", OUTDIR)
