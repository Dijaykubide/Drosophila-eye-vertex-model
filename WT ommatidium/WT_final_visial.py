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
"""

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, LineString, MultiLineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, split
from scipy.optimize import minimize


# ============================================================
# USER SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent

GEOMETRY_FILE = HERE / "tiled_sheet_v4 (1)(2).py"

OUTDIR = HERE / "WT_visual_final_outputs"
OUTDIR.mkdir(exist_ok=True)

LAMBDA0 = 0.26
K_AREA = 1.0

# WT baseline
W_AD_VALUES = [0.7]
MESH_VALUES = [0.25]
AREA_MODES = ["geometry"]

OVERLAY_W_AD = 0.7
OVERLAY_MESH = 0.25

# Makes ONLY the side IOC band thinner.
# Top/bottom IOC rows stay unchanged.
SIDE_PPC_IOC_RADIAL_SCALE = 1.10


# ============================================================
# LOAD ORIGINAL GEOMETRY
# ============================================================

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

ns = {"__file__": str(GEOMETRY_FILE)}

source = GEOMETRY_FILE.read_text()

exec(
    compile(
        source,
        str(GEOMETRY_FILE),
        "exec"
    ),
    ns
)

plt.savefig = _original_savefig

plt.close("all")


# ============================================================
# WT IOC CLEANUP
# ============================================================

_original_ioc_polys = [
    np.asarray(p, float)
    for p in ns["sheet_polys"]
]

_omm_for_cleanup = [
    np.asarray(o, float)
    for o in ns["omm_centres"]
]


# Build adjacency map for IOC polygons

_edge_to_cells = {}

for _ci, _p in enumerate(_original_ioc_polys):

    for _k in range(len(_p)):

        _a = tuple(
            np.round(
                _p[_k],
                6
            )
        )

        _b = tuple(
            np.round(
                _p[(_k + 1) % len(_p)],
                6
            )
        )

        _key = tuple(
            sorted(
                [_a, _b]
            )
        )

        _edge_to_cells.setdefault(
            _key,
            []
        ).append(_ci)


# ------------------------------------------------------------
# Fuse only the original artificial IOC seams
# ------------------------------------------------------------

_fuse_pairs = []
_fuse_info = []

for _key, _cids in _edge_to_cells.items():

    if len(_cids) != 2:
        continue

    _p1 = np.asarray(
        _key[0],
        float
    )

    _p2 = np.asarray(
        _key[1],
        float
    )

    # Only vertical seams in the original WT cleanup
    if abs(
        _p2[0] - _p1[0]
    ) >= 0.08:
        continue

    _mid = (
        _p1 + _p2
    ) / 2.0

    _kind = None

    for _o in _omm_for_cleanup:

        _d = _mid - _o

        # Central seam in top/bottom IOC bands
        if (
            abs(_d[0]) < 0.08
            and
            2.3 < abs(_d[1]) < 3.95
        ):
            _kind = "central_band_seam"
            break

        # Extra left-side seam
        if (
            abs(
                _d[0] + 3.0505
            ) < 0.10
            and
            abs(
                _d[1] + 0.112
            ) < 0.25
        ):
            _kind = "left_extra_seam"
            break

    if _kind is not None:

        _fuse_pairs.append(
            tuple(_cids)
        )

        _fuse_info.append(
            (
                _kind,
                tuple(_cids),
                _mid.copy()
            )
        )


# ------------------------------------------------------------
# Perform IOC polygon fusion
# ------------------------------------------------------------

_skip_ioc = set()

_pair_from = {}

for _a, _b in _fuse_pairs:

    _lo, _hi = sorted(
        (_a, _b)
    )

    _pair_from[_lo] = _hi

    _skip_ioc.add(_hi)


_corrected_ioc_polys = []

for _i, _p in enumerate(
    _original_ioc_polys
):

    if _i in _skip_ioc:
        continue

    if _i in _pair_from:

        _j = _pair_from[_i]

        _u = Polygon(
            _original_ioc_polys[_i]
        ).union(
            Polygon(
                _original_ioc_polys[_j]
            )
        )

        if (
            _u.geom_type != "Polygon"
            or
            not _u.is_valid
        ):
            raise RuntimeError(
                f"IOC fusion failed for {_i}, {_j}"
            )

        _corrected_ioc_polys.append(
            np.asarray(
                _u.exterior.coords[:-1],
                float
            )
        )

    else:

        _corrected_ioc_polys.append(
            _p.copy()
        )


ns["sheet_polys"] = _corrected_ioc_polys


# ============================================================
# MAKE PPC-PPC CONTACTS VERTICAL
# ============================================================

_local_center_x = -float(
    ns["_CC_DROP"][0]
)

ns["TOP"] = np.asarray(
    ns["TOP"],
    float
).copy()

ns["BOTTOM"] = np.asarray(
    ns["BOTTOM"],
    float
).copy()

ns["TOP"][0] = _local_center_x

ns["BOTTOM"][0] = _local_center_x


ns["cc_polys"] = [
    np.asarray(
        p,
        float
    ).copy()
    for p in ns["cc_polys"]
]

ns["cc_polys"][0][0, 0] = _local_center_x

ns["cc_polys"][1][0, 0] = _local_center_x


print(
    "WT cleanup: fused IOC seam pairs:",
    len(_fuse_pairs)
)

print(
    "WT cleanup: IOC polygons:",
    len(_original_ioc_polys),
    "->",
    len(ns["sheet_polys"])
)


# ============================================================
# CREATE PPC POLYGONS
# ============================================================

ioc_union = unary_union(
    [
        Polygon(p)
        for p in ns["sheet_polys"]
    ]
)

holes = [
    Polygon(r)
    for r in ioc_union.interiors
]

omm_centres = [
    np.asarray(o, float)
    for o in ns["omm_centres"]
]


# Match every hole to an ommatidium

used = set()

hole_by_center = []

for o in omm_centres:

    idx = min(
        (
            i
            for i in range(len(holes))
            if i not in used
        ),
        key=lambda i:
        holes[i].centroid.distance(
            Point(*o)
        )
    )

    used.add(idx)

    hole_by_center.append(
        holes[idx]
    )


cells = []


# ------------------------------------------------------------
# IOC cells
# ------------------------------------------------------------

for idx, p in enumerate(
    ns["sheet_polys"]
):

    cells.append(
        {
            "type": "IOC",
            "subtype": "IOC",
            "poly": orient(
                Polygon(p),
                sign=1.0
            ),
            "name": f"IOC{idx}",
        }
    )


# ------------------------------------------------------------
# CC and PPC cells
# ------------------------------------------------------------

for oi, o in enumerate(
    omm_centres
):

    ccp = []

    subtypes = [
        "CC_top",
        "CC_bottom",
        "CC_left",
        "CC_right"
    ]

    for k, p in enumerate(
        ns["cc_polys"]
    ):

        poly = orient(
            Polygon(
                np.asarray(
                    p,
                    float
                )
                +
                ns["_CC_DROP"]
                +
                o
            ),
            sign=1.0
        )

        ccp.append(poly)

        cells.append(
            {
                "type": "CC",
                "subtype": subtypes[k],
                "poly": poly,
                "name": f"OM{oi}_{subtypes[k]}",
                "omm": oi,
            }
        )


    ring = hole_by_center[oi].difference(
        unary_union(ccp)
    )


    top_seg = LineString(
        [
            ns["TOP"]
            +
            ns["_CC_DROP"]
            +
            o,

            np.array(
                [0.0, 2.420]
            )
            +
            o,
        ]
    )


    bottom_seg = LineString(
        [
            ns["BOTTOM"]
            +
            ns["_CC_DROP"]
            +
            o,

            np.array(
                [0.0, -2.656]
            )
            +
            o,
        ]
    )


    pieces = list(
        split(
            ring,
            MultiLineString(
                [
                    top_seg,
                    bottom_seg
                ]
            )
        ).geoms
    )


    if len(pieces) != 2:

        raise RuntimeError(
            f"Could not split PPC pocket for ommatidium {oi}"
        )


    for g in pieces:

        side = (
            "PPC_R"
            if g.centroid.x > o[0]
            else
            "PPC_L"
        )

        cells.append(
            {
                "type": "PPC",
                "subtype": side,
                "poly": orient(
                    g,
                    sign=1.0
                ),
                "name": f"OM{oi}_{side}",
                "omm": oi,
            }
        )


# ============================================================
# GLOBAL VERTEX REGISTRY
# ============================================================

verts = []

vmap = {}

cell_vids = []


def get_vid(pt):

    key = tuple(
        np.round(
            np.asarray(
                pt,
                float
            ),
            6
        )
    )

    if key not in vmap:

        vmap[key] = len(verts)

        verts.append(
            np.asarray(
                pt,
                float
            )
        )

    return vmap[key]


for c in cells:

    coords = np.asarray(
        c["poly"].exterior.coords[:-1],
        float
    )

    vids = [
        get_vid(p)
        for p in coords
    ]

    cell_vids.append(
        np.asarray(
            vids,
            dtype=int
        )
    )


V0 = np.asarray(
    verts,
    float
)


# ============================================================
# EDGE REGISTRY
# ============================================================

edges = {}

for ci, vids in enumerate(
    cell_vids
):

    for k in range(
        len(vids)
    ):

        a = int(
            vids[k]
        )

        b = int(
            vids[
                (k + 1)
                %
                len(vids)
            ]
        )

        key = (
            (a, b)
            if a < b
            else
            (b, a)
        )

        edges.setdefault(
            key,
            {"cells": []}
        )["cells"].append(ci)


# ============================================================
# EDGE CLASSIFICATION
# ============================================================

def nearest_omm(mid):

    return min(
        omm_centres,
        key=lambda o:
        np.linalg.norm(
            mid - o
        )
    )


def classify_edge(
    key,
    info
):

    cids = info["cells"]

    if len(cids) == 1:

        return "BOUNDARY"


    c1 = cells[cids[0]]

    c2 = cells[cids[1]]


    types = {
        c1["type"],
        c2["type"]
    }


    # CC-CC

    if types == {"CC"}:

        subs = {
            c1["subtype"],
            c2["subtype"]
        }

        return (
            "CC-CC central"
            if
            subs
            ==
            {
                "CC_top",
                "CC_bottom"
            }
            else
            "CC-CC side"
        )


    # CC-PPC

    if types == {
        "CC",
        "PPC"
    }:

        cc = (
            c1
            if c1["type"] == "CC"
            else c2
        )

        return (
            "CC-PPC AP"
            if
            cc["subtype"]
            in
            (
                "CC_left",
                "CC_right"
            )
            else
            "CC-PPC EqPl"
        )


    # PPC-PPC

    if types == {"PPC"}:

        return "PPC-PPC"


    # PPC-IOC

    if types == {
        "PPC",
        "IOC"
    }:

        return "PPC-IOC"


    # IOC-IOC

    if types == {"IOC"}:

        a, b = key

        mid = (
            V0[a]
            +
            V0[b]
        ) / 2.0

        d = (
            mid
            -
            nearest_omm(mid)
        )

        return (
            "IOC-IOC AP"
            if
            abs(d[0])
            >=
            abs(d[1])
            else
            "IOC-IOC EqPl"
        )


    return "all else"


for key, info in edges.items():

    info["class"] = classify_edge(
        key,
        info
    )


# ============================================================
# TABLE S1 VALUES
# ============================================================

TABLE = {

    "CC-CC side":
    {
        "cad": 1.00,
        "cmyo": 1.00
    },

    "CC-CC central":
    {
        "cad": 1.08,
        "cmyo": 1.06
    },

    "CC-PPC EqPl":
    {
        "cad": 1.79,
        "cmyo": 1.52
    },

    "CC-PPC AP":
    {
        "cad": 2.11,
        "cmyo": 1.52
    },

    "PPC-PPC":
    {
        "cad": 3.25,
        "cmyo": 1.33
    },

    "PPC-IOC":
    {
        "cad": 4.83,
        "cmyo": 1.93
    },

    "IOC-IOC AP":
    {
        "cad": 2.37,
        "cmyo": 2.24,
        "S_ad": 0.19,
        "S_myo": 1.33
    },

    "IOC-IOC EqPl":
    {
        "cad": 2.24,
        "cmyo": 2.30,
        "S_ad": 0.29,
        "S_myo": 2.02
    },

    "all else":
    {
        "cad": 1.00,
        "cmyo": 1.00
    },
}


def edge_lambda(
    edge_class,
    w_ad
):

    if edge_class == "BOUNDARY":

        return 0.0


    w_myo = 1.0 - w_ad


    row = TABLE.get(
        edge_class,
        TABLE["all else"]
    )


    if "S_ad" in row:

        S = (
            w_ad
            *
            row["S_ad"]
            +
            w_myo
            *
            row["S_myo"]
        )

    else:

        S = (
            w_myo
            *
            row["cmyo"]
            +
            w_ad
            /
            row["cad"]
        )


    return (
        LAMBDA0
        *
        S
    )


# ============================================================
# PPC ACTOMYOSIN MESH
# ============================================================

def closest_attachment(
    point,
    allowed_classes
):

    p = np.asarray(
        point,
        float
    )

    best = None


    for (a, b), info in edges.items():

        if (
            info["class"]
            not in allowed_classes
        ):
            continue


        va = V0[a]

        vb = V0[b]


        d = vb - va


        L2 = float(
            np.dot(
                d,
                d
            )
        )


        if L2 < 1e-12:
            continue


        t = float(
            np.clip(
                np.dot(
                    p - va,
                    d
                )
                /
                L2,
                0.0,
                1.0
            )
        )


        q = (
            va
            +
            t * d
        )


        dist = float(
            np.linalg.norm(
                p - q
            )
        )


        if (
            best is None
            or
            dist < best[0]
        ):

            best = (
                dist,
                a,
                b,
                t
            )


    return best


mesh_springs = []


for oi, o in enumerate(
    omm_centres
):

    for si, (a, b) in enumerate(
        ns["_dash"]
    ):

        p_in = (
            np.asarray(
                a,
                float
            )
            +
            ns["_CC_DROP"]
            +
            o
        )


        p_out = (
            np.asarray(
                b,
                float
            )
            +
            o
        )


        ai = closest_attachment(
            p_in,
            {
                "CC-PPC AP",
                "CC-PPC EqPl"
            }
        )


        ao = closest_attachment(
            p_out,
            {
                "PPC-IOC"
            }
        )


        if (
            ai is None
            or
            ao is None
            or
            ai[0] > 0.05
            or
            ao[0] > 0.05
        ):

            raise RuntimeError(
                f"Could not attach dashed bond OM{oi} #{si}"
            )


        mesh_springs.append(
            {
                "inner":
                (
                    ai[1],
                    ai[2],
                    ai[3]
                ),

                "outer":
                (
                    ao[1],
                    ao[2],
                    ao[3]
                ),

                "omm": oi,
            }
        )


# ============================================================
# THIN ONLY THE SIDE IOC BAND
# ============================================================

_vertex_proposals = {}


for (_a, _b), _info in edges.items():

    if _info["class"] != "PPC-IOC":

        continue


    _ppc_cell = None


    for _cid in _info["cells"]:

        if (
            cells[_cid]["type"]
            ==
            "PPC"
        ):

            _ppc_cell = cells[_cid]

            break


    if _ppc_cell is None:

        continue


    _o = np.asarray(
        omm_centres[
            _ppc_cell["omm"]
        ],
        float
    )


    _mid = (
        0.5
        *
        (
            V0[_a]
            +
            V0[_b]
        )
    )


    _dmid = (
        _mid
        -
        _o
    )


    # ONLY lateral left/right contacts.
    # Top/bottom remain unchanged.

    if (
        abs(_dmid[0])
        <
        abs(_dmid[1])
    ):

        continue


    for _v in (
        _a,
        _b
    ):

        _old = V0[_v].copy()


        _new = (

            _o

            +

            SIDE_PPC_IOC_RADIAL_SCALE

            *

            (
                _old
                -
                _o
            )

        )


        _vertex_proposals.setdefault(
            _v,
            []
        ).append(
            _new
        )


for _v, _props in _vertex_proposals.items():

    V0[_v] = np.mean(
        np.asarray(
            _props
        ),
        axis=0
    )


print(
    f"Side-only thin-IOC trial: moved "
    f"{len(_vertex_proposals)} PPC-IOC boundary vertices "
    f"with radial scale "
    f"{SIDE_PPC_IOC_RADIAL_SCALE:.3f}; "
    f"top/bottom IOC rows unchanged."
)


# ============================================================
# FIX OUTER BOUNDARY
# ============================================================

fixed = set()


for (a, b), info in edges.items():

    if len(
        info["cells"]
    ) == 1:

        fixed.add(a)

        fixed.add(b)


# ============================================================
# PPC-PPC VERTICAL CONSTRAINT
# ============================================================

_ncoord = (
    2
    *
    len(V0)
)


_parent = np.arange(
    _ncoord,
    dtype=int
)


def _find(x):

    while (
        _parent[x]
        !=
        x
    ):

        _parent[x] = _parent[
            _parent[x]
        ]

        x = _parent[x]

    return x


def _union(
    a,
    b
):

    ra = _find(a)

    rb = _find(b)


    if ra != rb:

        _parent[rb] = ra


for (_a, _b), _info in edges.items():

    if (
        _info["class"]
        ==
        "PPC-PPC"
    ):

        _union(
            2 * _a,
            2 * _b
        )


_fixed_coords = set()


for _v in fixed:

    _fixed_coords.add(
        2 * _v
    )

    _fixed_coords.add(
        2 * _v + 1
    )


_root_fixed = {}


for _ci in range(
    _ncoord
):

    _r = _find(_ci)

    if (
        _ci
        in
        _fixed_coords
    ):

        _root_fixed[_r] = (
            V0.reshape(-1)[_ci]
        )


_root_to_var = {}

_coord_to_var = np.full(
    _ncoord,
    -1,
    dtype=int
)

_z0 = []

_flat0 = V0.reshape(-1)


for _ci in range(
    _ncoord
):

    _r = _find(_ci)


    if _r in _root_fixed:

        continue


    if _r not in _root_to_var:

        _root_to_var[_r] = len(
            _z0
        )

        _z0.append(
            float(
                _flat0[_ci]
            )
        )


    _coord_to_var[_ci] = (
        _root_to_var[_r]
    )


x0 = np.asarray(
    _z0,
    float
)


def _unpack(x):

    flat = _flat0.copy()


    for ci, vi in enumerate(
        _coord_to_var
    ):

        if vi >= 0:

            flat[ci] = x[vi]


    return flat.reshape(
        (-1, 2)
    )


def _pack_gradient(G):

    gf = np.asarray(
        G,
        float
    ).reshape(-1)


    gz = np.zeros(
        len(x0),
        float
    )


    mask = (
        _coord_to_var
        >=
        0
    )


    np.add.at(
        gz,
        _coord_to_var[mask],
        gf[mask]
    )


    return gz


free_ids = np.asarray(
    [
        i
        for i in range(
            len(V0)
        )
        if
        i not in fixed
    ],
    dtype=int
)


# ============================================================
# AREA / EDGE BOOKKEEPING
# ============================================================

loc_cell = []

loc_vid = []

loc_prev = []

loc_next = []

area_edge_from = []

area_edge_to = []

area_edge_cell = []


for ci, vids in enumerate(
    cell_vids
):

    n = len(vids)


    for k, v in enumerate(
        vids
    ):

        loc_cell.append(ci)

        loc_vid.append(v)

        loc_prev.append(
            vids[
                (k - 1)
                %
                n
            ]
        )

        loc_next.append(
            vids[
                (k + 1)
                %
                n
            ]
        )

        area_edge_from.append(v)

        area_edge_to.append(
            vids[
                (k + 1)
                %
                n
            ]
        )

        area_edge_cell.append(ci)


loc_cell = np.asarray(
    loc_cell,
    int
)

loc_vid = np.asarray(
    loc_vid,
    int
)

loc_prev = np.asarray(
    loc_prev,
    int
)

loc_next = np.asarray(
    loc_next,
    int
)

area_edge_from = np.asarray(
    area_edge_from,
    int
)

area_edge_to = np.asarray(
    area_edge_to,
    int
)

area_edge_cell = np.asarray(
    area_edge_cell,
    int
)


edge_keys = list(
    edges.keys()
)


edge_a = np.asarray(
    [
        a
        for a, b in edge_keys
    ],
    int
)


edge_b = np.asarray(
    [
        b
        for a, b in edge_keys
    ],
    int
)


edge_classes = np.asarray(
    [
        edges[k]["class"]
        for k in edge_keys
    ],
    object
)


nonboundary = (
    edge_classes
    !=
    "BOUNDARY"
)


mi_a = np.asarray(
    [
        s["inner"][0]
        for s in mesh_springs
    ],
    int
)


mi_b = np.asarray(
    [
        s["inner"][1]
        for s in mesh_springs
    ],
    int
)


mi_t = np.asarray(
    [
        s["inner"][2]
        for s in mesh_springs
    ],
    float
)


mo_a = np.asarray(
    [
        s["outer"][0]
        for s in mesh_springs
    ],
    int
)


mo_b = np.asarray(
    [
        s["outer"][1]
        for s in mesh_springs
    ],
    int
)


mo_t = np.asarray(
    [
        s["outer"][2]
        for s in mesh_springs
    ],
    float
)


# ============================================================
# IDEAL AREAS
# ============================================================

_vf0 = V0[
    area_edge_from
]

_vt0 = V0[
    area_edge_to
]


_cross0 = (

    _vf0[:, 0]

    *

    _vt0[:, 1]

    -

    _vt0[:, 0]

    *

    _vf0[:, 1]

)


A0_geometry = (
    0.5
    *
    np.bincount(
        area_edge_cell,
        weights=_cross0,
        minlength=len(cells)
    )
)


A0_paper = []


for c in cells:

    if (
        c["type"]
        ==
        "IOC"
    ):

        A0_paper.append(
            1.0
        )


    elif (
        c["type"]
        ==
        "PPC"
    ):

        A0_paper.append(
            8.7
        )


    elif (
        c["subtype"]
        in
        (
            "CC_top",
            "CC_bottom"
        )
    ):

        A0_paper.append(
            1.3
        )


    else:

        A0_paper.append(
            1.0
        )


A0_paper = np.asarray(
    A0_paper,
    float
)


# ============================================================
# ENERGY FUNCTION
# ============================================================

def make_energy(
    w_ad,
    mesh_fraction,
    area_mode
):

    lambdas = np.asarray(
        [
            edge_lambda(
                c,
                w_ad
            )
            for c in edge_classes
        ],
        float
    )


    A0 = (
        A0_geometry
        if
        area_mode
        ==
        "geometry"
        else
        A0_paper
    )


    Tmesh = (
        LAMBDA0
        *
        mesh_fraction
    )


    def energy_grad(x):

        V = _unpack(x)

        G = np.zeros_like(V)


        # ----------------------------------------------------
        # AREA ENERGY
        # ----------------------------------------------------

        vf = V[
            area_edge_from
        ]

        vt = V[
            area_edge_to
        ]


        cross = (

            vf[:, 0]

            *

            vt[:, 1]

            -

            vt[:, 0]

            *

            vf[:, 1]

        )


        areas = (

            0.5

            *

            np.bincount(
                area_edge_cell,
                weights=cross,
                minlength=len(cells)
            )

        )


        dA = (
            areas
            -
            A0
        )


        E = (

            0.5

            *

            K_AREA

            *

            float(
                np.dot(
                    dA,
                    dA
                )
            )

        )


        coeff = (

            K_AREA

            *

            dA[
                loc_cell
            ]

        )


        gx = (

            0.5

            *

            (
                V[
                    loc_next,
                    1
                ]

                -

                V[
                    loc_prev,
                    1
                ]
            )

            *

            coeff

        )


        gy = (

            0.5

            *

            (
                V[
                    loc_prev,
                    0
                ]

                -

                V[
                    loc_next,
                    0
                ]
            )

            *

            coeff

        )


        np.add.at(
            G[:, 0],
            loc_vid,
            gx
        )


        np.add.at(
            G[:, 1],
            loc_vid,
            gy
        )


        # ----------------------------------------------------
        # JUNCTION ENERGY
        # ----------------------------------------------------

        d = (

            V[edge_a]

            -

            V[edge_b]

        )


        lengths = np.linalg.norm(
            d,
            axis=1
        )


        safe = np.maximum(
            lengths,
            1e-12
        )


        E += float(
            np.dot(
                lambdas,
                lengths
            )
        )


        ge = (

            lambdas
            /
            safe

        )[:, None] * d


        np.add.at(
            G,
            edge_a,
            ge
        )


        np.add.at(
            G,
            edge_b,
            -ge
        )


        # ----------------------------------------------------
        # PPC ACTOMYOSIN MESH ENERGY
        # ----------------------------------------------------

        if Tmesh > 0:

            pin = (

                (1 - mi_t)[:, None]

                *

                V[mi_a]

                +

                mi_t[:, None]

                *

                V[mi_b]

            )


            pout = (

                (1 - mo_t)[:, None]

                *

                V[mo_a]

                +

                mo_t[:, None]

                *

                V[mo_b]

            )


            dm = (
                pin
                -
                pout
            )


            lm = np.linalg.norm(
                dm,
                axis=1
            )


            sm = np.maximum(
                lm,
                1e-12
            )


            E += (

                Tmesh

                *

                float(
                    np.sum(lm)
                )

            )


            gm = (

                Tmesh

                *

                dm

                /

                sm[:, None]

            )


            np.add.at(

                G,

                mi_a,

                (1 - mi_t)[:, None]

                *

                gm

            )


            np.add.at(

                G,

                mi_b,

                mi_t[:, None]

                *

                gm

            )


            np.add.at(

                G,

                mo_a,

                -(1 - mo_t)[:, None]

                *

                gm

            )


            np.add.at(

                G,

                mo_b,

                -mo_t[:, None]

                *

                gm

            )


        return (
            E,
            _pack_gradient(G)
        )


    return energy_grad


# ============================================================
# AREA CALCULATION
# ============================================================

def calculate_areas(V):

    vf = V[
        area_edge_from
    ]

    vt = V[
        area_edge_to
    ]


    cross = (

        vf[:, 0]

        *

        vt[:, 1]

        -

        vt[:, 0]

        *

        vf[:, 1]

    )


    return (

        0.5

        *

        np.bincount(
            area_edge_cell,
            weights=cross,
            minlength=len(cells)
        )

    )


# ============================================================
# RUN OPTIMIZER
# ============================================================

def run_one(
    w_ad,
    mesh,
    area_mode
):

    fg = make_energy(
        w_ad,
        mesh,
        area_mode
    )


    E0, g0 = fg(
        x0
    )


    result = minimize(

        fg,

        x0,

        method="L-BFGS-B",

        jac=True,

        options={
            "maxiter": 400,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 30
        }

    )


    V = _unpack(
        result.x
    )


    disp = np.linalg.norm(
        V - V0,
        axis=1
    )[free_ids]


    edge_lengths = np.linalg.norm(
        V[edge_a]
        -
        V[edge_b],
        axis=1
    )


    active_lengths = edge_lengths[
        nonboundary
    ]


    Afinal = calculate_areas(V)

    Astart = calculate_areas(V0)


    summary = {

        "area_mode":
        area_mode,

        "w_ad":
        w_ad,

        "w_myo":
        1.0 - w_ad,

        "mesh_fraction":
        mesh,

        "optimizer_success":
        bool(
            result.success
        ),

        "iterations":
        int(
            result.nit
        ),

        "energy_start":
        E0,

        "energy_final":
        float(
            result.fun
        ),

        "initial_force_norm":
        float(
            np.linalg.norm(
                g0
            )
        ),

        "rms_vertex_displacement":
        float(
            np.sqrt(
                np.mean(
                    disp ** 2
                )
            )
        ),

        "mean_vertex_displacement":
        float(
            np.mean(
                disp
            )
        ),

        "max_vertex_displacement":
        float(
            np.max(
                disp
            )
        ),

        "minimum_internal_edge":
        float(
            np.min(
                active_lengths
            )
        ),

        "edges_below_0.02":
        int(
            np.sum(
                active_lengths
                <
                0.02
            )
        ),

        "mean_abs_area_change_percent":
        float(
            np.mean(
                np.abs(
                    (
                        Afinal
                        -
                        Astart
                    )
                    /
                    Astart
                )
            )
            *
            100
        ),

        "max_ppc_ppc_dx":
        float(
            max(
                [
                    abs(
                        V[a, 0]
                        -
                        V[b, 0]
                    )
                    for
                    (a, b),
                    info
                    in
                    edges.items()
                    if
                    info["class"]
                    ==
                    "PPC-PPC"
                ]
                or
                [0.0]
            )
        ),
    }


    return (
        result,
        V,
        summary
    )


# ============================================================
# RUN WT CONDITION
# ============================================================

all_rows = []

solutions = {}


for area_mode in AREA_MODES:

    for w_ad in W_AD_VALUES:

        for mesh in MESH_VALUES:

            result, V, row = run_one(
                w_ad,
                mesh,
                area_mode
            )

            all_rows.append(
                row
            )

            solutions[
                (
                    area_mode,
                    w_ad,
                    mesh
                )
            ] = V


results = pd.DataFrame(
    all_rows
)


results.to_csv(
    OUTDIR
    /
    "WT_parameter_grid.csv",
    index=False
)


# ============================================================
# COLORS
# ============================================================

COLORS = {

    "CC-CC central":
    "#B30000",

    "CC-CC side":
    "#508C46",

    "CC-PPC AP":
    "#0056B6",

    "CC-PPC EqPl":
    "#54B9D1",

    "PPC-PPC":
    "#E040E0",

    "PPC-IOC":
    "#8E2C8E",

    "IOC-IOC AP":
    "#B8860B",

    "IOC-IOC EqPl":
    "#D5A72A",

    "BOUNDARY":
    "#222222",
}


# Fill colors

IOC_FILL = "#E6E6E6"

CC_FILL = "#C9DDF5"

PPC_FILL = "#FFFFFF"


# ============================================================
# HIDE ONLY THE FOUR GOLD SEAMS IN THE FINAL IMAGE
# ============================================================
#
# IMPORTANT:
# This changes ONLY the displayed figure.
#
# It DOES NOT remove these edges from the simulation mechanics.
#
# This keeps the stable WT simulation while displaying the IOC
# regions as fused in the figure.
# ============================================================

HIDE_GOLD_MIDPOINTS = np.asarray(
    [
        [-2.15,  1.46],
        [ 2.15,  1.46],
        [-2.15, -1.70],
        [ 2.15, -1.70],
    ],
    float
)


def _hide_this_display_edge(
    p,
    q,
    edge_class
):

    if edge_class not in (
        "IOC-IOC AP",
        "IOC-IOC EqPl"
    ):

        return False


    mid = (

        0.5

        *

        (
            np.asarray(
                p,
                float
            )

            +

            np.asarray(
                q,
                float
            )
        )

    )


    distance = np.min(

        np.linalg.norm(

            HIDE_GOLD_MIDPOINTS

            -

            mid,

            axis=1

        )

    )


    return (
        distance
        <
        0.12
    )


# ============================================================
# DRAW FINAL COLORED WT
# ============================================================

def draw_colored_wt(
    V,
    area_mode,
    w_ad,
    mesh
):

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )


    ax.set_aspect(
        "equal"
    )


    ax.set_facecolor(
        "white"
    )


    # --------------------------------------------------------
    # Fill cells
    # --------------------------------------------------------

    for ci, c in enumerate(
        cells
    ):

        pts = V[
            cell_vids[ci]
        ]


        if (
            pts[:, 0].max() < -6
            or
            pts[:, 0].min() > 6
            or
            pts[:, 1].max() < -6
            or
            pts[:, 1].min() > 6
        ):

            continue


        if (
            c["type"]
            ==
            "IOC"
        ):

            face = IOC_FILL


        elif (
            c["type"]
            ==
            "CC"
        ):

            face = CC_FILL


        else:

            face = PPC_FILL


        ax.add_patch(

            plt.Polygon(

                pts,

                closed=True,

                facecolor=face,

                edgecolor="none",

                zorder=0

            )

        )


    # --------------------------------------------------------
    # Draw junctions
    # --------------------------------------------------------

    for (a, b), info in edges.items():

        p = V[a]

        q = V[b]


        if (
            max(
                p[0],
                q[0]
            ) < -6

            or

            min(
                p[0],
                q[0]
            ) > 6

            or

            max(
                p[1],
                q[1]
            ) < -6

            or

            min(
                p[1],
                q[1]
            ) > 6
        ):

            continue


        # Hide only the four requested gold seams
        # in the DISPLAY.

        if _hide_this_display_edge(
            p,
            q,
            info["class"]
        ):

            continue


        ax.plot(

            [
                p[0],
                q[0]
            ],

            [
                p[1],
                q[1]
            ],

            color=
            COLORS[
                info["class"]
            ],

            lw=1.7,

            zorder=2,

            solid_capstyle="round"

        )


    # --------------------------------------------------------
    # Red dashed PPC actomyosin mesh
    # --------------------------------------------------------

    for s in mesh_springs:

        o = omm_centres[
            s["omm"]
        ]


        if not (
            -7 < o[0] < 7
            and
            -7 < o[1] < 7
        ):

            continue


        ai, bi, ti = s[
            "inner"
        ]


        ao, bo, to = s[
            "outer"
        ]


        pin = (

            (1 - ti)

            *

            V[ai]

            +

            ti

            *

            V[bi]

        )


        pout = (

            (1 - to)

            *

            V[ao]

            +

            to

            *

            V[bo]

        )


        ax.plot(

            [
                pin[0],
                pout[0]
            ],

            [
                pin[1],
                pout[1]
            ],

            color="#E8452C",

            lw=1.5,

            ls="--",

            dashes=(4, 3),

            zorder=3

        )


    # --------------------------------------------------------
    # Crop/display
    # --------------------------------------------------------

    ax.set_xlim(
        -5.5,
        5.5
    )


    ax.set_ylim(
        -5.5,
        5.5
    )


    ax.axis(
        "off"
    )


    fig.tight_layout(
        pad=0
    )


    outfile = (

        OUTDIR

        /

        (
            f"WT_final_visual_"
            f"wad{w_ad:.1f}_"
            f"wmyo{1-w_ad:.1f}_"
            f"mesh{mesh:.2f}.png"
        )

    )


    fig.savefig(

        outfile,

        dpi=300,

        facecolor="white",

        bbox_inches="tight",

        pad_inches=0.02

    )


    plt.close(fig)


    print(
        "Saved colored WT figure:",
        outfile
    )


# ============================================================
# DRAW
# ============================================================

for area_mode in AREA_MODES:

    V = solutions[
        (
            area_mode,
            OVERLAY_W_AD,
            OVERLAY_MESH
        )
    ]


    draw_colored_wt(
        V,
        area_mode,
        OVERLAY_W_AD,
        OVERLAY_MESH
    )


# ============================================================
# SAVE CONTACT COUNTS
# ============================================================

counts = Counter(
    info["class"]
    for info in edges.values()
)


pd.DataFrame(
    sorted(
        counts.items()
    ),
    columns=[
        "contact_class",
        "edge_count"
    ]
).to_csv(
    OUTDIR
    /
    "contact_class_counts.csv",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("Done.")

print(
    "Original geometry file was not modified."
)

print(
    f"Cells: {len(cells)} | "
    f"vertices: {len(V0)} | "
    f"fixed boundary vertices: {len(fixed)}"
)

print(
    f"Existing dashed PPC actomyosin bonds: "
    f"{len(mesh_springs)}"
)

print(
    f"Outputs: {OUTDIR}"
)

print(
    "Representative geometry-mode row:"
)

print(

    results[

        (
            results.area_mode
            ==
            "geometry"
        )

        &

        (
            results.w_ad
            ==
            OVERLAY_W_AD
        )

        &

        (
            results.mesh_fraction
            ==
            OVERLAY_MESH
        )

    ].to_string(
        index=False
    )

)