"""
tiled_sheet_v4.py — a sheet of ommatidia sharing their interommatidial cells.

The single-cluster geometry (cone cells with their coloured seams, 19 grey IOC
cells whose inner-facing edges are purple, magenta separators top and bottom,
and the red dashed inter-primary connectors) is stamped on the OMM_A / OMM_B
lattice. Cells that land on one another are merged, nearby vertices are welded
so shared edges coincide, and slivers left at the seams are absorbed into the
neighbour -- pulled inward, so each cluster keeps its pocket width and stays
left-right symmetric.

Spacing is set so exactly ONE layer of interommatidial cells sits between
neighbouring clusters. Vertical walls that only exist because a small
four-sided notch cell survived the merge are not drawn, so those regions read
as single hexagons.

Knobs worth touching:
  OMM_A / OMM_B  ommatidium lattice vectors (cluster spacing)
  N_OMM          how many ommatidia out from the centre to draw
  CELL_TOL       cells closer than this are treated as the same cell
  VERT_TOL       vertices closer than this get welded together
  _lim           half-width of the plotted area
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator

R = 3.0  # big hex "radius" (center to corner)

def polar(theta_deg, r):
    t = np.radians(theta_deg)
    return np.array([r * np.cos(t), r * np.sin(t)])

def hex_pts(center, radius, angle_offset=90):
    angs = [angle_offset + k * 60 for k in range(6)]
    return np.array([np.array(center) + polar(a, radius) for a in angs]), angs

# ---------------------------------------------------------------------------
# Big hex (pointy-top): corners at 90,30,-30,-90,-150,150
# ---------------------------------------------------------------------------
hex_corners, hex_corner_angles = hex_pts((0, 0), R, angle_offset=90)
hex_edge_mids = np.array([(hex_corners[i] + hex_corners[(i + 1) % 6]) / 2 for i in range(6)])
hex_edge_mid_angles = [np.degrees(np.arctan2(p[1], p[0])) for p in hex_edge_mids]



# ---------------------------------------------------------------------------
# 4 CC cells: EXACT topology traced from the reference photo's own contour
# (not a symmetric 4-fold pinwheel). TOP & BOTTOM are pentagons sharing a
# short central edge (A-B, red). LEFT & RIGHT are hexagons flanking them,
# meeting TOP/BOTTOM along the cyan / green seams.
# Coordinates below are the reference image's traced vertices, recentred on
# the shared point O and rescaled by k into this drawing's units.
# ---------------------------------------------------------------------------
SCALE = 1.45 / 164.5  # reference-pixels -> this drawing's units (matched by measurement)

TOP          = np.array([2.5, 144]) * SCALE
BOTTOM       = np.array([1.5, -148]) * SCALE
A            = np.array([-40.5, 0]) * SCALE
B            = np.array([40.5, 0]) * SCALE
UL_shoulder  = np.array([-85.5, 75]) * SCALE
UR_shoulder  = np.array([88.5, 75]) * SCALE
LL_shoulder  = np.array([-84.5, -78]) * SCALE
LR_shoulder  = np.array([88.5, -77]) * SCALE
UL_outer     = np.array([-148.5, 71]) * SCALE
LL_outer     = np.array([-149.5, -73]) * SCALE
LEFT_MID     = np.array([-163.5, -1]) * SCALE

# right wing: vertex removed, edge1 slanted more, edge2 now a single vertical
# edge (measured from the user's annotation, both endpoints share x=1.416)
V1 = np.array([1.416, 0.380])   # end of edge1 / start of edge2
V2 = np.array([1.416, -0.380])  # end of edge2 / start of edge3
# left wing: mirror of the right wing's fix (same construction, negated x)
V1_L = np.array([-1.416, 0.380])
V2_L = np.array([-1.416, -0.380])

cc_polys = [
    np.array([TOP, UR_shoulder, B, A, UL_shoulder]),                              # top pentagon
    np.array([BOTTOM, LL_shoulder, A, B, LR_shoulder]),                           # bottom pentagon
    np.array([V1_L, V2_L, LL_shoulder, A, UL_shoulder]),                          # left wing (pentagon)
    np.array([V1, V2, LR_shoulder, B, UR_shoulder]),                             # right wing (pentagon)
]

SEAM_EDGES = [
    (A, B, "#B30000"),
    (UL_shoulder, A, "#508C46"), (A, LL_shoulder, "#508C46"),
    (UR_shoulder, B, "#508C46"), (B, LR_shoulder, "#508C46"),
]

# ---------------------------------------------------------------------------
# CC cluster's own outer silhouette (10 edges, excludes the internal seams),
# used so the PPC petals' inner edges can be extended to actually TOUCH the
# cone-cell cluster instead of stopping short at a floating inner radius.
# ---------------------------------------------------------------------------
cc_boundary_edges = [
    (TOP, UR_shoulder), (UR_shoulder, V1), (V1, V2), (V2, LR_shoulder),
    (LR_shoulder, BOTTOM), (BOTTOM, LL_shoulder), (LL_shoulder, V2_L),
    (V2_L, V1_L), (V1_L, UL_shoulder), (UL_shoulder, TOP),
]

def ray_hit(origin, direction, edges):
    """First point where a ray (origin + t*direction, t>0) crosses any edge in `edges`."""
    best_t, best_pt = None, None
    for p1, p2 in edges:
        e = p2 - p1
        denom = direction[0] * e[1] - direction[1] * e[0]
        if abs(denom) < 1e-9:
            continue
        t = (p1[0] * e[1] - p1[1] * e[0] - origin[0] * e[1] + origin[1] * e[0]) / denom
        u = ((origin[0] + direction[0] * t - p1[0]) * e[0] + (origin[1] + direction[1] * t - p1[1]) * e[1]) / (e[0] ** 2 + e[1] ** 2)
        if t > 1e-6 and -1e-6 <= u <= 1 + 1e-6:
            if best_t is None or t < best_t:
                best_t, best_pt = t, origin + direction * t
    return best_pt

# ---------------------------------------------------------------------------
# PPC ring: 10 individual petal cells (5 right, 5 left) instead of 8 plain
# rays through one shared hexagon. Anchors are 10 EVENLY spaced angles
# (36 deg apart), so top(90) and bottom(-90) are shared between the two
# sides and each side gets exactly 5 petals between them.
# ---------------------------------------------------------------------------
def hex_boundary_point(angle_deg):
    """Where a ray from the origin at angle_deg exits the big hexagon."""
    d = polar(angle_deg, 1.0)
    for i in range(6):
        p1, p2 = hex_corners[i], hex_corners[(i + 1) % 6]
        e = p2 - p1
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-9:
            continue
        t = (p1[0] * e[1] - p1[1] * e[0]) / denom
        u = (p1[0] * d[1] - p1[1] * d[0]) / denom
        if t > 0 and -1e-9 <= u <= 1 + 1e-9:
            return d * t
    raise ValueError(angle_deg)

ALL_ANGLES = [-162, -126, -90, -54, -18, 18, 54, 90, 126, 162]   # 10 evenly-spaced (36 deg apart) boundary angles, all the way around
R_PEAK = 2.42        # side-vertex radius -- identical for every petal, and the point actually SHARED with neighbors
R_TIP1 = 2.85        # tip1's own radius (was 2.75 -- extended a bit per direction)
R_TIP2 = 2.85        # tip2's own radius (was 2.75 -- extended a bit per direction)
TIP_NARROW = 6.0     # half-gap (deg) between the two tip vertices, around each petal's own angular center

def petal_outer(angle_deg):
    return polar(angle_deg, R_PEAK)

def symmetric_inner(angle_deg):
    """
    Inner point (touches the CC boundary directly), with EXACT 4-fold
    (both-axis) symmetry enforced: always computed via ray_hit against just
    the upper-right quadrant (0-90 deg), then reflected into whichever
    quadrant angle_deg actually falls in. This guarantees every top/bottom
    AND left/right petal pair is identical, even though the CC cluster's own
    traced boundary isn't perfectly symmetric in either axis.
    """
    a = angle_deg % 360
    if a <= 90:
        canon, hx, hy = a, 1, 1
    elif a <= 180:
        canon, hx, hy = 180 - a, -1, 1
    elif a <= 270:
        canon, hx, hy = a - 180, -1, -1
    else:
        canon, hx, hy = 360 - a, 1, -1
    p = ray_hit(np.array([0.0, 0.0]), polar(canon, 1.0), cc_boundary_edges)
    return np.array([hx * p[0], hy * p[1]])

def make_petal_quad(a0, a1):
    in0, out0 = symmetric_inner(a0), petal_outer(a0)
    in1, out1 = symmetric_inner(a1), petal_outer(a1)
    a1_unwrapped = a1 if a1 > a0 else a1 + 360   # handle the 162 -> -162 wraparound (else the naive average lands at 0 deg, the opposite side)
    ac = (a0 + a1_unwrapped) / 2.0
    tip1 = polar(ac - TIP_NARROW, R_TIP1)
    tip2 = polar(ac + TIP_NARROW, R_TIP2)
    poly = np.array([in0, out0, tip1, tip2, out1, in1])
    edges = [
        (in0, out0, "solid"),    # side edge -- shared with the previous petal
        (out0, tip1, "solid"),   # tip, first half
        (tip1, tip2, "solid"),   # tip's own short flat
        (tip2, out1, "solid"),   # tip, second half
        (out1, in1, "solid"),    # side edge -- shared with the next petal
        (in1, in0, "hidden"),    # inner edge -- CC cluster's own border already draws this
    ]
    return poly, edges

def make_petal_a_special(a0, a1, special_side):
    """
    "a"-type petal (2a and its mirror 1a): tip1 (on the special_side end)
    is shifted (same radius, different angle) so its vertical drop lands
    EXACTLY on out0/out1 -- the shared point with the neighboring "b"-type
    petal. No dashed connector, no gap: side edges are unified.
    special_side: "a0" (2a's case) or "a1" (1a's mirrored case).
    """
    in0, out0 = symmetric_inner(a0), petal_outer(a0)
    in1, out1 = symmetric_inner(a1), petal_outer(a1)
    a1_unwrapped = a1 if a1 > a0 else a1 + 360
    ac = (a0 + a1_unwrapped) / 2.0
    if special_side == "a0":
        theta1 = np.degrees(np.arccos(np.clip(out0[0] / R_TIP1, -1, 1)))
        theta1 = theta1 if (ac - TIP_NARROW) >= 0 else -theta1
        tip1 = polar(theta1, R_TIP1)
        shift = (ac - TIP_NARROW) - theta1
        tip2 = polar((ac + TIP_NARROW) - shift, R_TIP2)
        poly = np.array([in0, out0, tip1, tip2, out1, in1])
        edges = [
            (in0, out0, "solid"), (out0, tip1, "solid"), (tip1, tip2, "solid"),
            (tip2, out1, "solid"), (out1, in1, "solid"), (in1, in0, "hidden"),
        ]
    else:
        theta2 = np.degrees(np.arccos(np.clip(out1[0] / R_TIP2, -1, 1)))
        theta2 = theta2 if (ac + TIP_NARROW) >= 0 else -theta2
        tip2 = polar(theta2, R_TIP2)
        shift = theta2 - (ac + TIP_NARROW)
        tip1 = polar((ac - TIP_NARROW) + shift, R_TIP1)
        poly = np.array([in0, out0, tip1, tip2, out1, in1])
        edges = [
            (in0, out0, "solid"), (out0, tip1, "solid"), (tip1, tip2, "solid"),
            (tip2, out1, "solid"), (out1, in1, "solid"), (in1, in0, "hidden"),
        ]
    return poly, edges

def make_petal_b_special(a0, a1, special_side):
    """
    "b"-type petal (1b and its mirror 2b): tip1/tip2 (hence M, their
    midpoint) are shifted (same radius, different angle) so M's vertical
    drop lands EXACTLY on out0/out1 -- the shared point with the
    neighboring "c"-type (quad) petal. No dashed connector, no gap.
    special_side: "a1" (1b's case, modifies the 162 side) or "a0" (2b's
    mirrored case, modifies the 18 side).
    """
    in0, out0 = symmetric_inner(a0), petal_outer(a0)
    in1, out1 = symmetric_inner(a1), petal_outer(a1)
    a1_unwrapped = a1 if a1 > a0 else a1 + 360
    ac = (a0 + a1_unwrapped) / 2.0
    target = out0 if special_side != "a1" else out1
    # solve for M's angle such that a vertical line through M (at the
    # midpoint-of-tips radius) has the same x as target
    R_M = (R_TIP1 + R_TIP2) / 2.0
    theta_M = np.degrees(np.arccos(np.clip(target[0] / R_M, -1, 1)))
    theta_M = theta_M if ac >= 0 else -theta_M
    shift = ac - theta_M
    tip1 = polar((ac - TIP_NARROW) - shift, R_TIP1)
    tip2 = polar((ac + TIP_NARROW) - shift, R_TIP2)
    M = (tip1 + tip2) / 2.0
    if special_side == "a1":
        poly = np.array([in0, out0, M, out1, in1])
        edges = [
            (in0, out0, "solid"), (out0, M, "solid"), (M, out1, "solid"),
            (out1, in1, "solid"), (in1, in0, "hidden"),
        ]
    else:
        poly = np.array([in0, out0, M, out1, in1])
        edges = [
            (in0, out0, "solid"), (out0, M, "solid"), (M, out1, "solid"),
            (out1, in1, "solid"), (in1, in0, "hidden"),
        ]
    return poly, edges

ppc_petals, ppc_edges = [], []
n = len(ALL_ANGLES)
_A_SPECIAL = {(54, 90): "a0", (90, 126): "a1", (-90, -54): "a1", (-126, -90): "a0"}   # 2a/1a (upper) and their horizontal mirrors 4a/3a (lower)
_B_SPECIAL = {(126, 162): "a1", (18, 54): "a0", (-162, -126): "a0", (-54, -18): "a1"}   # 1b/2b (upper) and their horizontal mirrors 3b/4b (lower)
_OVERRIDDEN_2A = {(54, 90), (90, 126), (-126, -90), (-90, -54),   # 2a,1a,3a,4a
                  (18, 54), (126, 162), (-162, -126), (-54, -18),  # 2b,1b,3b,4b
                  (-18, 18), (162, -162)}   # 2c,1c -- replaced below
for i in range(n):
    a0, a1 = ALL_ANGLES[i], ALL_ANGLES[(i + 1) % n]
    if (a0, a1) in _OVERRIDDEN_2A:
        continue
    if (a0, a1) in _A_SPECIAL:
        poly, edges = make_petal_a_special(a0, a1, _A_SPECIAL[(a0, a1)])
    elif (a0, a1) in _B_SPECIAL:
        poly, edges = make_petal_b_special(a0, a1, _B_SPECIAL[(a0, a1)])
    else:
        poly, edges = make_petal_quad(a0, a1)
    ppc_petals.append(poly)
    ppc_edges.extend(edges)

# ---------------------------------------------------------------------------
# 2a override: session-confirmed coordinates. in0/out1/in1 unchanged; out0
# moved down (to 1.15,1.6) so edge b (out0-tip1) is vertical; tip1 at
# (1.15,2.42); tip2 at (0.63,2.6). "common F" = the in1-in0 chord (0.736,
# hidden) stays as-is -- this is what 2b will attach to next.
# ---------------------------------------------------------------------------
_2a_in0 = symmetric_inner(54)
_2a_out1 = petal_outer(90)
_2a_in1 = symmetric_inner(90)
_2a_out0 = np.array([1.15, 1.6])
_2a_tip1 = np.array([1.15, 2.42])
_2a_tip2 = np.array([0.63, 2.6])
poly_2a_final = np.array([_2a_in0, _2a_out0, _2a_tip1, _2a_tip2, _2a_out1, _2a_in1])
edges_2a_final = [
    (_2a_in0, _2a_out0, "dashed"), (_2a_out0, _2a_tip1, "solid"),
    (_2a_tip1, _2a_tip2, "solid"), (_2a_tip2, _2a_out1, "solid"),
    (_2a_out1, _2a_in1, "solid"), (_2a_in1, _2a_in0, "hidden"),
]
ppc_petals.append(poly_2a_final)
ppc_edges.extend(edges_2a_final)

# 1a: vertical mirror of 2a (x -> -x), point order reversed to keep a
# consistent in0->out0->tip1->tip2->out1->in1 traversal for its own span
def _vmirror(p): return np.array([-p[0], p[1]])
def _hmirror(p): return np.array([p[0], -p[1]])

poly_1a_final = np.array([_vmirror(p) for p in [_2a_in1, _2a_out1, _2a_tip2, _2a_tip1, _2a_out0, _2a_in0]])
edges_1a_final = [
    (poly_1a_final[0], poly_1a_final[1], "solid"), (poly_1a_final[1], poly_1a_final[2], "solid"),
    (poly_1a_final[2], poly_1a_final[3], "solid"), (poly_1a_final[3], poly_1a_final[4], "solid"),
    (poly_1a_final[4], poly_1a_final[5], "solid"), (poly_1a_final[5], poly_1a_final[0], "hidden"),
]
ppc_petals.append(poly_1a_final)
ppc_edges.extend(edges_1a_final)

# 4a: horizontal mirror of 2a (y -> -y), same point order
poly_4a_final = np.array([_hmirror(p) for p in poly_2a_final])
edges_4a_final = [
    (poly_4a_final[0], poly_4a_final[1], "dashed"), (poly_4a_final[1], poly_4a_final[2], "solid"),
    (poly_4a_final[2], poly_4a_final[3], "solid"), (poly_4a_final[3], poly_4a_final[4], "solid"),
    (poly_4a_final[4], poly_4a_final[5], "solid"), (poly_4a_final[5], poly_4a_final[0], "hidden"),
]
ppc_petals.append(poly_4a_final)
ppc_edges.extend(edges_4a_final)

# 3a: horizontal mirror of 1a (= both mirrors of 2a)
poly_3a_final = np.array([_hmirror(p) for p in poly_1a_final])
edges_3a_final = [
    (poly_3a_final[0], poly_3a_final[1], "solid"), (poly_3a_final[1], poly_3a_final[2], "solid"),
    (poly_3a_final[2], poly_3a_final[3], "solid"), (poly_3a_final[3], poly_3a_final[4], "solid"),
    (poly_3a_final[4], poly_3a_final[5], "solid"), (poly_3a_final[5], poly_3a_final[0], "hidden"),
]
ppc_petals.append(poly_3a_final)
ppc_edges.extend(edges_3a_final)

# ---------------------------------------------------------------------------
# 2b override: session-confirmed coordinates. in0/in1 unchanged; out1 moved
# to EXACTLY match 2a's (already-moved) out0, so the shared boundary has
# zero gap; M and V2 repositioned per the user's markup (V2 directly below
# M, vertical).
# ---------------------------------------------------------------------------
_2b_in0 = symmetric_inner(18)
_2b_in1 = symmetric_inner(54)
_2b_out1 = _2a_out0.copy()            # exact match with 2a's out0 -- zero-gap shared boundary
_2b_M = np.array([1.7, 1.36])
_2b_V2 = np.array([1.7, 0.65])
poly_2b_final = np.array([_2b_in0, _2b_V2, _2b_M, _2b_out1, _2b_in1])
edges_2b_final = [
    (_2b_in0, _2b_V2, "dashed"), (_2b_V2, _2b_M, "solid"),
    (_2b_M, _2b_out1, "solid"), (_2b_out1, _2b_in1, "dashed"),
    (_2b_in1, _2b_in0, "hidden"),
]
ppc_petals.append(poly_2b_final)
ppc_edges.extend(edges_2b_final)

# 1b: vertical mirror of 2b (x -> -x), point order reversed for a
# consistent in0->...->in1 traversal on its own span
poly_1b_final = np.array([_vmirror(p) for p in [_2b_in1, _2b_out1, _2b_M, _2b_V2, _2b_in0]])
edges_1b_final = [
    (poly_1b_final[0], poly_1b_final[1], "solid"), (poly_1b_final[1], poly_1b_final[2], "solid"),
    (poly_1b_final[2], poly_1b_final[3], "solid"), (poly_1b_final[3], poly_1b_final[4], "solid"),
    (poly_1b_final[4], poly_1b_final[0], "hidden"),
]
ppc_petals.append(poly_1b_final)
ppc_edges.extend(edges_1b_final)

# 4b: horizontal mirror of 2b (y -> -y), same point order
poly_4b_final = np.array([_hmirror(p) for p in poly_2b_final])
edges_4b_final = [
    (poly_4b_final[0], poly_4b_final[1], "hidden"), (poly_4b_final[1], poly_4b_final[2], "solid"),
    (poly_4b_final[2], poly_4b_final[3], "solid"), (poly_4b_final[3], poly_4b_final[4], "dashed"),
    (poly_4b_final[4], poly_4b_final[0], "hidden"),
]
ppc_petals.append(poly_4b_final)
ppc_edges.extend(edges_4b_final)

# 3b: horizontal mirror of 1b (= both mirrors of 2b)
poly_3b_final = np.array([_hmirror(p) for p in poly_1b_final])
edges_3b_final = [
    (poly_3b_final[0], poly_3b_final[1], "solid"), (poly_3b_final[1], poly_3b_final[2], "solid"),
    (poly_3b_final[2], poly_3b_final[3], "solid"), (poly_3b_final[3], poly_3b_final[4], "hidden"),
    (poly_3b_final[4], poly_3b_final[0], "hidden"),
]
ppc_petals.append(poly_3b_final)
ppc_edges.extend(edges_3b_final)

# ---------------------------------------------------------------------------
# 2c override: session-confirmed coordinates. in0/in1 unchanged; out1 moved
# to EXACTLY match 2b.V2 (zero-gap shared boundary); out0/tip1/tip2
# repositioned per the user's markup.
# ---------------------------------------------------------------------------
_2c_in0 = symmetric_inner(-18)
_2c_in1 = symmetric_inner(18)
_2c_out0 = np.array([1.7, -0.65])
_2c_tip1 = np.array([2.25, -0.35])
_2c_tip2 = np.array([2.25, 0.35])
_2c_out1 = _2b_V2.copy()   # exact match with 2b.V2 -- zero-gap shared boundary
poly_2c_final = np.array([_2c_in0, _2c_out0, _2c_tip1, _2c_tip2, _2c_out1, _2c_in1])
edges_2c_final = [
    (_2c_in0, _2c_out0, "dashed"), (_2c_out0, _2c_tip1, "solid"),
    (_2c_tip1, _2c_tip2, "solid"), (_2c_tip2, _2c_out1, "solid"),
    (_2c_out1, _2c_in1, "dashed"), (_2c_in1, _2c_in0, "hidden"),
]
ppc_petals.append(poly_2c_final)
ppc_edges.extend(edges_2c_final)

# 1c: vertical mirror of 2c (x -> -x), point order reversed
poly_1c_final = np.array([_vmirror(p) for p in [_2c_in1, _2c_out1, _2c_tip2, _2c_tip1, _2c_out0, _2c_in0]])
edges_1c_final = [
    (poly_1c_final[0], poly_1c_final[1], "solid"), (poly_1c_final[1], poly_1c_final[2], "solid"),
    (poly_1c_final[2], poly_1c_final[3], "solid"), (poly_1c_final[3], poly_1c_final[4], "solid"),
    (poly_1c_final[4], poly_1c_final[5], "solid"), (poly_1c_final[5], poly_1c_final[0], "hidden"),
]
ppc_petals.append(poly_1c_final)
ppc_edges.extend(edges_1c_final)


# ---------------------------------------------------------------------------
# Interommatidial cells. First one: "1L", sitting in the notch above 1a,
# traced from the user's annotated screenshot (dashed alignment lines +
# hand-drawn hexagon). Attaches to 1a along its EXISTING out0-tip1 edge
# (hidden below, already drawn); apex is shared with 2L's mirror at x=0.
# ---------------------------------------------------------------------------
L_out0  = np.array([0.00, 2.42])   # = 2a.out1 / 1a.out0 (existing, shared w/ 2L)
L_apex  = np.array([0.00, 3.65])   # new apex, shared w/ 2L
L_shoulder = np.array([-0.63, 3.45])  # new, directly above 1a.tip1
L_tip1  = np.array([-0.63, 2.60])  # = 1a.tip1 (existing)

poly_1L = np.array([L_out0, L_apex, L_shoulder, L_tip1])
edges_1L = [
    (L_out0, L_apex, "hidden"),       # shared w/ 2L at x=0 -- not a real cell boundary, don't draw
    (L_apex, L_shoulder, "solid"),    # new diagonal
    (L_shoulder, L_tip1, "solid"),    # new vertical, lands exactly on 1a.tip1
    (L_tip1, L_out0, "solid"),        # bottom edge -- the red petal used to draw this
]

# 2L: mirror of 1L (x -> -x). Lands on 2a.tip2 = (0.63, 2.60) instead of 1a.tip1,
# and its "hidden" edge is 2a's own existing tip2-out1 edge.
def _vmirror2(p): return np.array([-p[0], p[1]])
poly_2L = np.array([_vmirror2(p) for p in poly_1L])
edges_2L = [
    (poly_2L[0], poly_2L[1], "hidden"),   # same shared seam as edges_1L[0] -- must match
    (poly_2L[1], poly_2L[2], "solid"),
    (poly_2L[2], poly_2L[3], "hidden"),   # will be shared with the 2L-2Lcopy gap-filler below
    (poly_2L[3], poly_2L[0], "solid"),
]

ioc_polys = [poly_1L, poly_2L]
ioc_edges = edges_1L + edges_2L

# _hmirror2 kept: still used by later mirrored cells.
def _hmirror2(p): return np.array([p[0], -p[1]])

# Copy of 2L, translated +1.25 in x (matches the x~1.25/x~1.9 reference lines
# found earlier). Its LEFT edge (at x=1.25) is shared with the gap-filler
# below, so it's hidden now rather than solid.
poly_2L_copy = poly_2L + np.array([1.25, 0.0])
edges_2L_copy = [
    (poly_2L_copy[0], poly_2L_copy[1], "hidden"),  # shared w/ the gap-filler
    (poly_2L_copy[1], poly_2L_copy[2], "solid"),
    (poly_2L_copy[2], poly_2L_copy[3], "solid"),
    (poly_2L_copy[3], poly_2L_copy[0], "solid"),
]
ioc_polys += [poly_2L_copy]
ioc_edges += edges_2L_copy

# Gap-filler between 2L and poly_2L_copy: right edge = poly_2L_copy's own
# left edge (x=1.25, shared/hidden); left edge = 2L's own shoulder-tip edge
# at x=0.63 (shared/hidden, using 2L's exact coords so it's a true zero-gap
# fit, not a re-derived approximation). Only the top/bottom diagonals show.
poly_gap = np.array([
    poly_2L_copy[0],   # (1.25, 2.42) -- shared w/ poly_2L_copy
    poly_2L_copy[1],   # (1.25, 3.65) -- shared w/ poly_2L_copy
    poly_2L[2],        # (0.63, 3.45) -- shared w/ 2L (exact)
    poly_2L[3],        # (0.63, 2.60) -- shared w/ 2L (exact)
])
edges_gap = [
    (poly_gap[0], poly_gap[1], "hidden"),  # = poly_2L_copy's left edge
    (poly_gap[1], poly_gap[2], "solid"),   # new top diagonal
    (poly_gap[2], poly_gap[3], "hidden"),  # = 2L's shoulder-tip edge
    (poly_gap[3], poly_gap[0], "solid"),   # new bottom diagonal
]
ioc_polys += [poly_gap]
ioc_edges += edges_gap

# "3L": a 6th standalone cell further out along the 2a/2b side, traced from
# the user's own annotated screenshot (magenta lines + direct outline trace).
# CORRECTED: the top isn't a single V1->V2 edge -- it peaks through (1.88,2.60),
# which is exactly poly_2L_copy's own tip vertex (a shared corner, not a shared
# edge, so no "hidden" needed). V1/V5 still sit ~0.1 from 2a.tip1/2a.out0.
poly_3L_new = np.array([
    [1.25, 2.42],
    [1.88, 2.60],   # = poly_2L_copy's tip vertex (exact coincidence)
    [2.52, 2.40],
    [2.52, 1.54],
    [1.89, 1.36],
    [1.25, 1.56],
])
edges_3L_new = [(poly_3L_new[i], poly_3L_new[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_3L_new]
ioc_edges += edges_3L_new

# Next hexagon over (traced from the newest screenshot): shares its top edge
# with poly_3L_new's own (2.52,1.54)-(1.89,1.36) edge -- that edge is hidden
# on both sides now. New points traced from the magenta outline.
poly_4th = np.array([
    [2.52, 1.54],   # = poly_3L_new[3] (shared corner)
    [1.89, 1.36],   # = poly_3L_new[4] (shared corner)
    [1.89, 0.49],
    [2.53, 0.29],
    [3.17, 0.49],
    [3.17, 1.36],
])
edges_4th = [(poly_4th[i], poly_4th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_4th]
ioc_edges += edges_4th

# New standalone hexagon near 2c.tip1, re-traced from a clean blue-outline
# screenshot (long, unambiguous parallel verticals at x=2.56/4.01). Its own
# L-top point measured 0.03 from poly_4th's tip (2.53,0.29) -- snapped to
# that exact value for a clean zero-gap corner connection.
poly_6th = np.array([
    [3.17, 0.49],   # snapped to poly_4th's exact vertex (was 3.18,0.48)
    [3.83, 0.29],
    [3.83, -0.56],
    [3.18, -0.76],
    [2.53, -0.56],
    [2.53, 0.29],
])
edges_6th = [(poly_6th[i], poly_6th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_6th]
ioc_edges += edges_6th

# Next cell down, traced from the blue overlay. Its top-right vertex is
# poly_6th's own bottom-left corner (2.53,-0.56), and its top edge runs up to
# poly_6th's bottom tip (3.18,-0.76) -- so it shares that whole edge.
# Verticals land exactly on the magenta lines at x=1.88 and x=3.18.
# Lower chain: congruent copies of poly_6th, each stepped by the same vector
# (6th's top tip -> its lower-left vertex) so every hexagon's top-right edge
# lands exactly on the previous one's lower-left edge -- no gap.
_step = poly_6th[4] - poly_6th[0]

poly_7th = poly_6th + _step
poly_7th[1] = poly_6th[3]          # snap junction exactly
poly_7th[2][0] = poly_7th[1][0]    # right edge exactly vertical
edges_7th = [(poly_7th[i], poly_7th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_7th]
ioc_edges += edges_7th

poly_8th = poly_7th + _step
poly_8th[0] = poly_7th[4]
poly_8th[1] = poly_7th[3]
edges_8th = [(poly_8th[i], poly_8th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_8th]
ioc_edges += edges_8th

poly_9th = np.array([
    [1.249, -2.644],
    [1.936, -2.870],
    [1.936, -3.708],
    [1.287, -3.891],
    [0.649, -3.708],
    [0.649, -2.870],
])
# keep the 8th/9th junction sealed after snapping 9th to the magenta lattice
poly_8th[4] = poly_9th[0]
poly_8th[3] = poly_9th[1]
poly_8th[5][0] = poly_8th[4][0]   # left edge exactly vertical
poly_8th[2][0] = poly_8th[1][0]   # right edge exactly vertical
edges_8th = [(poly_8th[i], poly_8th[(i + 1) % 6], "solid") for i in range(6)]
edges_9th = [(poly_9th[i], poly_9th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_9th]
ioc_edges += edges_9th

# 10th: one hexagon-width to the left on the same magenta lattice.
poly_10th = np.array([
    [0.000, -2.656],
    [0.649, -2.870],
    [0.649, -3.708],
    [0.000, -3.891],
    [-0.649, -3.708],
    [-0.649, -2.870],
])
edges_10th = [(poly_10th[i], poly_10th[(i + 1) % 6], "solid") for i in range(6)]
ioc_polys += [poly_10th]
ioc_edges += edges_10th

# ---------------------------------------------------------------------------
# Left side: exact mirror (x -> -x) of the right-hand chain. 1L/2L are already
# a symmetric pair about x=0, and the 10th straddles the axis, so mirroring
# those three would just duplicate what's there -- they're skipped.
# Each edge keeps its original solid/hidden style so seams behave the same.
# ---------------------------------------------------------------------------
_right_cells = [
    (poly_2L_copy, edges_2L_copy), (poly_gap, edges_gap),
    (poly_3L_new, edges_3L_new), (poly_4th, edges_4th),
    (poly_6th, edges_6th), (poly_7th, edges_7th),
    (poly_8th, edges_8th), (poly_9th, edges_9th),
]
for _poly, _edges in _right_cells:
    _m = np.array([[-p[0], p[1]] for p in _poly])
    _n = len(_m)
    _me = [(_m[i], _m[(i + 1) % _n], _edges[i][2]) for i in range(_n)]
    ioc_polys += [_m]
    ioc_edges += _me

# ---------------------------------------------------------------------------
# Vertex registry: every named point used above, keyed by rounded coordinate
# so points shared between neighboring shapes (e.g. 2a.in1 == 1a.in0) collapse
# into one dot with a combined label instead of drawing/labeling it twice.
# ---------------------------------------------------------------------------
vertex_registry = {}

def _register(point, label):
    key = (round(float(point[0]), 3), round(float(point[1]), 3))
    vertex_registry.setdefault(key, []).append(label)

for _name, _pt in [
    ("TOP", TOP), ("BOTTOM", BOTTOM), ("A", A), ("B", B),
    ("UL_shoulder", UL_shoulder), ("UR_shoulder", UR_shoulder),
    ("LL_shoulder", LL_shoulder), ("LR_shoulder", LR_shoulder),
    ("V1", V1), ("V2", V2), ("V1_L", V1_L), ("V2_L", V2_L),
]:
    _register(_pt, _name)

_petal_order = ["2a", "1a", "4a", "3a", "2b", "1b", "4b", "3b", "2c", "1c"]
_petal_roles = {
    "2a": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
    "1a": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
    "4a": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
    "3a": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
    "2b": ["in0", "V2", "M", "out1", "in1"],
    "1b": ["in0", "V2", "M", "out1", "in1"],
    "4b": ["in0", "V2", "M", "out1", "in1"],
    "3b": ["in0", "V2", "M", "out1", "in1"],
    "2c": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
    "1c": ["in0", "out0", "tip1", "tip2", "out1", "in1"],
}
for _pid, _poly in zip(_petal_order, ppc_petals):
    for _role, _pt in zip(_petal_roles[_pid], _poly):
        _register(_pt, f"{_pid}.{_role}")



# ---------------------------------------------------------------------------

# ===========================================================================
# TILED SHEET (manual placement). The hand-built 19-cell ring is stamped at
# every ommatidium; cells that land on each other are merged, and vertices that
# land near each other are snapped together so neighbouring rings become
# genuinely congruent instead of leaving hairline overlaps and gaps.
# ===========================================================================
OMM_A = np.array([4.30,  3.16])
OMM_B = np.array([4.30, -3.16])
N_OMM = 2
omm_centres = [i * OMM_A + j * OMM_B
               for i in range(-N_OMM, N_OMM + 1)
               for j in range(-N_OMM, N_OMM + 1)]

CELL_TOL = 0.45      # cells whose centres are this close are the same cell
VERT_TOL = 0.30      # vertices this close are welded together

# --- stamp, then merge duplicate cells --------------------------------------
raw = []
for o in omm_centres:
    for p in ioc_polys:
        raw.append(np.asarray(p, float) + o)

merged, m_cent = [], []
for p in raw:
    c = p.mean(axis=0)
    hit = None
    for k, mc in enumerate(m_cent):
        if np.linalg.norm(mc - c) < CELL_TOL:
            hit = k
            break
    if hit is None:
        m_cent.append(c)
        merged.append(p)

# --- weld nearby vertices so shared edges coincide exactly ------------------
# The vertices on each ring's inner (purple) boundary are pinned first: those
# are the ones the dashed connectors have to meet, so they must not drift.
_ioc_paths_pin = [Path(np.asarray(p, float)) for p in ioc_polys]
_pinned_local = []
for _p1, _p2, _st in ioc_edges:
    if _st == "hidden":
        continue
    _p1 = np.asarray(_p1, float); _p2 = np.asarray(_p2, float)
    _d = _p2 - _p1
    _L = np.linalg.norm(_d)
    if _L == 0:
        continue
    _nr = np.array([-_d[1], _d[0]]) / _L
    _mid = (_p1 + _p2) / 2.0
    _ins = [any(pa.contains_point(_mid + ss * 0.03 * _nr) for pa in _ioc_paths_pin)
            for ss in (1.0, -1.0)]
    if all(_ins) or not any(_ins):
        continue
    _ed = _nr if not _ins[0] else -_nr
    if float(np.dot(_ed, -_mid)) > 0:
        _pinned_local += [_p1, _p2]

anchors = []
for o in omm_centres:
    for v in _pinned_local:
        w = np.asarray(v, float) + o
        if not any(np.linalg.norm(a - w) < VERT_TOL for a in anchors):
            anchors.append(w)

def _weld(v):
    for a in anchors:
        if np.linalg.norm(a - v) < VERT_TOL:
            return a
    anchors.append(v.copy())
    return anchors[-1]

sheet_polys = [np.array([_weld(v) for v in p]) for p in merged]

# --- absorb sliver cells ----------------------------------------------------
# Where two rings meet, the little 4-vertex notch cells can get welded into a
# very narrow strip that is not a real cell. Drop each sliver and push the
# neighbour that shares its long edge across to the sliver's far side, so the
# space is filled by one properly sized cell instead of two bad ones.
def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

_SLIVER_AREA = 0.45
_slivers = [i for i, p in enumerate(sheet_polys) if _poly_area(p) < _SLIVER_AREA]

# Absorb each sliver by pulling the neighbour's shared edge across to the
# sliver's far side -- but always toward the nearest ommatidium centre, never
# away from it, so the pocket around a cluster keeps its correct width and the
# dashed connectors still land on the purple boundary.
for i in _slivers:
    sl = sheet_polys[i]
    xs = np.unique(np.round(sl[:, 0], 3))
    if len(xs) != 2:
        continue
    o = min(omm_centres, key=lambda o: np.linalg.norm(sl.mean(axis=0) - o))
    # the edge nearer the ommatidium centre is the one that must stay put
    x_keep = xs[int(np.argmin([abs(xs[0] - o[0]), abs(xs[1] - o[0])]))]
    x_drop = xs[1] if x_keep == xs[0] else xs[0]
    for j, p in enumerate(sheet_polys):
        if j == i or j in _slivers:
            continue
        share = np.isclose(p[:, 0], x_keep, atol=0.005)
        if share.sum() >= 2:
            p[share, 0] = x_drop

sheet_polys = [p for k, p in enumerate(sheet_polys) if k not in set(_slivers)]

fig, ax = plt.subplots(figsize=(11, 11))
ax.set_aspect("equal")
ax.axis("off")

for poly in sheet_polys:
    ax.add_patch(MplPolygon(poly, closed=True, facecolor="#E6E6E6",
                             edgecolor="none", zorder=1))

_paths = [Path(p) for p in sheet_polys]
def _col(p1, p2):
    mid = (p1 + p2) / 2.0
    d = p2 - p1
    n = np.linalg.norm(d)
    if n == 0:
        return "#888888"
    nrm = np.array([-d[1], d[0]]) / n
    o = min(omm_centres, key=lambda o: np.linalg.norm(mid - o))
    for s in (1.0, -1.0):
        q = mid + s * 0.05 * nrm
        if not any(pa.contains_point(q) for pa in _paths):
            if np.linalg.norm(q - o) < np.linalg.norm(mid - o):
                return "#8E2C8E"
    return "#888888"

# an edge is a "quad wall" when one of the two cells meeting along it is one of
# the small four-sided notch cells rather than a full hexagon
_quad_edges = set()
for _p in sheet_polys:
    if len(_p) != 4:
        continue
    for _k in range(4):
        _a, _b = _p[_k], _p[(_k + 1) % 4]
        _quad_edges.add(tuple(sorted([tuple(np.round(_a, 2)), tuple(np.round(_b, 2))])))

def _quad_wall(p1, p2):
    return tuple(sorted([tuple(np.round(p1, 2)), tuple(np.round(p2, 2))])) in _quad_edges

_IOC_OUTER_COLOR = "#B8860B"   # dark goldenrod -- replaces the old neutral grey (#888888)
                                # for IOC-IOC walls that don't face a cluster pocket

_seen = set()
for poly in sheet_polys:
    n = len(poly)
    for k in range(n):
        p1, p2 = poly[k], poly[(k + 1) % n]
        key = tuple(sorted([tuple(np.round(p1, 2)), tuple(np.round(p2, 2))]))
        if key in _seen:
            continue
        _seen.add(key)
        _c = _col(p1, p2)
        _is_vert = abs(p2[0] - p1[0]) < 0.03
        if _c == "#888888" and _is_vert and _quad_wall(p1, p2):
            # Most notch-cell walls stay visible now. Only the two the user
            # flagged -- the one sitting right on a cluster's own vertical axis,
            # and the one exactly 1.25 further along that same chain -- are
            # genuinely mid-hexagon splits, so only those two stay hidden.
            # (Checked against every ommatidium centre, not just the nearest
            # one, since right at the seam between two clusters "nearest"
            # flips and a plain nearest-centre check misses the second line.)
            _mid = (p1 + p2) / 2.0
            _hide = False
            for _o in omm_centres:
                _xl = _mid[0] - _o[0]
                _yl = _mid[1] - _o[1]
                if (abs(_xl) < 0.06 or abs(abs(_xl) - 1.25) < 0.08) and (2.3 < abs(_yl) < 3.95):
                    _hide = True
                    break
            if _hide:
                continue
        _draw_c = _IOC_OUTER_COLOR if _c == "#888888" else _c
        _x1, _y1, _x2, _y2 = p1[0], p1[1], p2[0], p2[1]
        # Edges meant to be vertical can land a hair off-plumb after welding
        # (sub-0.02-unit x mismatch) -- snap those two x's together so the
        # line draws perfectly straight, without moving the underlying cells.
        if 0 < abs(_x2 - _x1) < 0.05:
            _x1 = _x2 = (_x1 + _x2) / 2.0
        ax.plot([_x1, _x2], [_y1, _y2], color=_draw_c, lw=1.4, zorder=1.5)

# --- clusters and connectors -------------------------------------------------
_CC_DROP = np.array([-0.0176, -0.1004])
_cc = [np.asarray(p, float) + _CC_DROP for p in cc_polys]
_seams = [(np.asarray(a, float) + _CC_DROP, np.asarray(b, float) + _CC_DROP, c)
          for a, b, c in SEAM_EDGES]
_mag = [(np.asarray(TOP, float) + _CC_DROP, np.array([0.0, 2.420])),
        (np.asarray(BOTTOM, float) + _CC_DROP, np.array([0.0, -2.656]))]
_dash = [
    (np.array([0.591, 0.813]),  np.array([1.250, 1.560])),
    (np.array([1.312, 0.426]),  np.array([1.890, 0.490])),
    (np.array([1.312, -0.426]), np.array([1.890, -0.760])),
    (np.array([0.591, -0.813]), np.array([1.249, -1.810])),
    (np.array([-0.591, 0.813]),  np.array([-1.250, 1.560])),
    (np.array([-1.312, 0.426]),  np.array([-1.890, 0.490])),
    (np.array([-1.312, -0.426]), np.array([-1.890, -0.760])),
    (np.array([-0.591, -0.813]), np.array([-1.249, -1.810])),
]
for o in omm_centres:
    for a, b in _mag:
        p, q = a + o, b + o
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#E040E0", lw=1.8, zorder=3)
    for a, b in _dash:
        a2 = a + _CC_DROP + o
        b2 = b + o
        dv = (a2 - b2) / np.linalg.norm(a2 - b2)
        ax.plot([b2[0], (a2 + dv * 0.12)[0]], [b2[1], (a2 + dv * 0.12)[1]],
                color="#E8452C", lw=1.8, zorder=3, linestyle="--", dashes=(4, 3))
    # cc_polys order is [top, bottom, left wing, right wing]: top/bottom pentagons
    # (the cone cell) get a cyan outer border; the two wings (primary pigment
    # cells) keep the original blue outer border.
    for poly, _cc_ec in zip(_cc, ["#54B9D1", "#54B9D1", "#0056B6", "#0056B6"]):
        ax.add_patch(MplPolygon(poly + o, closed=True, facecolor="#C4DEF8",
                                 edgecolor=_cc_ec, linewidth=2.0, zorder=4))
    for a, b, col in _seams:
        p, q = a + o, b + o
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=2.6, zorder=5,
                solid_capstyle="round")

_lim = 10.0
ax.set_xlim(-_lim, _lim); ax.set_ylim(-_lim, _lim)
plt.tight_layout()
import time
_ts = int(time.time())
from pathlib import Path

output_dir = Path(__file__).resolve().parent
output_file = output_dir / f"tiled_sheet_v4_{_ts}.png"

plt.savefig(output_file, dpi=170,
            bbox_inches="tight", facecolor="white")

print(f"Saved image to: {output_file}")
print(f"saved tiled_sheet_v4_{_ts}.png  ommatidia:", len(omm_centres),
      " cells after merge:", len(sheet_polys), " welded vertices:", len(anchors))

