#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NSGA-II (Batch-level encoding) with:
- Scheme A: Lateness and waiting are incorporated into objectives (NOT penalty).
  * Cost = transport_cost + waiting_cost + lateness_cost
  * Emission = transport_emission + waiting_emission
  * Makespan = max completion time
- Penalty is reserved for HARD constraints only:
  * missing allocation (miss_alloc)
  * missing timetable (miss_tt)
  * capacity excess (cap_excess)
- Constraint-domination:
  * feasible dominates infeasible
  * feasible compares by Pareto on objectives
  * infeasible compares by penalty first then Pareto
- Multi-runs analytics:
  * HV, IGD+ (with pseudo-reference P*), Spacing, feasible ratio
  * mutation roulette plots
  * Pareto 3D plots (final + all generations for best run)
  * runtime summary Excel
"""

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========================
# Global settings
# ========================

TIME_BUCKET_H = 1.0  # simulation granularity: 1 hour
CHINA_REGIONS = {"CN", "China"}
EUROPE_REGIONS = {"WE", "EE", "EU", "Europe"}

# ---- Time window handling:
# HARD_TIME_WINDOW=False means lateness is allowed but penalised in COST (Scheme A).
# HARD_TIME_WINDOW=True means lateness is forbidden (treated as infeasible).
HARD_TIME_WINDOW = False

# ---- HARD constraint penalty weights
PEN_MISS_TT = 5e7
PEN_MISS_ALLOC = 1e9
PEN_CAP_EXCESS_PER_TEU = 5e7

# ---- Scheme A: lateness and waiting are in objectives (cost/emission)
# Waiting: cost per TEU per hour; emission in gCO2 per TEU per hour
WAITING_COST_PER_TEU_HOUR_DEFAULT = 0.5
WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT = 0.0

# Lateness: cost per TEU per hour late (Scheme A)
# (Tune this so it is comparable to transport cost scale; start large to enforce LT.)
LATE_COST_PER_TEU_HOUR = 1e6

# ---- Static roulette priors
W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20
OPS = ["add", "del", "mod", "mode"]

# ---- Path library diversity (ONLY change you asked for)
PATHS_TOPK_PER_CRITERION = 30   # take top 30 by cost/time/emission each
PATH_LIB_CAP_TOTAL = 90         # union cap (30*3=90). keep as 90.
DFS_MAX_PATHS_PER_OD = 1200     # increase DFS sampling pool to make top-k meaningful


# ========================
# Helpers
# ========================

def normalize_mode(mode_raw: str) -> str:
    m = str(mode_raw).strip().lower()
    if m in {"railway", "rail"}:
        return "rail"
    if m in {"road", "truck"}:
        return "road"
    if m in {"water", "ship", "sea"}:
        return "water"
    return m


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def parse_distance_km(x) -> float:
    s = str(x)
    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    return float(cleaned) if cleaned else 0.0


def unique_objective_tuples(
    objs: List[Tuple[float, float, float]],
    tol: float = 1e-9
) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for o in objs:
        dup = False
        for p in out:
            if all(abs(o[i] - p[i]) <= tol for i in range(3)):
                dup = True
                break
        if not dup:
            out.append(o)
    return out


# ========================
# Data structures
# ========================

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float       # TEU / bucket (e.g., TEU/hour)
    cost_per_teu_km: float
    emission_per_teu_km: float  # gCO2 per TEU-km
    speed_kmh: float


@dataclass
class TimetableEntry:
    from_node: str
    to_node: str
    mode: str
    frequency_per_week: float
    first_departure_hour: float
    headway_hours: float


@dataclass
class Batch:
    batch_id: int
    origin: str
    destination: str
    quantity: float  # TEU
    ET: float
    LT: float


@dataclass
class Path:
    path_id: int
    origin: str
    destination: str
    nodes: List[str]
    modes: List[str]
    arcs: List[Arc]
    base_cost_per_teu: float
    base_emission_per_teu: float  # gCO2 per TEU
    base_travel_time_h: float

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return self.nodes == other.nodes and self.modes == other.modes

    def __hash__(self):
        return hash((tuple(self.nodes), tuple(self.modes)))


@dataclass
class PathAllocation:
    path: Path
    share: float

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            mode = self.path.modes[i]
            chain += f"{node}--({mode})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    # key=(origin, dest, batch_id)
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)

    # objectives: (cost, emission, makespan)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

    # HARD-constraint handling
    penalty: float = 0.0
    feasible: bool = False          # hard-feasible (and lateness rule if HARD_TIME_WINDOW=True)
    feasible_hard: bool = False     # strict: also no lateness (always tracked)

    # diagnostics
    vio_breakdown: Dict[str, float] = field(default_factory=dict)


# ========================
# Merge & normalise shares
# ========================

def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []

    merged: Dict[Path, float] = {}
    for a in allocs:
        merged[a.path] = merged.get(a.path, 0.0) + float(a.share)

    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged.items()]
    total = sum(a.share for a in unique_allocs)

    if total <= 1e-12:
        avg = 1.0 / max(1, len(unique_allocs))
        for a in unique_allocs:
            a.share = avg
    else:
        for a in unique_allocs:
            a.share /= total

    # drop tiny shares
    unique_allocs = [a for a in unique_allocs if a.share > 0.001]

    if unique_allocs:
        total2 = sum(a.share for a in unique_allocs)
        if abs(total2 - 1.0) > 1e-9:
            for a in unique_allocs:
                a.share /= total2

    return unique_allocs


# ========================
# Load data
# ========================

def load_waiting_params(xls: pd.ExcelFile) -> Tuple[float, float]:
    """
    Optional sheet: 'Waiting_Costs'
    Accept columns:
      - WaitingCost_per_TEU_h  or  WaitCost_per_TEU_h  (fallback to default)
      - WaitEmission_gCO2_per_TEU_h  or  WaitingEmission_gCO2_per_TEU_h
    """
    wc = WAITING_COST_PER_TEU_HOUR_DEFAULT
    we = WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT

    if "Waiting_Costs" not in xls.sheet_names:
        print(f"[INFO] Waiting_Costs sheet not found. Using defaults: cost={wc}, emission={we}")
        return wc, we

    try:
        df = pd.read_excel(xls, "Waiting_Costs")
        # take first non-null
        def pick(colnames, default):
            for c in colnames:
                if c in df.columns:
                    vals = df[c].dropna().tolist()
                    if vals:
                        return safe_float(vals[0], default=default)
            return default

        wc = pick(["WaitingCost_per_TEU_h", "WaitCost_per_TEU_h"], wc)
        we = pick(["WaitEmission_gCO2_per_TEU_h", "WaitingEmission_gCO2_per_TEU_h"], we)
        print(f"[INFO] Loaded waiting params: cost={wc}, emission={we} (gCO2/TEU/h)")
        return wc, we
    except Exception as e:
        print(f"[WARN] Failed to read Waiting_Costs ({e}). Using defaults: cost={wc}, emission={we}")
        return wc, we


def load_network_from_extended(filename: str):
    xls = pd.ExcelFile(filename)

    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).str.strip().tolist()
    node_region = dict(
        zip(nodes_df["EnglishName"].astype(str).str.strip(),
            nodes_df["Region"].astype(str).str.strip())
    )

    # optional waiting params
    waiting_cost_per_teu_h, wait_emis_g_per_teu_h = load_waiting_params(xls)

    arcs_df = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []

    DAILY_HOURS = 24.0  # assumes Capacity_TEU is TEU/day
    for _, row in arcs_df.iterrows():
        mode = normalize_mode(row.get("Mode", "road"))

        if mode == "road":
            speed = 75.0
        elif mode == "water":
            speed = 30.0
        else:
            speed = 50.0  # rail or others

        distance = parse_distance_km(row.get("Distance_km", 0.0))

        if "Capacity_TEU" in arcs_df.columns and not pd.isna(row.get("Capacity_TEU", np.nan)):
            raw_cap = safe_float(row.get("Capacity_TEU"), default=1e9)
        else:
            raw_cap = 1e9

        capacity = raw_cap * (TIME_BUCKET_H / DAILY_HOURS)

        arcs.append(Arc(
            from_node=str(row.get("OriginEN", "")).strip(),
            to_node=str(row.get("DestEN", "")).strip(),
            mode=mode,
            distance=distance,
            capacity=capacity,
            cost_per_teu_km=safe_float(row.get("Cost_$_per_km"), default=0.0),
            emission_per_teu_km=safe_float(row.get("Emission_gCO2_per_tkm"), default=0.0),
            speed_kmh=speed
        ))

    # timetable
    tdf = pd.read_excel(xls, "Timetable")
    timetables: List[TimetableEntry] = []
    for _, row in tdf.iterrows():
        freq = safe_float(row.get("Frequency_per_week"), default=1.0)
        hd = row.get("Headway_Hours", np.nan)
        hd = 168.0 / max(freq, 1.0) if pd.isna(hd) else safe_float(hd, default=168.0)

        v = row.get("FirstDepartureHour", np.nan)
        fd = 0.0
        if not pd.isna(v):
            try:
                s = str(v).strip()
                fd = float(s.split(":")[0]) if ":" in s else float(s)
            except Exception:
                fd = 0.0

        timetables.append(TimetableEntry(
            from_node=str(row.get("OriginEN", "")).strip(),
            to_node=str(row.get("DestEN", "")).strip(),
            mode=normalize_mode(row.get("Mode", "")),
            frequency_per_week=freq,
            first_departure_hour=fd,
            headway_hours=hd
        ))

    # batches
    bdf = pd.read_excel(xls, "Batches")
    batches: List[Batch] = []
    for _, row in bdf.iterrows():
        origin = str(row.get("OriginEN", "")).strip()
        dest = str(row.get("DestEN", "")).strip()
        o_reg = node_region.get(origin)
        d_reg = node_region.get(dest)

        if o_reg in CHINA_REGIONS and d_reg in EUROPE_REGIONS:
            batches.append(Batch(
                batch_id=int(row.get("BatchID", 0)),
                origin=origin,
                destination=dest,
                quantity=safe_float(row.get("QuantityTEU"), default=0.0),
                ET=safe_float(row.get("ET"), default=0.0),
                LT=safe_float(row.get("LT"), default=0.0)
            ))

    print(f"[INFO] Number of batches loaded: {len(batches)}")
    return node_names, arcs, timetables, batches, waiting_cost_per_teu_h, wait_emis_g_per_teu_h


def build_graph(arcs: List[Arc]) -> Dict[str, List[Tuple[str, Arc]]]:
    g: Dict[str, List[Tuple[str, Arc]]] = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append((a.to_node, a))
    return g


def build_timetable_dict(timetables: List[TimetableEntry]) -> Dict[Tuple[str, str, str], List[TimetableEntry]]:
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]] = {}
    for t in timetables:
        tt_dict.setdefault((t.from_node, t.to_node, t.mode), []).append(t)
    return tt_dict


def build_arc_lookup(arcs: List[Arc]) -> Dict[Tuple[str, str, str], Arc]:
    mp: Dict[Tuple[str, str, str], Arc] = {}
    for a in arcs:
        k = (a.from_node, a.to_node, a.mode)
        if k not in mp:
            mp[k] = a
    return mp


# ========================
# Path library
# ========================

def random_dfs_paths(graph, origin, dest, max_len=12, max_paths=200) -> List[List[Arc]]:
    paths: List[List[Arc]] = []

    def dfs(node, cur_arcs, visited):
        if len(paths) >= max_paths or len(cur_arcs) > max_len:
            return
        if node == dest and cur_arcs:
            paths.append(cur_arcs.copy())
            return

        neighbors = graph.get(node, [])
        random.shuffle(neighbors)
        for nxt, arc in neighbors:
            if nxt in visited:
                continue
            dfs(nxt, cur_arcs + [arc], visited | {nxt})

    dfs(origin, [], {origin})
    return paths


def repair_arc_seq_with_road_fallback(
    arc_seq: List[Arc],
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]],
    arc_lookup: Dict[Tuple[str, str, str], Arc]
) -> Optional[List[Arc]]:
    new_seq: List[Arc] = []
    for arc in arc_seq:
        if arc.mode == "road":
            new_seq.append(arc)
            continue

        if tt_dict.get((arc.from_node, arc.to_node, arc.mode), []):
            new_seq.append(arc)
            continue

        # no timetable -> try road fallback on same (u,v)
        k_road = (arc.from_node, arc.to_node, "road")
        if k_road in arc_lookup:
            new_seq.append(arc_lookup[k_road])
        else:
            return None

    return new_seq


def select_topk_by_cost_time_emis(paths: List[Path], k: int = 30, cap_total: int = 90) -> List[Path]:
    """
    YOUR REQUEST:
    take Top-k by each criterion (cost, time, emission), then union and dedupe.
    - Maximum size is cap_total (default 90=30*3).
    """
    if not paths:
        return []

    by_cost = sorted(paths, key=lambda p: p.base_cost_per_teu)
    by_time = sorted(paths, key=lambda p: p.base_travel_time_h)
    by_emis = sorted(paths, key=lambda p: p.base_emission_per_teu)

    picked: List[Path] = []
    used = set()

    def add_list(lst, kk):
        nonlocal picked, used
        for p in lst[:kk]:
            if p not in used:
                picked.append(p)
                used.add(p)

    add_list(by_cost, k)
    add_list(by_time, k)
    add_list(by_emis, k)

    # If too many, trim with a stable rule (keep diversity):
    # Here: keep the earliest appearance order (cost->time->emis).
    if cap_total is not None and len(picked) > cap_total:
        picked = picked[:cap_total]

    return picked


def build_path_library(node_names, arcs, batches, tt_dict, arc_lookup) -> Dict[Tuple[str, str], List[Path]]:
    graph = build_graph(arcs)
    path_lib: Dict[Tuple[str, str], List[Path]] = {}
    next_path_id = 0

    for b in batches:
        od = (b.origin, b.destination)
        if od in path_lib:
            continue

        arc_paths = random_dfs_paths(
            graph, b.origin, b.destination,
            max_len=12,
            max_paths=DFS_MAX_PATHS_PER_OD
        )
        paths_for_od: List[Path] = []

        for arc_seq in arc_paths:
            repaired = repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup)
            if repaired is None:
                continue

            nodes = [repaired[0].from_node] + [a.to_node for a in repaired]
            if len(set(nodes)) != len(nodes):
                continue  # avoid loops
            modes = [a.mode for a in repaired]

            paths_for_od.append(Path(
                path_id=next_path_id,
                origin=b.origin,
                destination=b.destination,
                nodes=nodes,
                modes=modes,
                arcs=repaired,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in repaired),
                base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in repaired),
                base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in repaired),
            ))
            next_path_id += 1

        if paths_for_od:
            # <<< CHANGED HERE: instead of cost-top30, use union of (cost/time/emis)-top30 each >>>
            path_lib[od] = select_topk_by_cost_time_emis(
                paths_for_od,
                k=PATHS_TOPK_PER_CRITERION,
                cap_total=PATH_LIB_CAP_TOTAL
            )

    return path_lib


def sanity_check_path_lib(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]):
    missing = []
    for b in batches:
        od = (b.origin, b.destination)
        if len(path_lib.get(od, [])) == 0:
            missing.append((b.batch_id, od))
    if missing:
        for bid, od in missing[:20]:
            print(f"[SANITY CHECK] ❌ missing paths for Batch {bid} OD={od}")
        raise RuntimeError("Path library missing some ODs. Infeasible forever (miss_alloc>0).")
    print("[SANITY CHECK] ✅ All batches have at least one usable path in path_lib.")


def repair_missing_allocations(ind: Individual, batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]):
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        if ind.od_allocations.get(key, []):
            continue
        paths = path_lib.get((b.origin, b.destination), [])
        if paths:
            ind.od_allocations[key] = [PathAllocation(path=paths[0], share=1.0)]


# ========================
# Simulation & evaluation
# ========================

def next_departure_time(t: float, entries: List[TimetableEntry]) -> float:
    best = float("inf")
    for e in entries:
        if t <= e.first_departure_hour:
            dep = e.first_departure_hour
        else:
            waited = (t - e.first_departure_hour)
            n = math.ceil(waited / max(e.headway_hours, 1e-6))
            dep = e.first_departure_hour + n * e.headway_hours
        best = min(best, dep)
    return best


def simulate_path_time_capacity(
    path: Path,
    batch: Batch,
    flow_teu: float,
    tt_dict,
    arc_flow_map,
) -> Tuple[float, float, int]:
    """
    Returns:
      travel_time_h, total_wait_h, miss_tt_count
    If missing timetable on non-road -> returns (inf, inf, miss_tt)
    """
    t = batch.ET
    total_wait = 0.0
    miss_tt = 0

    for arc in path.arcs:
        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

        if arc.mode == "road":
            entries = []
        else:
            entries = tt_dict.get((arc.from_node, arc.to_node, arc.mode), [])

        if (arc.mode != "road") and (not entries):
            miss_tt += 1
            return float("inf"), float("inf"), miss_tt

        dep = t if not entries else next_departure_time(t, entries)

        wait_here = max(0.0, dep - t)
        total_wait += wait_here

        arr = dep + travel_time

        # capacity occupancy: departure slot bucket (simplified)
        start_slot = int(dep)
        arc_key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (arc_key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0.0) + flow_teu

        t = arr

    return t - batch.ET, total_wait, miss_tt


def evaluate_individual(
    ind: Individual,
    batches: List[Batch],
    arcs: List[Arc],
    tt_dict,
    waiting_cost_per_teu_h: float,
    wait_emis_g_per_teu_h: float
):
    """
    Scheme A objectives:
      cost = transport_cost + waiting_cost + lateness_cost
      emission = transport_emission + waiting_emission
      makespan = max completion time

    Penalty (hard constraints):
      miss_alloc, miss_tt, cap_excess
      (lateness is not in penalty unless HARD_TIME_WINDOW=True for feasibility flag)
    """
    total_cost = 0.0
    total_emission_g = 0.0
    makespan = 0.0

    arc_flow_map: Dict[Tuple[Tuple[str, str, str], int], float] = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    miss_alloc = 0
    miss_tt = 0
    cap_excess = 0.0

    late_h_total = 0.0        # hours (diagnostic, share-weighted)
    late_teu_h_total = 0.0    # TEU*hour (for Scheme A cost)

    wait_h_total = 0.0        # hours (diagnostic, share-weighted)
    wait_teu_h_total = 0.0    # TEU*hour (for cost/emission)

    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            miss_alloc += 1
            continue

        batch_finish_time = b.ET

        for alloc in allocs:
            if alloc.share <= 1e-12:
                continue

            flow = alloc.share * b.quantity  # TEU
            p = alloc.path

            # transport base
            total_cost += p.base_cost_per_teu * flow
            total_emission_g += p.base_emission_per_teu * flow  # gCO2

            travel_time, wait_h, mtt = simulate_path_time_capacity(p, b, flow, tt_dict, arc_flow_map)

            if math.isinf(travel_time):
                miss_tt += mtt
                continue

            # waiting adds to objectives
            wait_teu_h = flow * wait_h
            wait_teu_h_total += wait_teu_h
            wait_h_total += alloc.share * wait_h

            total_cost += waiting_cost_per_teu_h * wait_teu_h
            total_emission_g += wait_emis_g_per_teu_h * wait_teu_h

            arrival_time = b.ET + travel_time
            batch_finish_time = max(batch_finish_time, arrival_time)

            # lateness: Scheme A => add lateness cost into cost objective
            if arrival_time > b.LT:
                late_h = arrival_time - b.LT
                late_h_total += alloc.share * late_h
                late_teu_h = flow * late_h
                late_teu_h_total += late_teu_h
                total_cost += LATE_COST_PER_TEU_HOUR * late_teu_h

        makespan = max(makespan, batch_finish_time)

    # capacity check
    for (arc_key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(arc_key, 1e9)
        if flow > cap:
            cap_excess += (flow - cap)

    penalty = (
        PEN_MISS_ALLOC * float(miss_alloc) +
        PEN_MISS_TT * float(miss_tt) +
        PEN_CAP_EXCESS_PER_TEU * float(cap_excess)
    )

    ind.objectives = (float(total_cost), float(total_emission_g), float(makespan))
    ind.penalty = float(penalty)

    # feasibility definition (hard constraints + optional hard lateness)
    hard_ok = (miss_alloc == 0 and miss_tt == 0 and cap_excess <= 1e-9)
    strict_no_late = (late_h_total <= 1e-9)  # diagnostic strict
    ind.feasible_hard = bool(hard_ok and strict_no_late)
    ind.feasible = bool(hard_ok and (strict_no_late if HARD_TIME_WINDOW else True))

    ind.vio_breakdown = {
        "miss_alloc": float(miss_alloc),
        "miss_tt": float(miss_tt),
        "cap_excess": float(cap_excess),
        "late_h": float(late_h_total),
        "late_teu_h": float(late_teu_h_total),
        "wait_h": float(wait_h_total),
        "wait_teu_h": float(wait_teu_h_total),
    }


# ========================
# GA operators
# ========================

def clone_gene(alloc: PathAllocation) -> PathAllocation:
    return PathAllocation(path=alloc.path, share=float(alloc.share))


def crossover_structural(ind1: Individual, ind2: Individual, batches: List[Batch]) -> Tuple[Individual, Individual]:
    child1 = Individual()
    child2 = Individual()

    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1 = ind1.od_allocations.get(key, [])
        g2 = ind2.od_allocations.get(key, [])

        if not g1 and not g2:
            continue

        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]
            continue

        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]
            continue

        cut1 = random.randint(0, len(g1))
        cut2 = random.randint(0, len(g2))

        c1 = [clone_gene(x) for x in g1[:cut1]] + [clone_gene(x) for x in g2[cut2:]]
        c2 = [clone_gene(x) for x in g2[:cut2]] + [clone_gene(x) for x in g1[cut1:]]

        child1.od_allocations[key] = merge_and_normalize(c1)
        child2.od_allocations[key] = merge_and_normalize(c2)

    return child1, child2


def random_initial_individual(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]], max_paths=3) -> Individual:
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths:
            continue
        k = random.randint(1, min(max_paths, len(paths)))
        chosen = random.sample(paths, k)
        raw = [PathAllocation(path=p, share=random.random()) for p in chosen]
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = merge_and_normalize(raw)
    return ind


def mutate_add(ind: Individual, batch: Batch, path_lib):
    key = (batch.origin, batch.destination, batch.batch_id)
    od = (batch.origin, batch.destination)
    allocs = ind.od_allocations.get(key, [])
    pool = path_lib.get(od, [])
    if not pool:
        return False

    cur = {a.path for a in allocs}
    candidates = [p for p in pool if p not in cur]

    if not candidates:
        if allocs:
            repl = random.choice(pool)
            allocs[random.randrange(len(allocs))] = PathAllocation(path=repl, share=0.2)
            ind.od_allocations[key] = merge_and_normalize(allocs)
            return True
        return False

    new_path = random.choice(candidates)
    allocs.append(PathAllocation(path=new_path, share=0.2))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_del(ind: Individual, batch: Batch):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if len(allocs) <= 1:
        return False
    allocs.pop(random.randrange(len(allocs)))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mod(ind: Individual, batch: Batch):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs:
        return False
    target = random.choice(allocs)
    target.share *= random.uniform(0.5, 1.5)
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def path_from_arcs(new_arcs: List[Arc], origin: str, destination: str, path_id: int = -1) -> Optional[Path]:
    if not new_arcs:
        return None
    nodes = [new_arcs[0].from_node] + [a.to_node for a in new_arcs]
    if nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(set(nodes)) != len(nodes):
        return None
    modes = [a.mode for a in new_arcs]
    base_cost = sum(a.cost_per_teu_km * a.distance for a in new_arcs)
    base_emis = sum(a.emission_per_teu_km * a.distance for a in new_arcs)
    base_time = sum(a.distance / max(a.speed_kmh, 1.0) for a in new_arcs)
    return Path(
        path_id=path_id,
        origin=origin,
        destination=destination,
        nodes=nodes,
        modes=modes,
        arcs=new_arcs,
        base_cost_per_teu=base_cost,
        base_emission_per_teu=base_emis,
        base_travel_time_h=base_time
    )


def mutate_mode(ind: Individual, batch: Batch, tt_dict, arc_lookup, max_trials: int = 20):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs:
        return False

    idx = random.randrange(len(allocs))
    old_alloc = allocs[idx]
    p = old_alloc.path
    if not p.arcs:
        return False

    arc_i = random.randrange(len(p.arcs))
    old_arc = p.arcs[arc_i]
    u, v = old_arc.from_node, old_arc.to_node

    modes_all = [m for m in ["road", "rail", "water"] if m != old_arc.mode]
    if not modes_all:
        return False

    for _ in range(max_trials):
        new_mode = random.choice(modes_all)
        k_arc = (u, v, new_mode)
        if k_arc not in arc_lookup:
            continue

        if new_mode != "road":
            if not tt_dict.get((u, v, new_mode), []):
                continue

        new_arcs = list(p.arcs)
        new_arcs[arc_i] = arc_lookup[k_arc]
        new_path = path_from_arcs(new_arcs, p.origin, p.destination, path_id=-1)
        if new_path is None:
            continue

        allocs_new = deepcopy(allocs)
        allocs_new[idx] = PathAllocation(path=new_path, share=old_alloc.share)
        ind.od_allocations[key] = merge_and_normalize(allocs_new)
        return True

    return False


# ========================
# Adaptive roulette
# ========================

def _normalise_probs(scores: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in scores.values())
    if s <= 1e-12:
        k = len(scores)
        return {op: 1.0 / k for op in scores.keys()}
    return {op: max(v, 0.0) / s for op, v in scores.items()}


def is_improved(before: Individual, after: Individual, eps_pen=1e-9, eps_obj=1e-12) -> bool:
    """
    Success definition aligned with constraint-domination:
    1) infeasible -> feasible is success
    2) if both infeasible: lower penalty is success
    3) if both feasible: objective-sum decrease as tie-break (simple & stable)
    """
    if after.feasible and (not before.feasible):
        return True
    if (not after.feasible) and (not before.feasible):
        return after.penalty < before.penalty - eps_pen

    # both feasible
    sb = float(sum(before.objectives))
    sa = float(sum(after.objectives))
    return sa < sb - eps_obj


class AdaptiveRoulette:
    def __init__(self, ops: List[str], init_probs: Dict[str, float], ema_alpha: float = 0.10,
                 min_prob: float = 0.05, score_eps: float = 1e-3):
        self.ops = list(ops)
        self.ema_alpha = float(ema_alpha)
        self.min_prob = float(min_prob)
        self.score_eps = float(score_eps)

        base = {op: float(init_probs.get(op, 1.0 / len(self.ops))) for op in self.ops}
        base = _normalise_probs(base)
        self.quality = {op: base[op] for op in self.ops}
        self.prob = self._quality_to_prob()

    def _quality_to_prob(self) -> Dict[str, float]:
        scores = {op: (self.quality[op] + self.score_eps) for op in self.ops}
        p = _normalise_probs(scores)

        k = len(self.ops)
        mp = min(self.min_prob, 1.0 / k - 1e-9) if k > 1 else 1.0
        remain = 1.0 - k * mp
        p = {op: mp + remain * p[op] for op in self.ops}

        s = sum(p.values())
        return {op: p[op] / s for op in self.ops}

    def sample(self) -> str:
        r = random.random()
        cum = 0.0
        for op in self.ops:
            cum += self.prob[op]
            if r <= cum:
                return op
        return self.ops[-1]

    def update(self, op: str, success: bool):
        y = 1.0 if success else 0.0
        q = self.quality.get(op, 0.0)
        self.quality[op] = (1.0 - self.ema_alpha) * q + self.ema_alpha * y
        self.prob = self._quality_to_prob()


def apply_mutation_op(ind: Individual, op: str, batch: Batch, path_lib, tt_dict, arc_lookup) -> bool:
    if op == "add":
        return mutate_add(ind, batch, path_lib)
    if op == "del":
        return mutate_del(ind, batch)
    if op == "mod":
        return mutate_mod(ind, batch)
    if op == "mode":
        return mutate_mode(ind, batch, tt_dict, arc_lookup)
    return False


def mutate_roulette_adaptive(
    ind: Individual,
    batches: List[Batch],
    path_lib, tt_dict, arc_lookup,
    roulette: AdaptiveRoulette,
    parent_snapshot: Individual,
    arcs: List[Arc],
    waiting_cost_per_teu_h: float,
    wait_emis_g_per_teu_h: float
) -> Tuple[str, bool, bool]:
    batch = random.choice(batches)
    op = roulette.sample()
    ok = apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup)
    if not ok:
        roulette.update(op, success=False)
        return op, False, False

    repair_missing_allocations(ind, batches, path_lib)
    evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)

    success = is_improved(parent_snapshot, ind)
    roulette.update(op, success=success)
    return op, True, success


# ========================
# NSGA-II components
# ========================

def dominates(a: Individual, b: Individual) -> bool:
    # constraint-domination
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False

    if a.feasible and b.feasible:
        return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
                any(x < y for x, y in zip(a.objectives, b.objectives)))

    # both infeasible
    if a.penalty < b.penalty - 1e-12:
        return True
    if b.penalty < a.penalty - 1e-12:
        return False

    return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
            any(x < y for x, y in zip(a.objectives, b.objectives)))


def non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    S: Dict[Individual, List[Individual]] = {p: [] for p in pop}
    n: Dict[Individual, int] = {p: 0 for p in pop}
    fronts: List[List[Individual]] = [[]]

    for p in pop:
        for q in pop:
            if p is q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[Individual] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    return fronts[:-1]


def crowding_distance(front: List[Individual]) -> Dict[Individual, float]:
    l = len(front)
    d: Dict[Individual, float] = {ind: 0.0 for ind in front}
    if l == 0:
        return d

    m = len(front[0].objectives)
    for i in range(m):
        front.sort(key=lambda x: x.objectives[i])
        d[front[0]] = float("inf")
        d[front[-1]] = float("inf")
        rng = front[-1].objectives[i] - front[0].objectives[i]
        if abs(rng) <= 1e-12:
            continue
        for j in range(1, l - 1):
            d[front[j]] += (front[j + 1].objectives[i] - front[j - 1].objectives[i]) / rng
    return d


def tournament_select(pop: List[Individual], dists: Dict[Individual, float], ranks: Dict[Individual, int]) -> Individual:
    a, b = random.sample(pop, 2)
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    if dists[a] > dists[b]:
        return a
    return b


class HypervolumeCalculator:
    """
    Monte Carlo HV in 3D
    (Best for trend monitoring; keep consistent reference point within each run.)
    """
    def __init__(self, ref_point: Tuple[float, float, float], num_samples=10000):
        self.ref_point = np.array(ref_point, dtype=float)
        self.num_samples = int(num_samples)
        self.ideal_point = np.zeros(3, dtype=float)
        self.samples = np.random.uniform(low=self.ideal_point, high=self.ref_point, size=(self.num_samples, 3))

    def calculate(self, pareto_front_inds: List[Individual]) -> float:
        if not pareto_front_inds:
            return 0.0
        front_objs = np.array([ind.objectives for ind in pareto_front_inds], dtype=float)
        valid_mask = np.all(front_objs <= self.ref_point, axis=1)
        valid_objs = front_objs[valid_mask]
        if len(valid_objs) == 0:
            return 0.0

        S = self.samples[:, np.newaxis, :]
        O = valid_objs[np.newaxis, :, :]
        dominated = np.all(O <= S, axis=2)  # sample dominated by some point
        dominated_samples = np.any(dominated, axis=1)
        return float(np.sum(dominated_samples) / float(self.num_samples))


def unique_individuals_by_objectives(front: List[Individual], tol: float = 1e-3) -> List[Individual]:
    uniq: List[Individual] = []
    seen: List[Tuple[float, float, float]] = []
    for ind in front:
        obj = ind.objectives
        is_dup = False
        for o in seen:
            if (abs(obj[0] - o[0]) <= tol and abs(obj[1] - o[1]) <= tol and abs(obj[2] - o[2]) <= tol):
                is_dup = True
                break
        if not is_dup:
            seen.append(obj)
            uniq.append(ind)
    return uniq


# ========================
# Metrics: P*, IGD+, Spacing
# ========================

def dominates_obj(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))


def nondominated_set(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    pts = unique_objective_tuples(points, tol=1e-9)
    nd = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if dominates_obj(q, p):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return nd


def normalize_points(points: List[Tuple[float, float, float]], mins, maxs):
    out = []
    for p in points:
        pp = []
        for i in range(3):
            rng = maxs[i] - mins[i]
            if rng <= 1e-12:
                pp.append(0.0)
            else:
                pp.append((p[i] - mins[i]) / rng)
        out.append(tuple(pp))
    return out


def igd_plus(P_star: List[Tuple[float, float, float]], A: List[Tuple[float, float, float]]) -> float:
    if not P_star or not A:
        return float("inf")
    P = np.array(P_star, dtype=float)
    Q = np.array(A, dtype=float)

    dists = []
    for p in P:
        diff = Q - p
        diff = np.maximum(diff, 0.0)  # IGD+ uses only dominated dimensions
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(float(np.min(dist)))
    return float(np.mean(dists))


def spacing_metric(A: List[Tuple[float, float, float]]) -> float:
    if A is None or len(A) < 2:
        return 0.0
    Q = np.array(A, dtype=float)
    n = Q.shape[0]
    dmin = []
    for i in range(n):
        diff = Q - Q[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist[i] = np.inf
        dmin.append(float(np.min(dist)))
    dmin = np.array(dmin)
    return float(np.sqrt(np.sum((dmin - np.mean(dmin)) ** 2) / max(1, n - 1)))


# ========================
# Plot helpers (mutation + metrics)
# ========================

def aggregate_mutation_over_runs(mut_runs: List[dict], generations: int):
    share_runs = []
    rate_runs = []
    eff_runs = []
    prob_runs = []

    for tr in mut_runs:
        attempt = np.vstack([tr["attempt"][op] for op in OPS]).astype(float)  # (4,G)
        success = np.vstack([tr["success"][op] for op in OPS]).astype(float)  # (4,G)
        total_attempt = np.sum(attempt, axis=0)

        share = attempt / np.maximum(total_attempt, 1.0)
        rate = success / np.maximum(attempt, 1.0)
        eff = share * rate

        share_runs.append(share)
        rate_runs.append(rate)
        eff_runs.append(eff)

        if "prob" in tr:
            prob = np.vstack([tr["prob"][op] for op in OPS]).astype(float)
            prob_runs.append(prob)

    share_runs = np.stack(share_runs, axis=0)  # (R,4,G)
    rate_runs = np.stack(rate_runs, axis=0)
    eff_runs = np.stack(eff_runs, axis=0)

    out = {
        "share_mean": np.mean(share_runs, axis=0),
        "share_std": np.std(share_runs, axis=0),
        "rate_mean": np.mean(rate_runs, axis=0),
        "rate_std": np.std(rate_runs, axis=0),
        "eff_mean": np.mean(eff_runs, axis=0),
        "eff_std": np.std(eff_runs, axis=0),
    }

    if prob_runs:
        prob_runs = np.stack(prob_runs, axis=0)
        out["prob_mean"] = np.mean(prob_runs, axis=0)
        out["prob_std"] = np.std(prob_runs, axis=0)

    return out


def plot_mutation_attempt_stacked(gen, share_mean, save="mutation_attempt_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(
        gen,
        share_mean[0], share_mean[1], share_mean[2], share_mean[3],
        labels=["Add", "Delete", "Modify", "Mode"],
        alpha=0.85
    )
    plt.margins(0, 0)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Attempt Share")
    plt.title("Mutation Roulette: Attempt Share (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_mutation_adaptive_prob_stacked(gen, prob_mean, save="mutation_adaptive_prob_stacked.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.stackplot(
        gen,
        prob_mean[0], prob_mean[1], prob_mean[2], prob_mean[3],
        labels=["Add", "Delete", "Modify", "Mode"],
        alpha=0.85
    )
    plt.margins(0, 0)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Selection Probability")
    plt.title("Adaptive Roulette: Operator Selection Probability (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_mutation_success_rate_2x2(gen, rate_mean, rate_std, save="mutation_success_rate.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, op in enumerate(OPS):
        ax = axes[i]
        m = rate_mean[i]
        s = rate_std[i]
        ax.plot(gen, m, linewidth=2.5, label=f"{op} mean")
        ax.fill_between(gen, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.2, label="± std")
        ax.set_title(op.upper())
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Mutation Operators: Success Rate (mean ± std over runs)", y=0.98)
    fig.text(0.5, 0.03, "Generation", ha="center")
    fig.text(0.03, 0.5, "Success Rate", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    plt.savefig(save)
    print("Saved:", save)


def plot_mutation_effective_contribution(gen, eff_mean, save="mutation_effective_contribution.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    for i, op in enumerate(OPS):
        plt.plot(gen, eff_mean[i], linewidth=2.0, label=op)
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Attempt × Success")
    plt.title("Mutation Operators: Effective Contribution (mean over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std, save="convergence_hv_igd.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)
    ax.plot(gen, hv_mean, linewidth=2.0, label="HV (mean)")
    ax.fill_between(gen, hv_mean - hv_std, hv_mean + hv_std, alpha=0.2, label="HV ± std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("HV")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(gen, igd_mean, linewidth=2.0, linestyle="--", label="IGD+ (mean)")
    ax2.fill_between(gen, np.maximum(igd_mean - igd_std, 0), igd_mean + igd_std, alpha=0.15, label="IGD+ ± std")
    ax2.set_ylabel("IGD+ (lower is better)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Convergence: HV and IGD+ (mean ± std over runs)")
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_spacing_curve(gen, sp_mean, sp_std, save="diversity_spacing.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, sp_mean, linewidth=2.0, label="Spacing (mean)")
    plt.fill_between(gen, np.maximum(sp_mean - sp_std, 0), sp_mean + sp_std, alpha=0.2, label="± std")
    plt.xlabel("Generation")
    plt.ylabel("Spacing (lower is better)")
    plt.title("Diversity: Spacing (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio.png"):
    plt.figure(figsize=(10, 4), dpi=300)
    plt.plot(gen, fr_mean, linewidth=2.0, label="Feasible Ratio (mean)")
    plt.fill_between(gen, np.clip(fr_mean - fr_std, 0, 1), np.clip(fr_mean + fr_std, 0, 1), alpha=0.2, label="± std")
    plt.ylim(0, 1.0)
    plt.xlabel("Generation")
    plt.ylabel("Feasible Ratio")
    plt.title("Constraint Handling: Feasible Ratio (mean ± std over runs)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_violation_breakdown_stacked(gen, vio_mean_dict_mean, save="violation_breakdown.png"):
    miss_alloc = np.array(vio_mean_dict_mean["miss_alloc"])
    miss_tt = np.array(vio_mean_dict_mean["miss_tt"])
    cap_excess = np.array(vio_mean_dict_mean["cap_excess"])
    late_h = np.array(vio_mean_dict_mean["late_h"])
    wait_h = np.array(vio_mean_dict_mean["wait_h"])

    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(gen, miss_alloc, label="miss_alloc")
    plt.bar(gen, miss_tt, bottom=miss_alloc, label="miss_tt")
    plt.bar(gen, cap_excess, bottom=miss_alloc + miss_tt, label="cap_excess")
    plt.bar(gen, late_h, bottom=miss_alloc + miss_tt + cap_excess, label="late_h")
    plt.bar(gen, wait_h, bottom=miss_alloc + miss_tt + cap_excess + late_h, label="wait_h")

    plt.xlabel("Generation")
    plt.ylabel("Mean Violation Components (stacked)")
    plt.title("Diagnostics Breakdown (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


# ========================
# Pareto 3D plots
# ========================

def plot_pareto_3d_final_only(pareto_points: List[Tuple[float, float, float]], save: str, title: str):
    if not pareto_points:
        print("[WARN] No final Pareto points to plot.")
        return
    A = np.array(pareto_points, dtype=float)
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], marker="o", s=30, alpha=0.9)
    ax.set_xlabel("Cost")
    ax.set_ylabel("Emission (gCO2)")
    ax.set_zlabel("Makespan")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


def plot_pareto_3d_all_generations(
    all_gen_points_with_gen: List[Tuple[float, float, float, int]],
    final_pareto_points: Optional[List[Tuple[float, float, float]]] = None,
    save: str = "pareto_3d_allgens_best_run.png",
    title: str = "Pareto Points Across All Generations (Best Run)",
    cmap_name: str = "turbo",
):
    if not all_gen_points_with_gen:
        print("[WARN] No all-generation Pareto points to plot.")
        return

    P = np.array([(c, e, t) for (c, e, t, g) in all_gen_points_with_gen], dtype=float)
    G = np.array([g for (_, _, _, g) in all_gen_points_with_gen], dtype=float)

    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=G, cmap=cmap_name, s=10, alpha=0.55)
    cbar = plt.colorbar(sc, ax=ax, pad=0.10, fraction=0.04)
    cbar.set_label("Generation")
    gmin, gmax = int(np.min(G)), int(np.max(G))
    ticks = np.linspace(gmin, gmax, num=6).astype(int)
    cbar.set_ticks(ticks)

    if final_pareto_points:
        F = np.array(final_pareto_points, dtype=float)
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], marker="^", s=40, alpha=0.95, label="Final feasible Pareto")
        ax.legend(loc="best")

    ax.set_xlabel("Cost")
    ax.set_ylabel("Emission (gCO2)")
    ax.set_zlabel("Makespan")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


# ========================
# Runner (one run)
# ========================

def run_nsga2_analytics(filename="data.xlsx", pop_size=60, generations=200):
    print("Loading data...")
    node_names, arcs, timetables, batches, waiting_cost_per_teu_h, wait_emis_g_per_teu_h = load_network_from_extended(filename)

    tt_dict = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)

    print("Building path library...")
    path_lib = build_path_library(node_names, arcs, batches, tt_dict, arc_lookup)
    sanity_check_path_lib(batches, path_lib)

    init_probs = _normalise_probs({"add": W_ADD, "del": W_DEL, "mod": W_MOD, "mode": W_MODE})
    roulette = AdaptiveRoulette(ops=OPS, init_probs=init_probs, ema_alpha=0.10, min_prob=0.05, score_eps=1e-3)

    # init pop
    population: List[Individual] = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib)
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)
        population.append(ind)

    # HV ref point per run
    all_objs = np.array([ind.objectives for ind in population], dtype=float)
    ref_point = np.max(all_objs, axis=0) * 1.2
    hv_calc = HypervolumeCalculator(ref_point=tuple(ref_point), num_samples=10000)
    print(f"Reference Point for HV: {ref_point}")

    # history
    front_hist_objs: List[List[Tuple[float, float, float]]] = []
    hv_history: List[float] = []
    feasible_ratio_hist: List[float] = []
    vio_mean_hist: Dict[str, List[float]] = {k: [] for k in ["miss_alloc", "miss_tt", "cap_excess", "late_h", "wait_h"]}

    mut_tracker = {
        "attempt": {op: [0] * generations for op in OPS},
        "success": {op: [0] * generations for op in OPS},
        "prob":    {op: [0.0] * generations for op in OPS},
    }

    for gen in range(generations):
        for op in OPS:
            mut_tracker["prob"][op][gen] = float(roulette.prob[op])

        fronts = non_dominated_sort(population)
        front0 = fronts[0]

        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        current_front_objs = [ind.objectives for ind in front0_unique]
        front_hist_objs.append(current_front_objs)

        hv_history.append(hv_calc.calculate(front0_unique))
        feasible_ratio_hist.append(sum(1 for ind in population if ind.feasible) / float(len(population)))

        for k in vio_mean_hist.keys():
            vals = [ind.vio_breakdown.get(k, 0.0) for ind in population]
            vio_mean_hist[k].append(float(np.mean(vals)) if vals else 0.0)

        if gen % 10 == 0 or gen == generations - 1:
            feas_n = len(feasible_front0)
            best_pen = min(ind.penalty for ind in front0_unique) if front0_unique else float("inf")
            best_cost = min(ind.objectives[0] for ind in front0_unique) if front0_unique else float("inf")
            print(f"Gen {gen:03d} | Front0={len(front0):2d} | FeasFront0={feas_n:2d} | "
                  f"FeasRatio={feasible_ratio_hist[-1]:.2%} | BestCost={best_cost:.3e} | BestPenalty={best_pen:.3e} | HV={hv_history[-1]:.4f}")

        # ranks & crowding
        ranks: Dict[Individual, int] = {}
        for r, fr in enumerate(fronts):
            for ind in fr:
                ranks[ind] = r

        dists: Dict[Individual, float] = {}
        for fr in fronts:
            dists.update(crowding_distance(fr))

        # mating pool
        mating_pool: List[Individual] = []
        while len(mating_pool) < pop_size:
            mating_pool.append(tournament_select(population, dists, ranks))

        # offspring
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)

            if random.random() < 0.7:
                c1, c2 = crossover_structural(p1, p2, batches)
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)

            if random.random() < 0.35:
                snap = deepcopy(c1)
                op, ok, suc = mutate_roulette_adaptive(
                    c1, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            if random.random() < 0.35:
                snap = deepcopy(c2)
                op, ok, suc = mutate_roulette_adaptive(
                    c2, batches, path_lib, tt_dict, arc_lookup,
                    roulette, snap, arcs,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h
                )
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if suc else 0)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)
            evaluate_individual(c2, batches, arcs, tt_dict, waiting_cost_per_teu_h, wait_emis_g_per_teu_h)

            offspring.append(c1)
            offspring.append(c2)

        # environmental selection
        combined = population + offspring
        fronts2 = non_dominated_sort(combined)

        new_pop: List[Individual] = []
        for fr in fronts2:
            if len(new_pop) + len(fr) <= pop_size:
                new_pop.extend(fr)
            else:
                d = crowding_distance(fr)
                fr.sort(key=lambda x: d[x], reverse=True)
                new_pop.extend(fr[:pop_size - len(new_pop)])
                break

        population = new_pop

    # final pareto: feasible only
    final_fronts = non_dominated_sort(population)
    f0 = final_fronts[0]
    feasible_f0 = [ind for ind in f0 if ind.feasible]
    pareto = unique_individuals_by_objectives(feasible_f0, tol=1e-3)

    return (population, pareto, batches,
            front_hist_objs, hv_history,
            feasible_ratio_hist, vio_mean_hist,
            mut_tracker)


# ========================
# Output helpers
# ========================

def print_pure_structure(ind: Individual, batches: List[Batch], sol_name="Solution"):
    print(f"\n===== {sol_name} Final Structure (Node+Mode | Share) =====")
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            continue
        print(f"\nBatch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n")
        for a in allocs:
            print(a)


def save_pareto_solutions(pareto: List[Individual], batches: List[Batch], filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-3)
    pareto = [ind for ind in pareto if ind.feasible]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (Feasible Only) =====\n\n")
        if not pareto:
            f.write("NO FEASIBLE SOLUTION FOUND.\n")
            f.write("Check: path_lib coverage, timetable coverage, capacity scale, or HARD_TIME_WINDOW.\n")
            print(f"Saved 0 feasible Pareto solutions to {filename}")
            return

        for i, ind in enumerate(pareto):
            cost, emis, t = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Objectives: Cost={cost:.6f}, Emission_gCO2={emis:.6f}, Makespan={t:.6f}, "
                    f"Penalty={ind.penalty:.6f}, Feasible={ind.feasible}, FeasibleHardNoLate={ind.feasible_hard}, "
                    f"Breakdown={ind.vio_breakdown}\n\n")

            for b in batches:
                key = (b.origin, b.destination, b.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if not allocs:
                    continue
                f.write(f"Batch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n\n")
                for a in allocs:
                    f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"Saved {len(pareto)} feasible Pareto solutions to {filename}")


# ========================
# Main (multi-runs) + plots
# ========================

if __name__ == "__main__":
    filename = "data.xlsx"
    pop_size = 60
    generations = 200
    runs = 30

    run_front_hist = []
    run_hv_hist = []
    run_feasible_ratio = []
    run_vio_mean = []
    mut_runs = []
    run_rows = []

    best_run_idx = -1
    best_run_hv = -1.0
    best_pareto: Optional[List[Individual]] = None
    best_batches: Optional[List[Batch]] = None
    best_front_hist: Optional[List[List[Tuple[float, float, float]]]] = None

    print(f"[CONFIG] HARD_TIME_WINDOW={HARD_TIME_WINDOW}  (False=lateness soft via cost; True=lateness hard)")
    print(f"[CONFIG] Waiting in objectives: cost_per_teu_h default={WAITING_COST_PER_TEU_HOUR_DEFAULT}, emis_g_per_teu_h default={WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT}")
    print(f"[CONFIG] Lateness in cost: LATE_COST_PER_TEU_HOUR={LATE_COST_PER_TEU_HOUR:.3e}")
    print(f"[CONFIG] Path lib: topK per criterion={PATHS_TOPK_PER_CRITERION}, cap_total={PATH_LIB_CAP_TOTAL}, dfs_pool={DFS_MAX_PATHS_PER_OD}")

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id}/{runs-1}, seed={seed} ==========")
        t0 = time.perf_counter()

        pop, pareto, batches, front_hist, hv_hist, fr_hist, vio_hist, mut_tracker = run_nsga2_analytics(
            filename=filename, pop_size=pop_size, generations=generations
        )

        t1 = time.perf_counter()
        runtime_s = float(t1 - t0)
        runtime_min = runtime_s / 60.0

        run_front_hist.append(front_hist)
        run_hv_hist.append(hv_hist)
        run_feasible_ratio.append(fr_hist)
        run_vio_mean.append(vio_hist)
        mut_runs.append(mut_tracker)

        final_hv = hv_hist[-1] if hv_hist else 0.0
        final_fr = fr_hist[-1] if fr_hist else 0.0
        pareto_size = len(pareto)

        run_rows.append({
            "run_id": run_id,
            "seed": seed,
            "runtime_s": runtime_s,
            "runtime_min": runtime_min,
            "final_HV": final_hv,
            "final_feasible_ratio": final_fr,
            "final_pareto_size": pareto_size
        })

        print(f"[RUN {run_id}] Final HV={final_hv:.4f}, Pareto size={pareto_size}, Runtime={runtime_s:.2f}s")

        if final_hv > best_run_hv:
            best_run_hv = final_hv
            best_run_idx = run_id
            best_pareto = pareto
            best_batches = batches
            best_front_hist = front_hist

    # ===== Save runtime summary to Excel
    df_runs = pd.DataFrame(run_rows).sort_values("run_id").reset_index(drop=True)
    print("\n=========== RUN TIME SUMMARY (per run) ===========")
    print(df_runs.to_string(index=False))
    print("=================================================\n")

    excel_name = "run_time_summary.xlsx"
    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="RunSummary", index=False)
    print("Saved Excel:", excel_name)

    # ===== Build P* from all runs & all generations front histories
    all_points = []
    for r in range(runs):
        for gen_front in run_front_hist[r]:
            all_points.extend(gen_front)

    P_star = nondominated_set(all_points)
    print(f"\n[P*] Pseudo-reference front size = {len(P_star)}")

    if len(P_star) > 0:
        P_arr = np.array(P_star, dtype=float)
        mins = np.min(P_arr, axis=0)
        maxs = np.max(P_arr, axis=0)
    else:
        mins = np.zeros(3, dtype=float)
        maxs = np.ones(3, dtype=float)

    # ===== Compute IGD+ and Spacing histories per run
    igd_runs = []
    sp_runs = []
    for r in range(runs):
        igd_hist = []
        sp_hist = []
        for gen_front in run_front_hist[r]:
            A = gen_front
            Pn = normalize_points(P_star, mins, maxs) if P_star else []
            An = normalize_points(A, mins, maxs) if A else []
            igd_hist.append(igd_plus(Pn, An))
            sp_hist.append(spacing_metric(An))
        igd_runs.append(igd_hist)
        sp_runs.append(sp_hist)

    # ===== Aggregate mean/std over runs (per generation)
    hv_runs = np.array(run_hv_hist, dtype=float)        # (R,G)
    igd_runs = np.array(igd_runs, dtype=float)
    sp_runs = np.array(sp_runs, dtype=float)
    fr_runs = np.array(run_feasible_ratio, dtype=float)

    gen = np.arange(generations)

    hv_mean, hv_std = np.mean(hv_runs, axis=0), np.std(hv_runs, axis=0)
    igd_mean, igd_std = np.mean(igd_runs, axis=0), np.std(igd_runs, axis=0)
    sp_mean, sp_std = np.mean(sp_runs, axis=0), np.std(sp_runs, axis=0)
    fr_mean, fr_std = np.mean(fr_runs, axis=0), np.std(fr_runs, axis=0)

    # violation breakdown mean over runs (per gen)
    vio_keys = ["miss_alloc", "miss_tt", "cap_excess", "late_h", "wait_h"]
    vio_mean_dict_mean = {k: [0.0] * generations for k in vio_keys}
    for k in vio_keys:
        mat = np.array([run_vio_mean[r][k] for r in range(runs)], dtype=float)  # (R,G)
        vio_mean_dict_mean[k] = list(np.mean(mat, axis=0))

    # mutation aggregation
    mut_agg = aggregate_mutation_over_runs(mut_runs, generations)

    # ===== Summary (final gen)
    def summarize_final(arr):
        arr = np.array(arr, dtype=float)
        final = arr[:, -1]
        return float(np.min(final)), float(np.max(final)), float(np.mean(final)), float(np.std(final))

    hv_best, hv_worst, hv_m, hv_s = summarize_final(hv_runs)
    igd_best, igd_worst, igd_m, igd_s = summarize_final(igd_runs)
    sp_best, sp_worst, sp_m, sp_s = summarize_final(sp_runs)
    fr_best, fr_worst, fr_m, fr_s = summarize_final(fr_runs)

    print("\n=========== SUMMARY OVER RUNS (Final Generation) ===========")
    print(f"Runs: {runs}")
    print(f"HV      (higher better) | best={hv_best:.4f}, worst={hv_worst:.4f}, mean={hv_m:.4f}, std={hv_s:.4f}")
    print(f"IGD+    (lower better ) | best={igd_best:.4f}, worst={igd_worst:.4f}, mean={igd_m:.4f}, std={igd_s:.4f}")
    print(f"Spacing (lower better ) | best={sp_best:.4f}, worst={sp_worst:.4f}, mean={sp_m:.4f}, std={sp_s:.4f}")
    print(f"FeasRatio (higher better)| best={fr_best:.2%}, worst={fr_worst:.2%}, mean={fr_m:.2%}, std={fr_s:.2%}")
    print(f"Best run index (by HV): {best_run_idx}, HV={best_run_hv:.4f}")
    print("============================================================\n")

    # ===== Save outputs for best run
    if best_pareto is not None and best_batches is not None:
        for i, ind in enumerate(best_pareto[:3]):
            print_pure_structure(ind, best_batches, f"BestRun Pareto Sol {i}")
        save_pareto_solutions(best_pareto, best_batches, filename="result.txt")

    # ===== Pareto 3D plots (best run)
    best_final_points = []
    if best_pareto:
        best_final_points = [ind.objectives for ind in best_pareto if ind.feasible]
        best_final_points = unique_objective_tuples(best_final_points, tol=1e-9)

    plot_pareto_3d_final_only(
        pareto_points=best_final_points,
        save="pareto_3d_best_run_final.png",
        title=f"Pareto Front (Final Gen) - Best Run #{best_run_idx}"
    )

    all_points_best_with_gen = []
    if best_front_hist:
        for g, gen_pts in enumerate(best_front_hist):
            for p in gen_pts:
                all_points_best_with_gen.append((p[0], p[1], p[2], g))

    # unique by rounded objectives + generation (avoid heavy duplicates)
    seen = set()
    uniq = []
    for (c, e, t, g) in all_points_best_with_gen:
        key = (round(c, 6), round(e, 6), round(t, 6), g)
        if key not in seen:
            seen.add(key)
            uniq.append((c, e, t, g))
    all_points_best_with_gen = uniq

    plot_pareto_3d_all_generations(
        all_gen_points_with_gen=all_points_best_with_gen,
        final_pareto_points=best_final_points if best_final_points else None,
        save="pareto_3d_allgens_best_run.png",
        title=f"Pareto Points Across {generations} Generations (Best Run #{best_run_idx})",
        cmap_name="turbo"
    )

    # ===== Academic plots (metrics & mutation)
    plot_mutation_attempt_stacked(gen, mut_agg["share_mean"], save="mutation_attempt_stacked.png")
    plot_mutation_success_rate_2x2(gen, mut_agg["rate_mean"], mut_agg["rate_std"], save="mutation_success_rate.png")
    plot_mutation_effective_contribution(gen, mut_agg["eff_mean"], save="mutation_effective_contribution.png")
    if "prob_mean" in mut_agg:
        plot_mutation_adaptive_prob_stacked(gen, mut_agg["prob_mean"], save="mutation_adaptive_prob_stacked.png")

    plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std, save="convergence_hv_igd.png")
    plot_spacing_curve(gen, sp_mean, sp_std, save="diversity_spacing.png")
    plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio.png")
    plot_violation_breakdown_stacked(gen, vio_mean_dict_mean, save="violation_breakdown.png")

    print("\nSaved figures:")
    print(" - pareto_3d_best_run_final.png")
    print(" - pareto_3d_allgens_best_run.png")
    print(" - mutation_attempt_stacked.png")
    print(" - mutation_success_rate.png")
    print(" - mutation_effective_contribution.png")
    print(" - mutation_adaptive_prob_stacked.png (if available)")
    print(" - convergence_hv_igd.png")
    print(" - diversity_spacing.png")
    print(" - feasible_ratio.png")
    print(" - violation_breakdown.png")
    print(" - run_time_summary.xlsx")
    print(" - result.txt")
