#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========================
# Global settings
# ========================

TIME_BUCKET_H = 1.0  # simulation granularity: 1 hour
CHINA_REGIONS = {"CN", "China"}
EUROPE_REGIONS = {"WE", "EE", "EU", "Europe"}
BIG_M = 1e6  # big penalty for capacity violation

# ---- Mutation roulette weights (tuneable)
W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20


# ========================
# 1. Data structures
# ========================

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float       # TEU / time-bucket
    cost_per_teu_km: float
    emission_per_teu_km: float
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
    quantity: float
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
    base_emission_per_teu: float
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

    def __eq__(self, other):
        if not isinstance(other, PathAllocation):
            return NotImplemented
        return (self.path == other.path and abs(self.share - other.share) < 1e-10)

    def __hash__(self):
        share_for_hash = round(self.share, 10)
        return hash((self.path, share_for_hash))

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            mode = self.path.modes[i]
            chain += f"{node}--({mode})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    # key = (origin, destination, batch_id)
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))  # (cost, emission, makespan)
    penalty: float = 0.0
    feasible: bool = False


# ========================
# 2. Core utilities: merge & normalise shares
# ========================

def clone_gene(alloc: PathAllocation) -> PathAllocation:
    return PathAllocation(path=alloc.path, share=alloc.share)


def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []

    merged_map: Dict[Path, float] = {}
    for a in allocs:
        merged_map[a.path] = merged_map.get(a.path, 0.0) + a.share

    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged_map.items()]

    total_share = sum(a.share for a in unique_allocs)
    if total_share <= 1e-9:
        if not unique_allocs:
            return []
        avg = 1.0 / len(unique_allocs)
        for a in unique_allocs:
            a.share = avg
    else:
        factor = 1.0 / total_share
        for a in unique_allocs:
            a.share *= factor

    # remove tiny shares
    unique_allocs = [a for a in unique_allocs if a.share > 0.001]

    # re-normalise
    if unique_allocs:
        final_total = sum(a.share for a in unique_allocs)
        if abs(final_total - 1.0) > 1e-6 and final_total > 1e-12:
            for a in unique_allocs:
                a.share /= final_total

    return unique_allocs


# ========================
# 3. Load data + build graph/library
# ========================

def load_network_from_extended(filename: str):
    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")

    # Nodes
    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).tolist()
    node_region = dict(zip(nodes_df["EnglishName"].astype(str),
                           nodes_df["Region"].astype(str)))

    # Arcs
    arcs_df = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []

    DAILY_HOURS = 24.0

    for _, row in arcs_df.iterrows():
        mode_raw = str(row["Mode"]).strip().lower()
        speed = 75.0 if mode_raw == "road" else (30.0 if mode_raw == "water" else 50.0)
        mode = "rail" if mode_raw == "rail" else mode_raw

        dist_str = str(row["Distance_km"])
        cleaned = "".join(ch for ch in dist_str if (ch.isdigit() or ch == "."))
        distance = float(cleaned) if cleaned else 0.0

        # capacity: TEU/day -> TEU/time-bucket
        if "Capacity_TEU" in arcs_df.columns and not pd.isna(row["Capacity_TEU"]):
            raw_cap = float(row["Capacity_TEU"])
        else:
            raw_cap = 1e9
        capacity = raw_cap * (TIME_BUCKET_H / DAILY_HOURS)

        arcs.append(Arc(
            from_node=str(row["OriginEN"]).strip(),
            to_node=str(row["DestEN"]).strip(),
            mode=mode,
            distance=distance,
            capacity=capacity,
            cost_per_teu_km=float(row["Cost_$_per_km"]),
            emission_per_teu_km=float(row["Emission_gCO2_per_tkm"]),
            speed_kmh=speed
        ))

    # Timetable
    tdf = pd.read_excel(xls, "Timetable")
    timetables: List[TimetableEntry] = []
    for _, row in tdf.iterrows():
        freq = float(row["Frequency_per_week"])
        hd = row["Headway_Hours"]
        hd = 168.0 / max(freq, 1.0) if pd.isna(hd) else float(hd)

        v = row["FirstDepartureHour"]
        fd = 0.0
        if not pd.isna(v):
            try:
                fd = float(str(v).split(":")[0]) if ":" in str(v) else float(v)
            except Exception:
                fd = 0.0

        timetables.append(TimetableEntry(
            from_node=str(row["OriginEN"]).strip(),
            to_node=str(row["DestEN"]).strip(),
            mode=str(row["Mode"]).strip().lower(),
            frequency_per_week=freq,
            first_departure_hour=fd,
            headway_hours=hd
        ))

    # Batches
    bdf = pd.read_excel(xls, "Batches")
    batches: List[Batch] = []
    for _, row in bdf.iterrows():
        origin = str(row["OriginEN"]).strip()
        dest = str(row["DestEN"]).strip()
        o_reg = node_region.get(origin)
        d_reg = node_region.get(dest)

        if o_reg in CHINA_REGIONS and d_reg in EUROPE_REGIONS:
            batches.append(Batch(
                batch_id=int(row["BatchID"]),
                origin=origin,
                destination=dest,
                quantity=float(row["QuantityTEU"]),
                ET=float(row["ET"]),
                LT=float(row["LT"])
            ))

    print(f"[INFO] Number of batches loaded: {len(batches)}")
    return node_names, arcs, timetables, batches


def build_graph(arcs: List[Arc]) -> Dict[str, List[Tuple[str, Arc]]]:
    g: Dict[str, List[Tuple[str, Arc]]] = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append((a.to_node, a))
    return g


def build_timetable_dict(timetables: List[TimetableEntry]) -> Dict[Tuple[str, str, str], List[TimetableEntry]]:
    tt_dict: Dict[Tuple[str, str, str], List[TimetableEntry]] = {}
    for t in timetables:
        key = (t.from_node, t.to_node, t.mode)
        tt_dict.setdefault(key, []).append(t)
    return tt_dict


def random_dfs_paths(graph, origin, dest, max_len=8, max_paths=50) -> List[List[Arc]]:
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


def build_path_library(node_names, arcs, batches, timetables) -> Dict[Tuple[str, str], List[Path]]:
    graph = build_graph(arcs)
    path_lib: Dict[Tuple[str, str], List[Path]] = {}
    next_path_id = 0

    for batch in batches:
        od = (batch.origin, batch.destination)
        if od in path_lib:
            continue

        arc_paths = random_dfs_paths(graph, batch.origin, batch.destination)
        paths_for_od: List[Path] = []

        for arc_seq in arc_paths:
            nodes = [arc_seq[0].from_node] + [a.to_node for a in arc_seq]
            modes = [a.mode for a in arc_seq]

            paths_for_od.append(Path(
                path_id=next_path_id,
                origin=batch.origin,
                destination=batch.destination,
                nodes=nodes,
                modes=modes,
                arcs=arc_seq,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in arc_seq),
                base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in arc_seq),
                base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in arc_seq),
            ))
            next_path_id += 1

        if paths_for_od:
            paths_for_od.sort(key=lambda p: p.base_cost_per_teu)
            path_lib[od] = paths_for_od[:20]

    return path_lib


# ========================
# 4. Evaluation (objectives + feasibility)
# ========================

def simulate_path_time_capacity(path: Path, batch: Batch, flow: float, tt_dict,
                                arc_flow_map) -> float:
    """
    - Road: depart immediately (ignore timetable)
    - Rail/Water: wait for next departure (simplified to first timetable entry)
    """
    t = batch.ET
    for arc in path.arcs:
        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

        if arc.mode == "road":
            entries = []
        else:
            key = (arc.from_node, arc.to_node, arc.mode)
            entries = tt_dict.get(key, [])

        if not entries:
            dep = t
        else:
            e = entries[0]
            if t <= e.first_departure_hour:
                dep = e.first_departure_hour
            else:
                waited = (t - e.first_departure_hour)
                n = math.ceil(waited / e.headway_hours)
                dep = e.first_departure_hour + n * e.headway_hours

        arr = dep + travel_time

        start_slot = int(dep)
        key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0) + flow

        t = arr

    return t - batch.ET


def evaluate_individual(ind: Individual, batches, path_lib, arcs, tt_dict):
    total_cost = 0.0
    total_emission = 0.0
    makespan = 0.0

    arc_flow_map: Dict[Tuple[Tuple[str, str, str], int], float] = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    penalty = 0.0

    missing_alloc_violation = False
    late_violation = False
    capacity_violation = False

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])

        if not allocs:
            penalty += 1e9
            missing_alloc_violation = True
            continue

        batch_finish_time = 0.0

        for alloc in allocs:
            if alloc.share <= 1e-6:
                continue

            flow = alloc.share * batch.quantity
            path = alloc.path

            total_cost += path.base_cost_per_teu * flow
            total_emission += path.base_emission_per_teu * flow

            travel_time = simulate_path_time_capacity(path, batch, flow, tt_dict, arc_flow_map)
            arrival_time = batch.ET + travel_time

            batch_finish_time = max(batch_finish_time, arrival_time)

            if arrival_time > batch.LT:
                penalty += (arrival_time - batch.LT) * 1000.0
                late_violation = True

        makespan = max(makespan, batch_finish_time)

    for (key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(key, 1e9)
        if flow > cap:
            penalty += (flow - cap) * BIG_M
            capacity_violation = True

    ind.objectives = (total_cost + penalty,
                      total_emission + penalty,
                      makespan)
    ind.penalty = penalty
    ind.feasible = not (missing_alloc_violation or late_violation or capacity_violation)


# ========================
# 5. Genetic operators
# ========================

def crossover_structural(ind1: Individual, ind2: Individual,
                         batches: List[Batch]) -> Tuple[Individual, Individual]:
    """
    Fragment crossover per batch key; share fixed by merge_and_normalize.
    """
    child1 = Individual()
    child2 = Individual()

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        genes1 = ind1.od_allocations.get(key, [])
        genes2 = ind2.od_allocations.get(key, [])

        if not genes1 and not genes2:
            continue
        if not genes1:
            child1.od_allocations[key] = [clone_gene(g) for g in genes2]
            child2.od_allocations[key] = [clone_gene(g) for g in genes2]
            continue
        if not genes2:
            child1.od_allocations[key] = [clone_gene(g) for g in genes1]
            child2.od_allocations[key] = [clone_gene(g) for g in genes1]
            continue

        cut1 = random.randint(0, len(genes1))
        cut2 = random.randint(0, len(genes2))

        c1_genes = [clone_gene(g) for g in genes1[:cut1]] + \
                   [clone_gene(g) for g in genes2[cut2:]]
        c2_genes = [clone_gene(g) for g in genes2[:cut2]] + \
                   [clone_gene(g) for g in genes1[cut1:]]

        child1.od_allocations[key] = merge_and_normalize(c1_genes)
        child2.od_allocations[key] = merge_and_normalize(c2_genes)

    return child1, child2


# ---- Mutation caches
_ARC_LOOKUP: Dict[Tuple[str, str, str], Arc] = {}
_ARC_MODE_OPTIONS: Dict[Tuple[str, str], List[str]] = {}
_CACHE_BUILT = False


def _build_arc_caches_from_path_lib(path_lib: Dict[Tuple[str, str], List[Path]]):
    global _ARC_LOOKUP, _ARC_MODE_OPTIONS, _CACHE_BUILT
    if _CACHE_BUILT:
        return

    arc_lookup: Dict[Tuple[str, str, str], Arc] = {}
    mode_opts: Dict[Tuple[str, str], set] = {}

    for _, paths in path_lib.items():
        for p in paths:
            for a in p.arcs:
                k3 = (a.from_node, a.to_node, a.mode)
                arc_lookup[k3] = a
                k2 = (a.from_node, a.to_node)
                mode_opts.setdefault(k2, set()).add(a.mode)

    _ARC_LOOKUP = arc_lookup
    _ARC_MODE_OPTIONS = {k: sorted(list(v)) for k, v in mode_opts.items()}
    _CACHE_BUILT = True


def _roulette_choice(items: List[Tuple[str, float]]) -> str:
    names = [x[0] for x in items]
    w = np.array([x[1] for x in items], dtype=float)
    s = float(w.sum())
    if s <= 0:
        return names[-1]
    w = w / s
    return str(np.random.choice(names, p=w))


def _mut_add_path(allocs: List[PathAllocation], od: Tuple[str, str],
                  path_lib: Dict[Tuple[str, str], List[Path]]):
    paths_in_lib = path_lib.get(od, [])
    if not paths_in_lib:
        return allocs

    current_structures = {a.path for a in allocs}
    candidates = [p for p in paths_in_lib if p not in current_structures]
    if not candidates:
        return allocs

    new_path = random.choice(candidates)
    allocs.append(PathAllocation(path=new_path, share=0.2))
    return merge_and_normalize(allocs)


def _mut_delete_path(allocs: List[PathAllocation]):
    if len(allocs) <= 1:
        return allocs
    allocs.pop(random.randint(0, len(allocs) - 1))
    return merge_and_normalize(allocs)


def _mut_modify_share(allocs: List[PathAllocation]):
    if not allocs:
        return allocs
    target = random.choice(allocs)
    target.share *= random.uniform(0.5, 1.5)
    return merge_and_normalize(allocs)


def _mut_mode_single_arc(allocs: List[PathAllocation]):
    if not allocs:
        return allocs

    a = random.choice(allocs)
    p = a.path
    if not p.arcs:
        return allocs

    pos = random.randrange(len(p.arcs))
    old_arc = p.arcs[pos]
    u, v = old_arc.from_node, old_arc.to_node
    old_mode = old_arc.mode

    modes = _ARC_MODE_OPTIONS.get((u, v), [])
    if len(modes) <= 1:
        return allocs

    other_modes = [m for m in modes if m != old_mode]
    if not other_modes:
        return allocs

    new_mode = random.choice(other_modes)
    new_arc = _ARC_LOOKUP.get((u, v, new_mode))
    if new_arc is None:
        return allocs

    new_arcs = list(p.arcs)
    new_arcs[pos] = new_arc
    new_modes = list(p.modes)
    if pos < len(new_modes):
        new_modes[pos] = new_mode
    else:
        return allocs

    new_path = Path(
        path_id=-1,
        origin=p.origin,
        destination=p.destination,
        nodes=list(p.nodes),
        modes=new_modes,
        arcs=new_arcs,
        base_cost_per_teu=sum(x.cost_per_teu_km * x.distance for x in new_arcs),
        base_emission_per_teu=sum(x.emission_per_teu_km * x.distance for x in new_arcs),
        base_travel_time_h=sum(x.distance / max(x.speed_kmh, 1.0) for x in new_arcs),
    )

    for i in range(len(allocs)):
        if allocs[i] == a:
            allocs[i] = PathAllocation(path=new_path, share=a.share)
            break

    return merge_and_normalize(allocs)


def mutate_structural(ind: Individual, batches: List[Batch],
                      path_lib: Dict[Tuple[str, str], List[Path]]):
    """
    Roulette among: add / del / modify-share / change-mode(one arc)
    """
    _build_arc_caches_from_path_lib(path_lib)

    batch = random.choice(batches)
    od = (batch.origin, batch.destination)
    key = (batch.origin, batch.destination, batch.batch_id)

    allocs = ind.od_allocations.get(key, [])
    paths_in_lib = path_lib.get(od, [])
    if not paths_in_lib:
        return

    can_del = len(allocs) > 1

    op = _roulette_choice([
        ("add", W_ADD),
        ("del", W_DEL if can_del else 0.0),
        ("mod", W_MOD),
        ("mode", W_MODE),
    ])

    if op == "add":
        allocs = _mut_add_path(allocs, od, path_lib)
    elif op == "del":
        allocs = _mut_delete_path(allocs)
    elif op == "mod":
        allocs = _mut_modify_share(allocs)
    else:
        allocs = _mut_mode_single_arc(allocs)

    ind.od_allocations[key] = allocs


# ========================
# 6. Metrics & timing
# ========================

class HypervolumeCalculator:
    """
    Monte-Carlo HV within [0, ref_point]^m, returns dominated sample ratio (0..1).
    Use ONLY on normalised objectives (e.g., within [0,1]) for meaningful values.
    """
    def __init__(self, ref_point: Tuple[float, float, float], num_samples=20000):
        self.ref_point = np.array(ref_point, dtype=float)
        self.num_samples = int(num_samples)
        self.ideal_point = np.zeros(3, dtype=float)
        self.samples = np.random.uniform(
            low=self.ideal_point,
            high=self.ref_point,
            size=(self.num_samples, 3)
        )

    def calculate(self, pareto_front_inds: List) -> float:
        if not pareto_front_inds:
            return 0.0

        front_objs = np.array([ind.objectives for ind in pareto_front_inds], dtype=float)

        valid_mask = np.all(front_objs <= self.ref_point, axis=1)
        valid_objs = front_objs[valid_mask]
        if len(valid_objs) == 0:
            return 0.0

        S = self.samples[:, np.newaxis, :]
        O = valid_objs[np.newaxis, :, :]

        is_dominated = np.all(O <= S, axis=2)
        dominated_samples = np.any(is_dominated, axis=1)

        ratio = np.sum(dominated_samples) / float(self.num_samples)
        return float(ratio)


def unique_individuals_by_objectives(front: List[Individual],
                                     tol: float = 1e-3) -> List[Individual]:
    unique: List[Individual] = []
    seen_objs: List[Tuple[float, float, float]] = []

    for ind in front:
        obj = ind.objectives
        is_dup = False
        for o in seen_objs:
            if (abs(obj[0] - o[0]) <= tol and
                    abs(obj[1] - o[1]) <= tol and
                    abs(obj[2] - o[2]) <= tol):
                is_dup = True
                break
        if not is_dup:
            seen_objs.append(obj)
            unique.append(ind)

    return unique


@dataclass
class RunStats:
    run_total_time: float = 0.0
    init_time: float = 0.0
    ndsort_time: float = 0.0
    crowd_time: float = 0.0
    hv_time: float = 0.0          # optional debug hv timing
    crossover_time: float = 0.0
    mutation_time: float = 0.0
    evaluation_time: float = 0.0

    ndsort_calls: int = 0
    crowd_calls: int = 0
    hv_calls: int = 0
    crossover_calls: int = 0
    mutation_calls: int = 0
    evaluation_calls: int = 0

    best_crossover_time: float = float("inf")
    eval_slower_than_best_crossover: int = 0

    gen_records: List[Dict] = field(default_factory=list)

    def add_gen_record(self, rec: Dict):
        self.gen_records.append(rec)

    def summary_dict(self) -> Dict:
        return {
            "run_total_time_s": self.run_total_time,
            "init_time_s": self.init_time,
            "ndsort_time_s": self.ndsort_time,
            "crowd_time_s": self.crowd_time,
            "hv_time_s": self.hv_time,
            "crossover_time_s": self.crossover_time,
            "mutation_time_s": self.mutation_time,
            "evaluation_time_s": self.evaluation_time,
            "ndsort_calls": self.ndsort_calls,
            "crowd_calls": self.crowd_calls,
            "hv_calls": self.hv_calls,
            "crossover_calls": self.crossover_calls,
            "mutation_calls": self.mutation_calls,
            "evaluation_calls": self.evaluation_calls,
            "best_crossover_time_s": (self.best_crossover_time if self.best_crossover_time < float("inf") else None),
            "eval_slower_than_best_crossover_calls": self.eval_slower_than_best_crossover,
        }


# ========================
# 6.2 Reference-front P*, normalisation, IGD+, Spacing, HV_norm
# ========================

def _dominates_obj(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_filter_objectives(objs: List[Tuple[float, float, float]], tol: float = 1e-9) -> List[Tuple[float, float, float]]:
    if not objs:
        return []

    # de-dup
    uniq = []
    for o in objs:
        ok = True
        for u in uniq:
            if (abs(o[0]-u[0]) <= tol and abs(o[1]-u[1]) <= tol and abs(o[2]-u[2]) <= tol):
                ok = False
                break
        if ok:
            uniq.append(o)

    # non-dominated
    nd = []
    for i, a in enumerate(uniq):
        dominated = False
        for j, b in enumerate(uniq):
            if i == j:
                continue
            if _dominates_obj(b, a):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    return nd


def build_reference_front_Pstar(all_feasible_nd_objs: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    return pareto_filter_objectives(all_feasible_nd_objs, tol=1e-6)


def normalise_by_Pstar(A_objs: List[Tuple[float, float, float]],
                       Pstar: List[Tuple[float, float, float]]):
    A = np.array(A_objs, dtype=float)
    P = np.array(Pstar, dtype=float)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    A_norm = (A - mins) / ranges
    return A_norm, mins, maxs


def igd_plus(A_norm: np.ndarray, P_norm: np.ndarray) -> float:
    """
    IGD+:
      for each p in P, compute min over a in A of || max(a - p, 0) ||
      then average over p.
    """
    if A_norm.size == 0 or P_norm.size == 0:
        return float("inf")
    dists = []
    for p in P_norm:
        diff = np.maximum(A_norm - p, 0.0)
        vals = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(np.min(vals))
    return float(np.mean(dists))


def spacing_metric(A_norm: np.ndarray) -> float:
    if A_norm.shape[0] <= 2:
        return 0.0
    D = np.sqrt(((A_norm[:, None, :] - A_norm[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    di = D.min(axis=1)
    dbar = di.mean()
    return float(np.sqrt(((di - dbar) ** 2).mean()))


class _TmpInd:
    def __init__(self, obj):
        self.objectives = tuple(obj)


def compute_gen_metrics_wrt_Pstar(history_fronts: List[Tuple[int, List[Tuple[float, float, float]]]],
                                  Pstar: List[Tuple[float, float, float]],
                                  hv_norm_samples=20000):
    gens = [g for g, _ in history_fronts]
    if not Pstar:
        return gens, [float("inf")]*len(gens), [float("inf")]*len(gens), [0.0]*len(gens)

    P_norm, _, _ = normalise_by_Pstar(Pstar, Pstar)
    P_norm = np.clip(P_norm, 0.0, 1.0)

    hv_norm_calc = HypervolumeCalculator(ref_point=(1.0, 1.0, 1.0), num_samples=hv_norm_samples)

    igds, sps, hvns = [], [], []
    for gen, objs in history_fronts:
        if not objs:
            igds.append(float("inf"))
            sps.append(float("inf"))
            hvns.append(0.0)
            continue

        A_norm, _, _ = normalise_by_Pstar(objs, Pstar)
        A_norm = np.clip(A_norm, 0.0, 1.0)

        igds.append(float(igd_plus(A_norm, P_norm)))
        sps.append(float(spacing_metric(A_norm)))

        tmp_inds = [_TmpInd(o) for o in A_norm.tolist()]
        hvns.append(float(hv_norm_calc.calculate(tmp_inds)))

    return gens, igds, sps, hvns


# ========================
# 7. NSGA-II core
# ========================

def random_initial_individual(batches, path_lib, max_paths=3) -> Individual:
    ind = Individual()
    for batch in batches:
        od = (batch.origin, batch.destination)
        paths = path_lib.get(od, [])
        if not paths:
            continue

        k = random.randint(1, min(max_paths, len(paths)))
        chosen = random.sample(paths, k)
        raw_allocs = [PathAllocation(path=p, share=random.random()) for p in chosen]

        key = (batch.origin, batch.destination, batch.batch_id)
        ind.od_allocations[key] = merge_and_normalize(raw_allocs)
    return ind


def dominates(a: Individual, b: Individual) -> bool:
    # constraint-dominance
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False

    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and \
        any(x < y for x, y in zip(a.objectives, b.objectives))


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
        next_front: List[Individual] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(front: List[Individual]) -> Dict[Individual, float]:
    l = len(front)
    d: Dict[Individual, float] = {ind: 0.0 for ind in front}
    if l == 0:
        return d

    m = len(front[0].objectives)
    for i in range(m):
        front.sort(key=lambda x: x.objectives[i])
        d[front[0]] = float('inf')
        d[front[-1]] = float('inf')
        rng = front[-1].objectives[i] - front[0].objectives[i]
        if rng == 0:
            continue
        for j in range(1, l - 1):
            d[front[j]] += (front[j + 1].objectives[i] -
                            front[j - 1].objectives[i]) / rng
    return d


def tournament_select(pop: List[Individual],
                      dists: Dict[Individual, float],
                      ranks: Dict[Individual, int]) -> Individual:
    a, b = random.sample(pop, 2)
    if ranks[a] < ranks[b]:
        return a
    if ranks[b] < ranks[a]:
        return b
    if dists[a] > dists[b]:
        return a
    return b


def run_nsga2_analytics(filename="data.xlsx",
                        pop_size=60,
                        generations=300,
                        log_every=10):
    """
    Returns:
      population, pareto_front, batches, history_fronts, stats,
      plus (final_feasible_nd_objs, all_feasible_front0_objs_over_gens)
    """
    stats = RunStats()
    t_run0 = time.perf_counter()

    print("Loading data...")
    node_names, arcs, timetables, batches = load_network_from_extended(filename)
    tt_dict = build_timetable_dict(timetables)
    path_lib = build_path_library(node_names, arcs, batches, timetables)

    # init
    t_init0 = time.perf_counter()
    population: List[Individual] = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib)

        t0 = time.perf_counter()
        evaluate_individual(ind, batches, path_lib, arcs, tt_dict)
        dt = time.perf_counter() - t0
        stats.evaluation_time += dt
        stats.evaluation_calls += 1

        population.append(ind)
    stats.init_time = time.perf_counter() - t_init0

    history_fronts: List[Tuple[int, List[Tuple[float, float, float]]]] = []
    all_feasible_front0_objs: List[Tuple[float, float, float]] = []

    for gen in range(generations):
        t_gen0 = time.perf_counter()

        # NDSort
        t0 = time.perf_counter()
        fronts = non_dominated_sort(population)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        front0 = fronts[0]
        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        current_front_objs = [ind.objectives for ind in front0_unique]
        history_fronts.append((gen, current_front_objs))

        # collect feasible front0 objs for P*
        for ind in feasible_front0:
            all_feasible_front0_objs.append(ind.objectives)

        feasible_pop = sum(1 for ind in population if ind.feasible)
        if (gen % log_every == 0) or (gen == generations - 1):
            print(f"Gen {gen}: Front0Unique={len(front0_unique)} | FeasiblePop={feasible_pop}/{len(population)}")

        # ranks
        ranks: Dict[Individual, int] = {}
        for r, f in enumerate(fronts):
            for ind in f:
                ranks[ind] = r

        # crowding
        t0 = time.perf_counter()
        dists: Dict[Individual, float] = {}
        for f in fronts:
            dists.update(crowding_distance(f))
        dt = time.perf_counter() - t0
        stats.crowd_time += dt
        stats.crowd_calls += 1

        # selection
        mating_pool: List[Individual] = []
        while len(mating_pool) < pop_size:
            mating_pool.append(tournament_select(population, dists, ranks))

        offspring: List[Individual] = []

        gen_crossover_time = 0.0
        gen_mutation_time = 0.0
        gen_eval_time = 0.0
        gen_eval_calls = 0
        gen_crossover_calls = 0
        gen_mutation_calls = 0

        while len(offspring) < pop_size:
            # crossover
            if random.random() < 0.7:
                p1, p2 = random.sample(mating_pool, 2)
                t0 = time.perf_counter()
                c1, c2 = crossover_structural(p1, p2, batches)
                dt = time.perf_counter() - t0

                stats.crossover_time += dt
                stats.crossover_calls += 1
                gen_crossover_time += dt
                gen_crossover_calls += 1

                if dt < stats.best_crossover_time:
                    stats.best_crossover_time = dt
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)

            # mutation
            if random.random() < 0.3:
                t0 = time.perf_counter()
                mutate_structural(c1, batches, path_lib)
                dt = time.perf_counter() - t0
                stats.mutation_time += dt
                stats.mutation_calls += 1
                gen_mutation_time += dt
                gen_mutation_calls += 1

            if random.random() < 0.3:
                t0 = time.perf_counter()
                mutate_structural(c2, batches, path_lib)
                dt = time.perf_counter() - t0
                stats.mutation_time += dt
                stats.mutation_calls += 1
                gen_mutation_time += dt
                gen_mutation_calls += 1

            # evaluation (time each call)
            t0 = time.perf_counter()
            evaluate_individual(c1, batches, path_lib, arcs, tt_dict)
            dt1 = time.perf_counter() - t0
            stats.evaluation_time += dt1
            stats.evaluation_calls += 1
            gen_eval_time += dt1
            gen_eval_calls += 1
            if dt1 > stats.best_crossover_time:
                stats.eval_slower_than_best_crossover += 1

            t0 = time.perf_counter()
            evaluate_individual(c2, batches, path_lib, arcs, tt_dict)
            dt2 = time.perf_counter() - t0
            stats.evaluation_time += dt2
            stats.evaluation_calls += 1
            gen_eval_time += dt2
            gen_eval_calls += 1
            if dt2 > stats.best_crossover_time:
                stats.eval_slower_than_best_crossover += 1

            offspring.append(c1)
            offspring.append(c2)

        # environmental selection
        combined = population + offspring

        t0 = time.perf_counter()
        fronts2 = non_dominated_sort(combined)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        new_pop: List[Individual] = []
        for f in fronts2:
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend(f)
            else:
                t0 = time.perf_counter()
                d = crowding_distance(f)
                dt = time.perf_counter() - t0
                stats.crowd_time += dt
                stats.crowd_calls += 1
                f.sort(key=lambda x: d[x], reverse=True)
                new_pop.extend(f[:pop_size - len(new_pop)])
                break
        population = new_pop

        gen_total = time.perf_counter() - t_gen0
        stats.add_gen_record({
            "gen": gen,
            "front0_unique": int(len(front0_unique)),
            "feasible_pop": int(sum(1 for ind in population if ind.feasible)),
            "gen_total_time_s": float(gen_total),
            "gen_crossover_time_s": float(gen_crossover_time),
            "gen_mutation_time_s": float(gen_mutation_time),
            "gen_evaluation_time_s": float(gen_eval_time),
            "gen_eval_calls": int(gen_eval_calls),
            "gen_crossover_calls": int(gen_crossover_calls),
            "gen_mutation_calls": int(gen_mutation_calls),
        })

    # final pareto (prefer feasible)
    final_fronts = non_dominated_sort(population)
    front0 = final_fronts[0]
    feasible_front0 = [ind for ind in front0 if ind.feasible]
    base_front = feasible_front0 if feasible_front0 else front0
    pareto_front = unique_individuals_by_objectives(base_front, tol=1e-3)

    # collect final feasible ND objs for P*
    final_feasible = [ind for ind in pareto_front if ind.feasible]
    final_feasible_nd_objs = [ind.objectives for ind in (final_feasible if final_feasible else pareto_front)]
    final_feasible_nd_objs = pareto_filter_objectives(final_feasible_nd_objs, tol=1e-6)

    stats.run_total_time = time.perf_counter() - t_run0

    # print timing summary for this run
    print("\n========== TIMING SUMMARY (this run) ==========")
    s = stats.summary_dict()
    for k, v in s.items():
        print(f"{k}: {v}")
    print(f"Final Pareto size (unique, prefer feasible): {len(pareto_front)}")
    print("=============================================\n")

    return population, pareto_front, batches, history_fronts, stats, final_feasible_nd_objs, all_feasible_front0_objs


# ========================
# 8. Output & plots
# ========================

def save_pareto_solutions(pareto: List[Individual],
                          batches: List[Batch],
                          filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-3)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (Unique, Feasible-first) =====\n\n")
        for i, ind in enumerate(pareto):
            cost, emit, time_ = ind.objectives
            f.write(f"===== Solution {i} =====\n")
            f.write(f"Objectives: Cost={cost:.6f}, Emission={emit:.6f}, Time={time_:.6f}, Penalty={ind.penalty:.6f}\n")
            f.write(f"Feasible={ind.feasible}\n\n")
            for batch in batches:
                key = (batch.origin, batch.destination, batch.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if allocs:
                    f.write(f"Batch {batch.batch_id}: {batch.origin}->{batch.destination} Q={batch.quantity}\n")
                    for a in allocs:
                        f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"[Saved] {len(pareto)} solutions to {filename}")


def plot_pareto_evolution_3d(history_fronts):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    num_gens = len(history_fronts)
    colors = cm.viridis(np.linspace(0, 1, max(1, num_gens)))

    for gen, objs in history_fronts:
        if not objs:
            continue
        xs = [o[0] for o in objs]
        ys = [o[1] for o in objs]
        zs = [o[2] for o in objs]

        alpha = 0.2 + 0.8 * (gen / max(1, num_gens - 1))
        s = 20 + 30 * (gen / max(1, num_gens - 1))

        label = f"Gen {gen}" if gen in [0, num_gens - 1] else ""
        ax1.scatter(xs, ys, zs, color=colors[gen], alpha=alpha, s=s, label=label)

    ax1.set_xlabel('Cost')
    ax1.set_ylabel('Emission')
    ax1.set_zlabel('Time')
    ax1.set_title('Pareto Front Evolution (Color=Generation)')
    ax1.legend()
    plt.tight_layout()
    plt.savefig("pareto_evolution_3d.png", dpi=300)
    plt.close()
    print("[Saved] pareto_evolution_3d.png")


# ========================
# 8.1 NEW: line plots (NO shadow) for HV/IGD+/SP
# ========================

def _sanitize_series(y: List[float], fallback: float = 0.0) -> np.ndarray:
    """
    Make a series plottable:
    - convert inf/nan -> nan
    - forward-fill nan
    - back-fill leading nan
    - if all nan -> fallback
    """
    arr = np.array(y, dtype=float)
    arr[~np.isfinite(arr)] = np.nan

    if np.all(np.isnan(arr)):
        return np.full_like(arr, fallback, dtype=float)

    # forward fill
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            if i > 0:
                arr[i] = arr[i - 1]

    # back fill leading nan (if first entries were nan)
    if np.isnan(arr[0]):
        first_valid = None
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                first_valid = i
                break
        if first_valid is None:
            return np.full_like(arr, fallback, dtype=float)
        arr[:first_valid] = arr[first_valid]

    # final cleanup
    arr = np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)
    return arr


def plot_single_run_line(series: List[float], title: str, ylabel: str, filename: str,
                         marker_every: int = 1):
    y = _sanitize_series(series, fallback=0.0)
    gens = np.arange(len(y))

    # 自动减少点的密度，避免太密
    if marker_every <= 0:
        marker_every = 1
    auto_me = max(1, len(gens) // 50)  # 大约最多 50 个marker
    me = max(marker_every, auto_me)

    plt.figure(figsize=(8, 5))
    plt.plot(gens, y, linewidth=2, marker='o', markevery=me)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Saved] {filename}")


def plot_mean_convergence_line(all_run_metrics: List[List[float]],
                               title: str, ylabel: str, filename: str):
    """
    all_run_metrics: list of runs, each is a list over gens.
    Produces mean line (with markers), NO shadow.
    """
    if not all_run_metrics:
        print(f"[WARN] Empty metrics for {title}, skip.")
        return

    clean_runs = []
    for run in all_run_metrics:
        clean_runs.append(_sanitize_series(run, fallback=0.0))

    data = np.vstack(clean_runs)  # (runs, gens)
    mean_curve = np.mean(data, axis=0)
    gens = np.arange(len(mean_curve))

    me = max(1, len(gens) // 50)  # ~50 markers
    plt.figure(figsize=(8, 5))
    plt.plot(gens, mean_curve, linewidth=2, marker='o', markevery=me, label='Mean')
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Saved] {filename}")


# ========================
# 9. Main: multi-run experiment with P* and metrics
# ========================

if __name__ == "__main__":
    filename = "data.xlsx"
    pop_size = 60
    generations = 400
    runs = 30

    print(f"\nStarting Experiment: Runs={runs}, Gen={generations}, Pop={pop_size}\n")

    # Cache runs
    cache_runs = []

    # For building global P*
    all_objs_for_Pstar: List[Tuple[float, float, float]] = []

    # Timing summaries across runs
    all_run_summaries: List[Dict] = []

    t_all0 = time.perf_counter()

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id}/{runs-1}, seed={seed} ==========\n")

        pop, pareto, batches, h_fronts, stats, final_feas_nd, feas_front0_allgens = run_nsga2_analytics(
            filename=filename,
            pop_size=pop_size,
            generations=generations,
            log_every=10
        )

        # Build P* candidates:
        all_objs_for_Pstar.extend(final_feas_nd)
        all_objs_for_Pstar.extend(pareto_filter_objectives(feas_front0_allgens, tol=1e-6))

        # Store run artifacts
        cache_runs.append({
            "run_id": run_id,
            "seed": seed,
            "population": pop,
            "pareto": pareto,
            "batches": batches,
            "history_fronts": h_fronts,
            "stats": stats
        })

        # Per-run summary
        run_summary = stats.summary_dict()
        run_summary["run_id"] = run_id
        run_summary["seed"] = seed
        run_summary["final_pareto_size"] = int(len(pareto))
        run_summary["final_feasible_pareto_size"] = int(sum(1 for ind in pareto if ind.feasible))
        run_summary["eval_over_crossover_ratio"] = float(
            (stats.evaluation_time / stats.crossover_time) if stats.crossover_time > 1e-12 else float("inf")
        )
        all_run_summaries.append(run_summary)

        print(f"[RUN {run_id}] Pareto size={len(pareto)} | "
              f"Feasible Pareto={run_summary['final_feasible_pareto_size']} | "
              f"Time total={stats.run_total_time:.2f}s | "
              f"eval={stats.evaluation_time:.2f}s | cross={stats.crossover_time:.2f}s")

    # Build global P*
    Pstar = build_reference_front_Pstar(all_objs_for_Pstar)
    print(f"\n[Global P*] Size: {len(Pstar)}")

    # Compute per-generation metrics for each run w.r.t. P*
    all_runs_igds = []
    all_runs_sps = []
    all_runs_hvs = []

    for item in cache_runs:
        h_fronts = item["history_fronts"]
        _, igds, sps, hvs = compute_gen_metrics_wrt_Pstar(h_fronts, Pstar, hv_norm_samples=20000)

        all_runs_igds.append(igds)
        all_runs_sps.append(sps)
        all_runs_hvs.append(hvs)

        # attach final metrics for best-run selection
        item["final_igd_plus"] = float(igds[-1]) if igds else float("inf")
        item["final_spacing"] = float(sps[-1]) if sps else float("inf")
        item["final_hv_norm"] = float(hvs[-1]) if hvs else 0.0

    # Select best run by FINAL HV_norm
    best_item = max(cache_runs, key=lambda x: x.get("final_hv_norm", 0.0))
    best_run_idx = best_item["run_id"]
    print(f"\n[Best Run Selection] by final HV_norm: Run={best_run_idx}, HV_norm={best_item['final_hv_norm']:.4f}, "
          f"IGD+={best_item['final_igd_plus']:.4f}, SP={best_item['final_spacing']:.4f}")

    # =========================
    # NEW: mean line plots (折线图, 带点) across 30 runs
    # =========================
    plot_mean_convergence_line(
        all_runs_hvs,
        title=f"HV_norm (Mean over {runs} runs)",
        ylabel="Hypervolume (Normalised)",
        filename="hv_norm_mean_line.png"
    )
    plot_mean_convergence_line(
        all_runs_igds,
        title=f"IGD+ (Mean over {runs} runs)",
        ylabel="IGD+ (w.r.t. P*)",
        filename="igd_plus_mean_line.png"
    )
    plot_mean_convergence_line(
        all_runs_sps,
        title=f"Spacing (Mean over {runs} runs)",
        ylabel="Spacing (SP)",
        filename="spacing_mean_line.png"
    )

    # =========================
    # NEW: best-run line plots (折线图, 带点)
    # =========================
    best_h_fronts = best_item["history_fronts"]
    _, best_igds, best_sps, best_hvs = compute_gen_metrics_wrt_Pstar(best_h_fronts, Pstar, hv_norm_samples=20000)

    plot_single_run_line(
        best_hvs,
        title="HV_norm_Pstar over generations",
        ylabel="HV_norm_Pstar",
        filename="best_hv_norm_line.png"
    )
    plot_single_run_line(
        best_igds,
        title="IGD+ over generations",
        ylabel="IGD+ (w.r.t. P*)",
        filename="best_igd_plus_line.png"
    )
    plot_single_run_line(
        best_sps,
        title="Spacing over generations",
        ylabel="Spacing (SP)",
        filename="best_spacing_line.png"
    )

    # Best run visualisations and outputs
    best_pareto = best_item["pareto"]
    best_batches = best_item["batches"]

    plot_pareto_evolution_3d(best_h_fronts)
    save_pareto_solutions(best_pareto, best_batches, filename="result.txt")

    # Save timing summary CSV
    df_sum = pd.DataFrame(all_run_summaries)
    df_sum.to_csv("timing_summary_runs.csv", index=False)
    print("[Saved] timing_summary_runs.csv")

    total_all_time = time.perf_counter() - t_all0
    print(f"\n[TOTAL] All runs wall time = {total_all_time:.2f} seconds")
    print("\nAll done.\n")
