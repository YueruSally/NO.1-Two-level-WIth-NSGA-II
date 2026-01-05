#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
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

# ---- Penalty weights (soft constraints)
PEN_LATE_PER_H = 1e6          # lateness hours -> penalty (soft)
PEN_MISS_TT = 5e7             # missing timetable (per arc) -> penalty
PEN_MISS_ALLOC = 1e9          # missing batch allocation -> penalty
PEN_CAP_EXCESS_PER_TEU = 5e7  # capacity excess (TEU) -> penalty

# ---- Mutation roulette weights
W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20

OPS = ["add", "del", "mod", "mode"]


# ========================
# 1. Data structures
# ========================

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float       # TEU / TimeBucket (e.g., TEU/hour)
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
        return hash((self.path, round(self.share, 10)))

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            mode = self.path.modes[i]
            chain += f"{node}--({mode})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    # (origin, dest, batch_id)
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    penalty: float = 0.0
    feasible: bool = False

    violation: float = 0.0
    vio_breakdown: Dict[str, float] = field(default_factory=dict)


# ========================
# 2. Utilities: merge & normalize shares
# ========================

def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []

    merged_map: Dict[Path, float] = {}
    for a in allocs:
        merged_map[a.path] = merged_map.get(a.path, 0.0) + a.share

    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged_map.items()]

    total_share = sum(a.share for a in unique_allocs)
    if total_share <= 1e-9:
        avg = 1.0 / len(unique_allocs)
        for a in unique_allocs:
            a.share = avg
    else:
        factor = 1.0 / total_share
        for a in unique_allocs:
            a.share *= factor

    # drop tiny shares
    unique_allocs = [a for a in unique_allocs if a.share > 0.001]

    if unique_allocs:
        final_total = sum(a.share for a in unique_allocs)
        if abs(final_total - 1.0) > 1e-6:
            for a in unique_allocs:
                a.share /= final_total

    return unique_allocs


# ========================
# 3. Data loading & graph
# ========================

def load_network_from_extended(filename: str):
    xls = pd.ExcelFile(filename)

    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).tolist()
    node_region = dict(zip(nodes_df["EnglishName"].astype(str),
                           nodes_df["Region"].astype(str)))

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

    tdf = pd.read_excel(xls, "Timetable")
    timetables: List[TimetableEntry] = []
    for _, row in tdf.iterrows():
        freq = float(row["Frequency_per_week"])
        hd = row.get("Headway_Hours", np.nan)
        hd = 168.0 / max(freq, 1.0) if pd.isna(hd) else float(hd)

        v = row.get("FirstDepartureHour", np.nan)
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


def build_arc_lookup(arcs: List[Arc]) -> Dict[Tuple[str, str, str], Arc]:
    """
    For mode mutation / repair: quick lookup (u,v,mode)->Arc.
    If duplicates exist, take the first.
    """
    mp: Dict[Tuple[str, str, str], Arc] = {}
    for a in arcs:
        k = (a.from_node, a.to_node, a.mode)
        if k not in mp:
            mp[k] = a
    return mp


# ========================
# 4. Path library (random DFS) + TT repair by ROAD fallback
# ========================

def random_dfs_paths(graph, origin, dest, max_len=8, max_paths=120) -> List[List[Arc]]:
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
    """
    If a non-road arc has no timetable, replace it with road arc (u,v,road) if exists.
    Return repaired arc list or None if cannot repair.
    """
    new_seq: List[Arc] = []
    for arc in arc_seq:
        if arc.mode == "road":
            new_seq.append(arc)
            continue

        key_tt = (arc.from_node, arc.to_node, arc.mode)
        if tt_dict.get(key_tt, []):
            new_seq.append(arc)
            continue

        # no timetable -> try road fallback
        k_road = (arc.from_node, arc.to_node, "road")
        if k_road in arc_lookup:
            new_seq.append(arc_lookup[k_road])
        else:
            return None
    return new_seq


def build_path_library(node_names, arcs, batches, tt_dict, arc_lookup) -> Dict[Tuple[str, str], List[Path]]:
    graph = build_graph(arcs)
    path_lib: Dict[Tuple[str, str], List[Path]] = {}
    next_path_id = 0

    for batch in batches:
        od = (batch.origin, batch.destination)
        if od in path_lib:
            continue

        arc_paths = random_dfs_paths(graph, batch.origin, batch.destination, max_len=8, max_paths=120)
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
                origin=batch.origin,
                destination=batch.destination,
                nodes=nodes,
                modes=modes,
                arcs=repaired,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in repaired),
                base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in repaired),
                base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in repaired),
            ))
            next_path_id += 1

        if paths_for_od:
            paths_for_od.sort(key=lambda p: p.base_cost_per_teu)
            path_lib[od] = paths_for_od[:30]  # keep more -> helps ADD

    return path_lib


def sanity_check_path_lib(batches: List[Batch], path_lib: Dict[Tuple[str, str], List[Path]]):
    bad = []
    sizes = []
    for b in batches:
        od = (b.origin, b.destination)
        n = len(path_lib.get(od, []))
        sizes.append(n)
        if n == 0:
            bad.append(b)

    if bad:
        print("\n[SANITY CHECK] ❌ Some batches have ZERO path in path_lib:")
        for b in bad:
            print(f"  Batch {b.batch_id}: {b.origin}->{b.destination} | paths=0")
        print("=> 这表示：当前网络+时刻表+road回退修复后，仍无可行路径。")
    else:
        print("\n[SANITY CHECK] ✅ All batches have at least one usable path in path_lib.")

    if sizes:
        print(f"[PathLib] size stats: min={min(sizes)}, mean={sum(sizes)/len(sizes):.2f}, max={max(sizes)}")


# ========================
# 5. Simulation & evaluation
# ========================

def simulate_path_time_capacity(
    path: Path,
    batch: Batch,
    flow: float,
    tt_dict,
    arc_flow_map,
) -> Tuple[float, float, int]:
    """
    Returns:
      travel_time_h (relative to ET),
      total_wait_h,
      miss_tt_count
    """
    t = batch.ET
    total_wait = 0.0
    miss_tt = 0

    for arc in path.arcs:
        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

        if arc.mode == "road":
            entries = []
        else:
            key = (arc.from_node, arc.to_node, arc.mode)
            entries = tt_dict.get(key, [])

        # non-road without timetable -> unusable
        if (arc.mode != "road") and (not entries):
            miss_tt += 1
            return float("inf"), float("inf"), miss_tt

        if not entries:
            dep = t
        else:
            e = entries[0]
            if t <= e.first_departure_hour:
                dep = e.first_departure_hour
            else:
                waited = (t - e.first_departure_hour)
                n = math.ceil(waited / max(e.headway_hours, 1e-6))
                dep = e.first_departure_hour + n * e.headway_hours

        wait_here = max(0.0, dep - t)
        total_wait += wait_here

        arr = dep + travel_time

        start_slot = int(dep)
        key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0.0) + flow

        t = arr

    return t - batch.ET, total_wait, miss_tt


def evaluate_individual(ind: Individual, batches, arcs, tt_dict):
    total_cost = 0.0
    total_emission = 0.0
    makespan = 0.0

    arc_flow_map: Dict[Tuple[Tuple[str, str, str], int], float] = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    vio_miss_alloc = 0
    vio_miss_tt = 0
    vio_late_h = 0.0
    vio_cap_excess = 0.0

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])

        if not allocs:
            vio_miss_alloc += 1
            continue

        batch_finish_time = batch.ET

        for alloc in allocs:
            if alloc.share <= 1e-9:
                continue

            flow = alloc.share * batch.quantity
            path = alloc.path

            # base components
            total_cost += path.base_cost_per_teu * flow
            total_emission += path.base_emission_per_teu * flow

            travel_time, wait_h, miss_tt = simulate_path_time_capacity(
                path, batch, flow, tt_dict, arc_flow_map
            )

            if math.isinf(travel_time):
                # no timetable -> violation
                vio_miss_tt += miss_tt
                # keep a small late marker so "violation" reflects badness
                vio_late_h += 1.0
                continue

            arrival_time = batch.ET + travel_time
            batch_finish_time = max(batch_finish_time, arrival_time)

            # lateness measured in hours (positive part)
            if arrival_time > batch.LT:
                vio_late_h += (arrival_time - batch.LT)

        makespan = max(makespan, batch_finish_time)

    # capacity check
    for (key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(key, 1e9)
        if flow > cap:
            vio_cap_excess += (flow - cap)

    penalty = (
        PEN_MISS_ALLOC * float(vio_miss_alloc)
        + PEN_MISS_TT * float(vio_miss_tt)
        + PEN_LATE_PER_H * float(vio_late_h)
        + PEN_CAP_EXCESS_PER_TEU * float(vio_cap_excess)
    )

    # objectives: cost & emission carry penalty for ranking; time kept separate
    ind.objectives = (total_cost + penalty,
                      total_emission + penalty,
                      makespan)
    ind.penalty = penalty

    # ===========================
    # ✅ SYNC WITH PROGRAM A:
    # Late IS a feasibility condition now.
    # (A: late_violation makes feasible=False)
    # ===========================
    ind.feasible = (
        vio_miss_alloc == 0
        and vio_miss_tt == 0
        and vio_cap_excess <= 1e-9
        and vio_late_h <= 1e-9
    )

    ind.violation = float(vio_miss_alloc) + float(vio_miss_tt) + float(vio_late_h) + float(vio_cap_excess)
    ind.vio_breakdown = {
        "miss_alloc": float(vio_miss_alloc),
        "miss_tt": float(vio_miss_tt),
        "late_h": float(vio_late_h),
        "cap_excess": float(vio_cap_excess),
    }


# ========================
# 6. Genetic operators
# ========================

def are_individuals_identical(genes1, genes2) -> bool:
    if len(genes1) != len(genes2):
        return False
    return all(g1 == g2 for g1, g2 in zip(genes1, genes2))


def find_common_intermediate_nodes(path1_nodes, path2_nodes):
    if not path1_nodes or not path2_nodes:
        return []
    intermediate1 = set(path1_nodes[1:-1])
    intermediate2 = set(path2_nodes[1:-1])
    return list(intermediate1.intersection(intermediate2))


def find_allocation_index(allocations: List[PathAllocation], target: PathAllocation) -> Optional[int]:
    for i, alloc in enumerate(allocations):
        if alloc == target:
            return i
    return None


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


def perform_single_point_crossover_arcwise(alloc1: PathAllocation, alloc2: PathAllocation, common_node: str):
    p1 = alloc1.path
    p2 = alloc2.path

    if not p1.arcs or not p2.arcs:
        return None, None

    i1 = p1.nodes.index(common_node)
    i2 = p2.nodes.index(common_node)

    new_arcs1 = p1.arcs[:i1] + p2.arcs[i2:]
    new_arcs2 = p2.arcs[:i2] + p1.arcs[i1:]

    new_p1 = path_from_arcs(new_arcs1, p1.origin, p1.destination, path_id=-1)
    new_p2 = path_from_arcs(new_arcs2, p2.origin, p2.destination, path_id=-1)

    if new_p1 is None or new_p2 is None:
        return None, None

    return PathAllocation(path=new_p1, share=alloc1.share), PathAllocation(path=new_p2, share=alloc2.share)


# ---------------------------
# ✅ SYNC WITH PROGRAM A:
# Use A-style "crossover_structural" as the ACTIVE crossover operator.
# ---------------------------

def clone_gene(alloc: PathAllocation) -> PathAllocation:
    return PathAllocation(path=alloc.path, share=alloc.share)


def crossover_structural(ind1: Individual, ind2: Individual,
                         batches: List[Batch]) -> Tuple[Individual, Individual]:
    """
    Program-A style crossover:
    - For each batch key (O,D,batch_id), do list-fragment crossover on PathAllocation list
    - DO NOT rebuild a new path (no arcwise stitching)
    - Repair shares with merge_and_normalize
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


def crossover_complex(ind1: Individual, ind2: Individual, batches: List[Batch]) -> Tuple[Individual, Individual]:
    """
    (Kept for reference, but no longer used in the main loop after syncing to Program A.)
    """
    child1 = deepcopy(ind1)
    child2 = deepcopy(ind2)

    # reset evaluation fields
    child1.objectives = (float("inf"), float("inf"), float("inf"))
    child2.objectives = (float("inf"), float("inf"), float("inf"))
    child1.penalty = 0.0; child2.penalty = 0.0
    child1.feasible = False; child2.feasible = False
    child1.violation = 0.0; child2.violation = 0.0
    child1.vio_breakdown = {}; child2.vio_breakdown = {}

    batch_keys = [(b.origin, b.destination, b.batch_id) for b in batches]
    random.shuffle(batch_keys)

    crossed = False

    for key in batch_keys:
        allocs1 = child1.od_allocations.get(key, [])
        allocs2 = child2.od_allocations.get(key, [])
        if not allocs1 or not allocs2:
            continue
        if are_individuals_identical(allocs1, allocs2):
            continue

        candidates = []
        for a1 in allocs1:
            for a2 in allocs2:
                common_nodes = find_common_intermediate_nodes(a1.path.nodes, a2.path.nodes)
                for cn in common_nodes:
                    candidates.append((a1, a2, cn))

        if not candidates:
            continue

        tries = 0
        new_a1 = None
        new_a2 = None
        while tries < 10 and (new_a1 is None or new_a2 is None):
            a1, a2, cn = random.choice(candidates)
            new_a1, new_a2 = perform_single_point_crossover_arcwise(a1, a2, cn)
            tries += 1

        if new_a1 is None or new_a2 is None:
            continue

        i1 = find_allocation_index(allocs1, a1)
        i2 = find_allocation_index(allocs2, a2)
        if i1 is None or i2 is None:
            continue

        new_allocs1 = deepcopy(allocs1)
        new_allocs2 = deepcopy(allocs2)
        new_allocs1[i1] = new_a1
        new_allocs2[i2] = new_a2

        child1.od_allocations[key] = merge_and_normalize(new_allocs1)
        child2.od_allocations[key] = merge_and_normalize(new_allocs2)

        crossed = True
        break

    if not crossed and batch_keys:
        k = random.choice(batch_keys)
        if k in child1.od_allocations and k in child2.od_allocations:
            child1.od_allocations[k], child2.od_allocations[k] = \
                child2.od_allocations[k], child1.od_allocations[k]

    return child1, child2


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


def mutate_add(ind: Individual, batch: Batch, path_lib):
    key = (batch.origin, batch.destination, batch.batch_id)
    od = (batch.origin, batch.destination)
    allocs = ind.od_allocations.get(key, [])
    paths_in_lib = path_lib.get(od, [])
    if not paths_in_lib:
        return False

    current_structures = {a.path for a in allocs}
    candidates = [p for p in paths_in_lib if p not in current_structures]

    if not candidates:
        # OPTIONAL fallback: replace one allocation with another lib path
        if allocs:
            repl = random.choice(paths_in_lib)
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
    allocs.pop(random.randint(0, len(allocs) - 1))
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


def mutate_mode(ind: Individual, batch: Batch, tt_dict, arc_lookup, max_trials: int = 20):
    key = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs:
        return False

    alloc_idx = random.randrange(len(allocs))
    old_alloc = allocs[alloc_idx]
    p = old_alloc.path
    if not p.arcs:
        return False

    arc_i = random.randrange(len(p.arcs))
    old_arc = p.arcs[arc_i]
    u, v = old_arc.from_node, old_arc.to_node

    modes_all = ["road", "rail", "water"]
    modes_all = [m for m in modes_all if m != old_arc.mode]
    if not modes_all:
        return False

    for _ in range(max_trials):
        new_mode = random.choice(modes_all)

        k_arc = (u, v, new_mode)
        if k_arc not in arc_lookup:
            continue

        if new_mode != "road":
            k_tt = (u, v, new_mode)
            if not tt_dict.get(k_tt, []):
                continue

        new_arcs = list(p.arcs)
        new_arcs[arc_i] = arc_lookup[k_arc]
        new_path = path_from_arcs(new_arcs, p.origin, p.destination, path_id=-1)
        if new_path is None:
            continue

        allocs_new = deepcopy(allocs)
        allocs_new[alloc_idx] = PathAllocation(path=new_path, share=old_alloc.share)
        ind.od_allocations[key] = merge_and_normalize(allocs_new)
        return True

    return False


def mutate_roulette(ind: Individual, batches: List[Batch], path_lib, tt_dict, arc_lookup):
    batch = random.choice(batches)

    r = random.random()
    total = W_ADD + W_DEL + W_MOD + W_MODE
    r *= total

    if r < W_ADD:
        ok = mutate_add(ind, batch, path_lib)
        return "add", ok
    r -= W_ADD
    if r < W_DEL:
        ok = mutate_del(ind, batch)
        return "del", ok
    r -= W_DEL
    if r < W_MOD:
        ok = mutate_mod(ind, batch)
        return "mod", ok
    ok = mutate_mode(ind, batch, tt_dict, arc_lookup)
    return "mode", ok


# ========================
# 7. NSGA-II components
# ========================

def dominates(a: Individual, b: Individual) -> bool:
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


class HypervolumeCalculator:
    """
    Monte-Carlo HV in [0,1] under a fixed reference point.
    """
    def __init__(self, ref_point: Tuple[float, float, float], num_samples=10000):
        self.ref_point = np.array(ref_point, dtype=float)
        self.num_samples = num_samples
        self.ideal_point = np.zeros(3, dtype=float)
        self.samples = np.random.uniform(
            low=self.ideal_point,
            high=self.ref_point,
            size=(self.num_samples, 3)
        )
        self.total_volume = float(np.prod(self.ref_point))

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
        is_dominated = np.all(O <= S, axis=2)
        dominated_samples = np.any(is_dominated, axis=1)
        ratio = np.sum(dominated_samples) / float(self.num_samples)
        return ratio


def unique_objective_tuples(objs: List[Tuple[float, float, float]], tol: float = 1e-6) -> List[Tuple[float, float, float]]:
    out = []
    for o in objs:
        dup = False
        for p in out:
            if all(abs(o[i] - p[i]) <= tol for i in range(3)):
                dup = True
                break
        if not dup:
            out.append(o)
    return out


def unique_individuals_by_objectives(front: List[Individual], tol: float = 1e-3) -> List[Individual]:
    unique: List[Individual] = []
    seen: List[Tuple[float, float, float]] = []
    for ind in front:
        obj = ind.objectives
        is_dup = False
        for o in seen:
            if (abs(obj[0] - o[0]) <= tol and
                abs(obj[1] - o[1]) <= tol and
                abs(obj[2] - o[2]) <= tol):
                is_dup = True
                break
        if not is_dup:
            seen.append(obj)
            unique.append(ind)
    return unique


# ========================
# 8. Metrics: P*, IGD+, Spacing
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
    """
    IGD+ for minimization.
    d+(p, a) = sqrt(sum(max(a_i - p_i, 0)^2)).
    """
    if not P_star or not A:
        return float("inf")
    P = np.array(P_star, dtype=float)
    Q = np.array(A, dtype=float)

    # for each p, compute min over q of d+(p,q)
    dists = []
    for p in P:
        diff = Q - p  # (|A|,3)
        diff = np.maximum(diff, 0.0)
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dists.append(float(np.min(dist)))
    return float(np.mean(dists))


def spacing_metric(A: List[Tuple[float, float, float]]) -> float:
    """
    Spacing: lower is better (more uniform).
    Use nearest-neighbor distances in objective space.
    """
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
# 9. NSGA-II runner (one run)
# ========================

def run_nsga2_analytics(filename="data.xlsx", pop_size=60, generations=200):
    print("Loading data...")
    node_names, arcs, timetables, batches = load_network_from_extended(filename)

    print("Pre-processing timetable...")
    tt_dict = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)

    print("Building path library...")
    path_lib = build_path_library(node_names, arcs, batches, tt_dict, arc_lookup)
    sanity_check_path_lib(batches, path_lib)

    # initial population
    population: List[Individual] = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict)
        population.append(ind)

    # HV reference point from initial pop (per-run)
    all_objs = np.array([ind.objectives for ind in population], dtype=float)
    max_vals = np.max(all_objs, axis=0)
    ref_point = max_vals * 1.2
    hv_calc = HypervolumeCalculator(ref_point=tuple(ref_point), num_samples=10000)
    print(f"\nReference Point for HV set to: {ref_point}")

    # histories
    front_hist_objs: List[List[Tuple[float, float, float]]] = []
    hv_history: List[float] = []

    feasible_ratio_hist: List[float] = []
    vio_mean_hist: Dict[str, List[float]] = {k: [] for k in ["miss_alloc", "miss_tt", "late_h", "cap_excess"]}

    # mutation tracker per generation (attempt/success)
    mut_tracker = {
        "attempt": {op: [0] * generations for op in OPS},
        "success": {op: [0] * generations for op in OPS},
    }

    for gen in range(generations):
        fronts = non_dominated_sort(population)
        front0 = fronts[0]

        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        current_front_objs = [ind.objectives for ind in front0_unique]
        front_hist_objs.append(current_front_objs)

        current_hv = hv_calc.calculate(front0_unique)
        hv_history.append(current_hv)

        # feasible ratio (in population)
        feas_ratio = sum(1 for ind in population if ind.feasible) / float(len(population))
        feasible_ratio_hist.append(feas_ratio)

        # violation mean (in population)
        for k in vio_mean_hist.keys():
            vals = [ind.vio_breakdown.get(k, 0.0) for ind in population]
            vio_mean_hist[k].append(float(np.mean(vals)) if vals else 0.0)

        feas_n = len(feasible_front0)
        best_cost = min(ind.objectives[0] for ind in front0_unique) if front0_unique else float("inf")
        best_vio = min(ind.violation for ind in front0_unique) if front0_unique else float("inf")

        print(f"Gen {gen:03d} | Front0={len(front0):2d} | FeasFront0={feas_n:2d} | "
              f"FeasRatio={feas_ratio:.2%} | BestCost={best_cost:.2f} | BestViolation={best_vio:.6f} | HV={current_hv:.4f}")

        if front0_unique:
            best_ind = min(front0_unique, key=lambda x: x.violation)
            bd = best_ind.vio_breakdown
            print(f"      breakdown(best-by-violation): late_h={bd.get('late_h',0):.3f}, cap={bd.get('cap_excess',0):.3f}, "
                  f"miss_tt={bd.get('miss_tt',0):.0f}, miss_alloc={bd.get('miss_alloc',0):.0f}")

        # ranks & crowding
        ranks: Dict[Individual, int] = {}
        for r, front in enumerate(fronts):
            for ind in front:
                ranks[ind] = r

        dists: Dict[Individual, float] = {}
        for front in fronts:
            dists.update(crowding_distance(front))

        # selection
        mating_pool: List[Individual] = []
        while len(mating_pool) < pop_size:
            mating_pool.append(tournament_select(population, dists, ranks))

        # crossover + mutation
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)

            if random.random() < 0.7:
                # ===========================
                # ✅ SYNC WITH PROGRAM A:
                # Use crossover_structural instead of crossover_complex
                # ===========================
                c1, c2 = crossover_structural(p1, p2, batches)
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)

            # roulette mutation + tracking
            if random.random() < 0.35:
                op, ok = mutate_roulette(c1, batches, path_lib, tt_dict, arc_lookup)
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if ok else 0)

            if random.random() < 0.35:
                op, ok = mutate_roulette(c2, batches, path_lib, tt_dict, arc_lookup)
                mut_tracker["attempt"][op][gen] += 1
                mut_tracker["success"][op][gen] += (1 if ok else 0)

            evaluate_individual(c1, batches, arcs, tt_dict)
            evaluate_individual(c2, batches, arcs, tt_dict)

            offspring.append(c1)
            offspring.append(c2)

        # environmental selection
        combined = population + offspring
        fronts = non_dominated_sort(combined)

        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                d = crowding_distance(front)
                front.sort(key=lambda x: d[x], reverse=True)
                new_pop.extend(front[:pop_size - len(new_pop)])
                break
        population = new_pop

    # final pareto
    final_fronts = non_dominated_sort(population)
    front0 = final_fronts[0]
    feasible_front0 = [ind for ind in front0 if ind.feasible]
    base_front = feasible_front0 if feasible_front0 else front0
    pareto_front = unique_individuals_by_objectives(base_front, tol=1e-3)

    return (population, pareto_front, batches,
            front_hist_objs, hv_history,
            feasible_ratio_hist, vio_mean_hist,
            mut_tracker)


# ========================
# 10. Output helpers
# ========================

def print_pure_structure(ind: Individual, batches: List[Batch], sol_name="Solution"):
    print(f"\n===== {sol_name} 最终结构 (Node+Mode | Share) =====")
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            continue
        print(f"\nBatch {batch.batch_id}: {batch.origin} -> {batch.destination}, Q={batch.quantity}\n")
        for a in allocs:
            print(a)


def save_pareto_solutions(pareto: List[Individual], batches: List[Batch], filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-3)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (All, Unique & Feasible-first) =====\n\n")
        for i, ind in enumerate(pareto):
            cost, emit, time_ = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Objectives: Cost={cost:.2f}, Emission={emit:.2f}, Time={time_:.2f}, "
                    f"Penalty={ind.penalty:.2f}, Feasible={ind.feasible}, "
                    f"Violation={ind.violation:.6f}, Breakdown={ind.vio_breakdown}\n\n")

            for batch in batches:
                key = (batch.origin, batch.destination, batch.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if not allocs:
                    continue
                f.write(f"Batch {batch.batch_id}: {batch.origin} -> {batch.destination}, Q={batch.quantity}\n\n")
                for a in allocs:
                    f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"Saved {len(pareto)} unique Pareto solutions to {filename}")


# ========================
# 11. Academic plots
# ========================

def aggregate_mutation_over_runs(mut_runs: List[dict], generations: int):
    # per run: attempt/success arrays -> compute share/rate/eff
    share_runs = []
    rate_runs = []
    eff_runs = []
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

    share_runs = np.stack(share_runs, axis=0)  # (R,4,G)
    rate_runs = np.stack(rate_runs, axis=0)
    eff_runs = np.stack(eff_runs, axis=0)

    return {
        "share_mean": np.mean(share_runs, axis=0),
        "share_std": np.std(share_runs, axis=0),
        "rate_mean": np.mean(rate_runs, axis=0),
        "rate_std": np.std(rate_runs, axis=0),
        "eff_mean": np.mean(eff_runs, axis=0),
        "eff_std": np.std(eff_runs, axis=0),
    }


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


def plot_mutation_success_rate_2x2(gen, rate_mean, rate_std, save="mutation_success_rate.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, op in enumerate(OPS):
        ax = axes[i]
        m = rate_mean[i]
        s = rate_std[i]
        ax.plot(gen, m, linewidth=2.5, label=f"{op} mean")
        ax.fill_between(gen, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        alpha=0.2, label="± std")
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
    """
    Stacked bar using mean violation components over runs.
    """
    miss_alloc = np.array(vio_mean_dict_mean["miss_alloc"])
    miss_tt = np.array(vio_mean_dict_mean["miss_tt"])
    late_h = np.array(vio_mean_dict_mean["late_h"])
    cap_excess = np.array(vio_mean_dict_mean["cap_excess"])

    plt.figure(figsize=(12, 4), dpi=300)
    plt.bar(gen, miss_alloc, label="miss_alloc")
    plt.bar(gen, miss_tt, bottom=miss_alloc, label="miss_tt")
    plt.bar(gen, late_h, bottom=miss_alloc + miss_tt, label="late_h")
    plt.bar(gen, cap_excess, bottom=miss_alloc + miss_tt + late_h, label="cap_excess")

    plt.xlabel("Generation")
    plt.ylabel("Mean Violation Components (stacked)")
    plt.title("Violation Breakdown (mean over runs)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(save)
    print("Saved:", save)


# ========================
# 12. Main (multi-runs) + P* + IGD+/Spacing
# ========================

if __name__ == "__main__":
    filename = "data.xlsx"
    pop_size = 60
    generations = 200
    runs = 30

    run_paretos = []
    run_front_hist = []
    run_hv_hist = []
    run_feasible_ratio = []
    run_vio_mean = []
    mut_runs = []

    best_run_idx = -1
    best_run_hv = -1.0
    best_pareto = None
    best_batches = None

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id} / {runs-1}, seed={seed} ==========\n")
        pop, pareto, batches, front_hist, hv_hist, feas_ratio_hist, vio_mean_hist, mut_tracker = run_nsga2_analytics(
            filename=filename,
            pop_size=pop_size,
            generations=generations
        )

        run_paretos.append(pareto)
        run_front_hist.append(front_hist)
        run_hv_hist.append(hv_hist)
        run_feasible_ratio.append(feas_ratio_hist)
        run_vio_mean.append(vio_mean_hist)
        mut_runs.append(mut_tracker)

        final_hv = hv_hist[-1] if hv_hist else 0.0
        print(f"[RUN {run_id}] Final HV = {final_hv:.4f}, Pareto size = {len(pareto)}")

        if final_hv > best_run_hv:
            best_run_hv = final_hv
            best_run_idx = run_id
            best_pareto = pareto
            best_batches = batches

    # ===== Build P* from all runs & all generations front histories
    all_points = []
    for r in range(runs):
        for gen_front in run_front_hist[r]:
            all_points.extend(gen_front)
    P_star = nondominated_set(all_points)
    print(f"\n[P*] Pseudo-reference front size = {len(P_star)}")

    # normalization range from P*
    P_arr = np.array(P_star, dtype=float)
    mins = np.min(P_arr, axis=0)
    maxs = np.max(P_arr, axis=0)

    # ===== Compute IGD+ and Spacing histories per run
    igd_runs = []
    sp_runs = []
    for r in range(runs):
        igd_hist = []
        sp_hist = []
        for gen_front in run_front_hist[r]:
            A = gen_front
            # normalize to reduce scale dominance
            Pn = normalize_points(P_star, mins, maxs)
            An = normalize_points(A, mins, maxs)

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
    vio_mean_dict_mean = {k: [0.0] * generations for k in ["miss_alloc", "miss_tt", "late_h", "cap_excess"]}
    for k in vio_mean_dict_mean.keys():
        mat = np.array([run_vio_mean[r][k] for r in range(runs)], dtype=float)  # (R,G)
        vio_mean_dict_mean[k] = list(np.mean(mat, axis=0))

    # mutation aggregation
    mut_agg = aggregate_mutation_over_runs(mut_runs, generations)

    # ===== Summary table (final gen)
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
    if best_pareto is not None:
        for i, ind in enumerate(best_pareto[:3]):
            print_pure_structure(ind, best_batches, f"BestRun Pareto Sol {i}")
        save_pareto_solutions(best_pareto, best_batches, filename="result.txt")

    # ===== Academic plots
    plot_mutation_attempt_stacked(gen, mut_agg["share_mean"], save="mutation_attempt_stacked.png")
    plot_mutation_success_rate_2x2(gen, mut_agg["rate_mean"], mut_agg["rate_std"], save="mutation_success_rate.png")
    plot_mutation_effective_contribution(gen, mut_agg["eff_mean"], save="mutation_effective_contribution.png")

    plot_convergence_curves(gen, hv_mean, hv_std, igd_mean, igd_std, save="convergence_hv_igd.png")
    plot_spacing_curve(gen, sp_mean, sp_std, save="diversity_spacing.png")
    plot_feasible_ratio(gen, fr_mean, fr_std, save="feasible_ratio.png")
    plot_violation_breakdown_stacked(gen, vio_mean_dict_mean, save="violation_breakdown.png")

    print("\nSaved figures:")
    print(" - mutation_attempt_stacked.png")
    print(" - mutation_success_rate.png")
    print(" - mutation_effective_contribution.png")
    print(" - convergence_hv_igd.png")
    print(" - diversity_spacing.png")
    print(" - feasible_ratio.png")
    print(" - violation_breakdown.png")
    print(" - result.txt")
