#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NSGA-II for multimodal routing with:
- Batch-level chromosome key = (origin, destination, batch_id)
- Time-dependent timetable waiting (rail/water)
- Node capacity queueing (service rate)
- Arc capacity queueing (service rate)
- Border/customs delay at border nodes
- Transshipment (mode change) cost & time at intermediate nodes
- Regional carbon policy heterogeneity:
    - emission factor by (region_group, mode)
    - carbon tax by region_group
    - applicability theta by (region_group, mode)

NEW (for your request):
(2) Waiting-related emissions added into objective f2:
    - timetable wait, arc wait, node wait, border time, transshipment time
    - arc service time, node service time (processing)
    - controlled by Waiting_Costs sheet column:
        WaitEmission_gCO2_per_TEU_h (or aliases)
(5) Roulette mutation behaviour tracking + plots:
    - per generation attempt/exec counts, shares, success rates
    - output CSV + stacked share plots + success-rate plots

NEW (this turn):
(6) Single waiting episode at a node > 48h => HARD infeasible (wait48)
    - checks applied to: timetable_wait, arc_wait, border_time, node_wait
    - recorded in ind.infeas_reason["wait48"] and ind.debug_info["batch_wait48_first"]
    - post-eval hard repair (only when wait48 triggered) to avoid the violated node if possible

Data file: data.xlsx
"""

import math
import random
import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ========================
# Global settings
# ========================

TIME_BUCKET_H = 1.0
CHINA_REGIONS = {"CN", "China"}
EUROPE_REGIONS = {"WE", "EE", "EU", "Europe"}
BIG_M = 1e6

# timetable safety
HEADWAY_CAP_H = 48.0

# NEW: single waiting episode hard cap (per node)
WAIT_EVENT_CAP_H = 48.0
WAIT48_HARD_PENALTY = 1e9
WAIT48_EPS = 1e-9

# mutation roulette weights
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
    emission_per_teu_km: float  # legacy field; evaluation uses regional emission factors
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

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            mode = self.path.modes[i]
            chain += f"{node}--({mode})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    penalty: float = 0.0
    feasible: bool = False
    infeas_reason: Dict[str, int] = field(default_factory=dict)
    debug_info: Dict = field(default_factory=dict)


# ========================
# 2. Share utilities
# ========================

def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []
    merged: Dict[Path, float] = {}
    for a in allocs:
        merged[a.path] = merged.get(a.path, 0.0) + a.share
    uniq = [PathAllocation(path=p, share=s) for p, s in merged.items()]

    total = sum(a.share for a in uniq)
    if total <= 1e-12:
        avg = 1.0 / len(uniq)
        for a in uniq:
            a.share = avg
    else:
        inv = 1.0 / total
        for a in uniq:
            a.share *= inv

    uniq = [a for a in uniq if a.share > 0.001]
    if uniq:
        total2 = sum(a.share for a in uniq)
        if total2 > 1e-12 and abs(total2 - 1.0) > 1e-6:
            for a in uniq:
                a.share /= total2
    return uniq


def clone_gene(a: PathAllocation) -> PathAllocation:
    return PathAllocation(path=a.path, share=a.share)


# ========================
# 3. Load data + parameters
# ========================

def _detect_node_capacity_columns(nodes_df: pd.DataFrame) -> List[str]:
    cols = [str(c) for c in nodes_df.columns]
    cand = []
    for c in cols:
        lc = c.lower()
        if ("teu" in lc) and (("cap" in lc) or ("capacity" in lc) or ("throughput" in lc)):
            cand.append(c)
    return cand


def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _parse_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    s = str(x)
    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "." or ch == "-"))
    if cleaned in ("", "-", "."):
        return float("nan")
    try:
        return float(cleaned)
    except Exception:
        return float("nan")


def load_network_from_extended(filename: str):
    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")

    DAILY_HOURS = 24.0

    # ---- Base sheets
    nodes_df = pd.read_excel(xls, "Nodes")
    arcs_df = pd.read_excel(xls, "Arcs_All")
    tdf = pd.read_excel(xls, "Timetable")
    bdf = pd.read_excel(xls, "Batches")

    # ---- Parameter sheets
    region_map_df = pd.read_excel(xls, "Region_Map")
    carbon_tax_df = pd.read_excel(xls, "Carbon_Tax")
    carbon_app_df = pd.read_excel(xls, "Carbon_Applicability")
    emis_df = pd.read_excel(xls, "Emission_Factors")
    mode_speed_df = pd.read_excel(xls, "Mode_Speeds")
    node_border_df = pd.read_excel(xls, "Node_Border")
    transship_df = pd.read_excel(xls, "Transshipment")
    waiting_df = pd.read_excel(xls, "Waiting_Costs")

    # Nodes
    node_names = nodes_df["EnglishName"].astype(str).tolist()
    node_region_code = dict(zip(nodes_df["EnglishName"].astype(str), nodes_df["Region"].astype(str)))

    # region code -> region group
    rc2g = dict(zip(region_map_df["RegionCode_in_Nodes"].astype(str),
                    region_map_df["RegionGroup"].astype(str)))

    def to_group(rc: str) -> str:
        rc = str(rc).strip()
        return rc2g.get(rc, rc)

    region_group_of_node = {n: to_group(r) for n, r in node_region_code.items()}

    # Node capacities -> TEU / time-bucket
    cap_cols = _detect_node_capacity_columns(nodes_df)
    node_caps: Dict[str, float] = {}
    for _, row in nodes_df.iterrows():
        n = str(row["EnglishName"]).strip()
        cap_val = None
        used = None
        for c in cap_cols:
            v = row.get(c, np.nan)
            if pd.notna(v):
                try:
                    cap_val = float(v)
                    used = str(c).lower()
                    break
                except Exception:
                    pass
        if cap_val is None:
            node_caps[n] = 1e18
        else:
            # Heuristic: TEU/h or TEU/day
            if used and ("teuh" in used or "hour" in used):
                node_caps[n] = cap_val * TIME_BUCKET_H
            else:
                node_caps[n] = cap_val * (TIME_BUCKET_H / DAILY_HOURS)

            if node_caps[n] <= 1e-6:
                node_caps[n] = 1e-6

    # Carbon tax (USD / tonCO2)
    carbon_tax_usd = dict(zip(carbon_tax_df["RegionGroup"].astype(str),
                              carbon_tax_df["CT_USD_per_tonCO2"].astype(float)))

    # Applicability theta(region, mode)
    theta: Dict[Tuple[str, str], int] = {}
    for _, row in carbon_app_df.iterrows():
        theta[(str(row["RegionGroup"]), str(row["Mode"]).lower())] = int(row["Theta"])

    # Emission factors (gCO2 / TEU-km)
    emis_factor_g_teu_km: Dict[Tuple[str, str], float] = {}
    for _, row in emis_df.iterrows():
        rg = str(row["RegionGroup"])
        m = str(row["Mode"]).lower()
        emis_factor_g_teu_km[(rg, m)] = float(row["gCO2_per_TEU_km_assuming10t"])

    # Mode speeds (km/h)
    mode_speed = dict(zip(mode_speed_df["Mode"].astype(str).str.lower(),
                          mode_speed_df["Speed_kmh"].astype(float)))

    # Transshipment (node, from_mode, to_mode) -> (cost, time)
    transship: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for _, row in transship_df.iterrows():
        key = (str(row["Node"]), str(row["FromMode"]).lower(), str(row["ToMode"]).lower())
        transship[key] = (float(row["TransferCost_USD_per_TEU"]), float(row["TransferTime_h"]))

    # Border info per node
    node_border: Dict[str, Dict[str, float]] = {}
    for _, row in node_border_df.iterrows():
        n = str(row["EnglishName"])
        node_border[n] = {
            "is_border": int(row["IsBorderNode"]),
            "customs_h": float(row["CustomsClearance_h"]),
            "bd_road_h": float(row["BorderDelay_road_h"]),
            "bd_rail_h": float(row["BorderDelay_rail_h"]),
            "bd_water_h": float(row["BorderDelay_water_h"]),
            "border_cap_day": float(row.get("BorderCapacity_TEUday", 0.0)),
        }

    # Waiting cost parameters + waiting/processing emission factor (gCO2 / TEU-hour)
    wait_emis_col = _pick_first_existing_column(
        waiting_df,
        ["WaitEmission_gCO2_per_TEU_h", "E_wait_gCO2_per_TEU_h", "WaitingEmission_gCO2_per_TEU_h"]
    )
    wait_emis_g_per_teu_h = None
    try:
        if wait_emis_col is not None and pd.notna(waiting_df.loc[0, wait_emis_col]):
            wait_emis_g_per_teu_h = float(waiting_df.loc[0, wait_emis_col])
    except Exception:
        wait_emis_g_per_teu_h = None

    if wait_emis_g_per_teu_h is None:
        wait_emis_g_per_teu_h = 0.0
        print("[INFO] Waiting_Costs: no wait-emission column found. "
              "Using default WAIT_EMISSION_gCO2_per_TEU_h = 0.0. "
              "Add column 'WaitEmission_gCO2_per_TEU_h' to enable waiting-emission modelling.")

    waiting_cost = {
        "hold": float(waiting_df.loc[0, "W_hold_USD_per_TEU_h"]),
        "proc": float(waiting_df.loc[0, "W_proc_USD_per_TEU_h"]),
        "cong": float(waiting_df.loc[0, "W_cong_USD_per_TEU_h"]),
        "emis_g_per_teu_h": float(wait_emis_g_per_teu_h),
    }

    # Arcs
    speed_col = _pick_first_existing_column(arcs_df, ["Speed_kmh", "SpeedKMH", "AvgSpeed_kmh", "Speed"])
    time_col  = _pick_first_existing_column(arcs_df, ["TravelTime_h", "Time_h", "TransitTime_h", "Duration_h", "TimeHours"])
    cap_h_col = _pick_first_existing_column(arcs_df, ["Capacity_TEUh", "Cap_TEUh"])
    cap_day_col = _pick_first_existing_column(arcs_df, ["Capacity_TEUday", "Cap_TEUday", "Capacity_TEU"])

    arcs: List[Arc] = []
    for _, row in arcs_df.iterrows():
        mode_raw = str(row["Mode"]).strip().lower()
        mode = "rail" if mode_raw == "rail" else mode_raw

        distance = _parse_float(row.get("Distance_km", 0.0))
        if not np.isfinite(distance):
            distance = 0.0

        # speed
        speed = None
        if speed_col is not None and pd.notna(row.get(speed_col, np.nan)):
            try:
                speed = float(row[speed_col])
            except Exception:
                speed = None

        if speed is None and time_col is not None and pd.notna(row.get(time_col, np.nan)):
            try:
                tt = float(row[time_col])
                if tt > 1e-9 and distance > 1e-9:
                    speed = distance / tt
            except Exception:
                speed = None

        if speed is None:
            speed = float(mode_speed.get(mode, 50.0))

        # capacity
        capacity = 1e18
        try:
            if cap_h_col is not None and pd.notna(row.get(cap_h_col, np.nan)):
                capacity = float(row[cap_h_col]) * TIME_BUCKET_H
            elif cap_day_col is not None and pd.notna(row.get(cap_day_col, np.nan)):
                cap_day = float(row[cap_day_col])
                capacity = cap_day * (TIME_BUCKET_H / DAILY_HOURS)
        except Exception:
            capacity = 1e18
        if capacity <= 1e-9:
            capacity = 1e-9

        arcs.append(Arc(
            from_node=str(row["OriginEN"]).strip(),
            to_node=str(row["DestEN"]).strip(),
            mode=mode,
            distance=float(distance),
            capacity=float(capacity),
            cost_per_teu_km=float(row["Cost_$_per_km"]),
            emission_per_teu_km=float(row["Emission_gCO2_per_tkm"]),
            speed_kmh=float(max(speed, 1.0)),
        ))

    # Timetable
    timetables: List[TimetableEntry] = []
    for _, row in tdf.iterrows():
        freq = float(row["Frequency_per_week"])
        hd = row.get("Headway_Hours", np.nan)
        if pd.isna(hd):
            hd = 168.0 / max(freq, 1.0)
        else:
            hd = float(hd)
        hd = min(float(hd), HEADWAY_CAP_H)

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
            first_departure_hour=float(fd),
            headway_hours=float(hd)
        ))

    # Batches (ET/LT auto unit)
    lt_vals = []
    for v in bdf["LT"].values.tolist():
        try:
            if pd.notna(v):
                lt_vals.append(float(v))
        except Exception:
            pass
    infer_days = False
    if lt_vals:
        med_lt = float(np.median(np.array(lt_vals, dtype=float)))
        infer_days = (med_lt <= 30.0)

    if infer_days:
        print("[INFO] Detected ET/LT likely in DAYS (median LT<=30). Auto converting to HOURS (*24).")
    else:
        print("[INFO] Detected ET/LT likely in HOURS (median LT>30). Keep as-is.")

    batches: List[Batch] = []
    for _, row in bdf.iterrows():
        origin = str(row["OriginEN"]).strip()
        dest = str(row["DestEN"]).strip()
        o_reg = node_region_code.get(origin)
        d_reg = node_region_code.get(dest)
        if o_reg in CHINA_REGIONS and d_reg in EUROPE_REGIONS:
            ET_raw = float(row["ET"])
            LT_raw = float(row["LT"])
            ET = ET_raw * 24.0 if infer_days else ET_raw
            LT = LT_raw * 24.0 if infer_days else LT_raw
            batches.append(Batch(
                batch_id=int(row["BatchID"]),
                origin=origin,
                destination=dest,
                quantity=float(row["QuantityTEU"]),
                ET=float(ET),
                LT=float(LT)
            ))

    print(f"[INFO] Number of batches loaded: {len(batches)}")

    return (
        node_names, node_caps, arcs, timetables, batches,
        region_group_of_node, carbon_tax_usd, theta,
        emis_factor_g_teu_km, transship, node_border, waiting_cost
    )


# ========================
# 4. Graph + timetable dict + Path Library (Dijkstra)
# ========================

def build_graph(arcs: List[Arc]) -> Dict[str, List[Arc]]:
    g: Dict[str, List[Arc]] = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append(a)
    return g


def build_timetable_dict(timetables: List[TimetableEntry]) -> Dict[Tuple[str, str, str], List[TimetableEntry]]:
    tt: Dict[Tuple[str, str, str], List[TimetableEntry]] = {}
    for t in timetables:
        tt.setdefault((t.from_node, t.to_node, t.mode), []).append(t)
    return tt


def next_departure_time(current_t: float, entries: List[TimetableEntry]) -> float:
    if not entries:
        return current_t
    best = float("inf")
    for e in entries:
        fd = float(e.first_departure_hour)
        hd = float(max(e.headway_hours, 1e-6))
        if current_t <= fd + 1e-12:
            dep = fd
        else:
            n = math.ceil((current_t - fd - 1e-10) / hd)
            dep = fd + n * hd
        best = min(best, dep)
    return best


def simulate_time_only(path: Path, ET: float, tt_dict,
                       transship: Dict[Tuple[str, str, str], Tuple[float, float]],
                       node_border: Dict[str, Dict[str, float]]) -> float:
    """
    Time-only simulation used for repair feasibility screening:
    - timetable waiting
    - transfer time on mode changes
    - border/customs time at arrival nodes
    (No queueing due to capacities; the queueing is handled only in evaluation.)
    """
    t = ET
    prev_mode = None

    for arc in path.arcs:
        # transshipment at arc.from_node if mode changes
        if prev_mode is not None and arc.mode != prev_mode:
            node_j = arc.from_node
            _, tt_h = transship.get((node_j, prev_mode, arc.mode), (150.0, 3.0))
            t += float(tt_h)
        prev_mode = arc.mode

        travel = arc.distance / max(arc.speed_kmh, 1.0)
        if arc.mode == "road":
            dep = t
        else:
            dep = next_departure_time(t, tt_dict.get((arc.from_node, arc.to_node, arc.mode), []))
        t = dep + travel

        # border/customs at arrival node
        info = node_border.get(arc.to_node)
        if info and int(info.get("is_border", 0)) == 1:
            cc = float(info.get("customs_h", 0.0))
            if arc.mode == "road":
                bd = float(info.get("bd_road_h", 0.0))
            elif arc.mode == "rail":
                bd = float(info.get("bd_rail_h", 0.0))
            else:
                bd = float(info.get("bd_water_h", 0.0))
            t += (cc + bd)

    return t - ET


def dijkstra_time_dependent(graph: Dict[str, List[Arc]], origin: str, dest: str,
                           start_time: float, tt_dict,
                           mode_weights: Optional[Dict[str, float]] = None,
                           max_hops: int = 50) -> Optional[List[Arc]]:
    pq = []
    heapq.heappush(pq, (start_time, start_time, origin, 0))

    min_arrival = {origin: start_time}
    prev: Dict[str, Tuple[str, Arc]] = {}

    if mode_weights is None:
        mode_weights = {}

    while pq:
        _, curr_real_time, u, hops = heapq.heappop(pq)

        if u == dest:
            break

        if curr_real_time > min_arrival.get(u, float("inf")) + 48.0:
            continue

        if hops >= max_hops:
            continue

        for arc in graph.get(u, []):
            v = arc.to_node

            if arc.mode == "road":
                dep_time = curr_real_time
            else:
                raw_dep = next_departure_time(curr_real_time, tt_dict.get((u, v, arc.mode), []))
                if raw_dep == float("inf"):
                    continue
                dep_time = raw_dep

            travel_time = arc.distance / max(arc.speed_kmh, 1.0)

            mw = mode_weights.get(arc.mode, 1.0)
            perceived_travel = travel_time * mw

            real_arrival_time = dep_time + travel_time

            priority_metric = dep_time + perceived_travel
            priority_metric *= (1.0 + random.uniform(-0.02, 0.02))

            if real_arrival_time < min_arrival.get(v, float("inf")):
                min_arrival[v] = real_arrival_time
                prev[v] = (u, arc)
                heapq.heappush(pq, (priority_metric, real_arrival_time, v, hops + 1))

    if dest not in prev:
        return None

    arcs_seq = []
    cur = dest
    while cur != origin:
        if cur not in prev:
            return None
        pu, arc = prev[cur]
        arcs_seq.append(arc)
        cur = pu
    arcs_seq.reverse()
    return arcs_seq


def build_path_library(arcs: List[Arc], batches: List[Batch], tt_dict,
                       K: int = 100, repeats: int = 40) -> Dict[Tuple[str, str], List[Path]]:
    graph = build_graph(arcs)
    lib: Dict[Tuple[str, str], List[Path]] = {}
    pid = 0

    print(f"[INFO] Building Diverse Time-Dependent Path Library for {len(batches)} batches...", flush=True)
    count_new = 0

    for i, b in enumerate(batches):
        od = (b.origin, b.destination)
        if od not in lib:
            lib[od] = []

        existing_signatures = set((tuple(p.nodes), tuple(p.modes)) for p in lib[od])

        found_seqs = []

        base = dijkstra_time_dependent(graph, b.origin, b.destination, b.ET, tt_dict, mode_weights=None)
        if base:
            found_seqs.append(base)

        for _ in range(repeats):
            mw = {
                "road": random.choice([1.0, 1.0, 3.0, 5.0, 10.0]),
                "rail": random.choice([1.0, 1.0, 2.0, 5.0]),
                "water": random.choice([0.5, 0.8, 1.0, 2.0])
            }
            seq = dijkstra_time_dependent(graph, b.origin, b.destination, b.ET, tt_dict, mode_weights=mw)
            if seq:
                found_seqs.append(seq)

        for seq in found_seqs:
            nodes = [seq[0].from_node] + [a.to_node for a in seq]
            modes = [a.mode for a in seq]
            sig = (tuple(nodes), tuple(modes))

            if sig not in existing_signatures:
                cost = sum(a.cost_per_teu_km * a.distance for a in seq)
                emis = sum(a.emission_per_teu_km * a.distance for a in seq)
                tt_base = sum(a.distance / max(a.speed_kmh, 1.0) for a in seq)

                new_p = Path(
                    path_id=pid,
                    origin=b.origin,
                    destination=b.destination,
                    nodes=nodes,
                    modes=modes,
                    arcs=seq,
                    base_cost_per_teu=cost,
                    base_emission_per_teu=emis,
                    base_travel_time_h=tt_base
                )
                pid += 1
                lib[od].append(new_p)
                existing_signatures.add(sig)
                count_new += 1

        if len(lib[od]) > K * 2:
            lib[od] = lib[od][:K * 2]

        if (i + 1) % 1 == 0:
            print(f"  Processed {i+1}/{len(batches)} batches... (Found {count_new} paths so far)", flush=True)

    print(f"[INFO] Path library built. Total unique paths across all ODs: {sum(len(v) for v in lib.values())}")
    return lib


def report_time_feasible_coverage(batches, path_lib, tt_dict, transship, node_border):
    bad = []
    print("\n[TIME FEASIBILITY CHECK] per-batch coverage:")
    for b in batches:
        od = (b.origin, b.destination)
        paths = path_lib.get(od, [])
        if not paths:
            print(f"  Batch {b.batch_id}: NO PATHS in library!")
            bad.append((b.batch_id, "no_paths"))
            continue

        feas_cnt = 0
        best_arr = float("inf")
        for p in paths:
            arr = b.ET + simulate_time_only(p, b.ET, tt_dict, transship, node_border)
            best_arr = min(best_arr, arr)
            if arr <= b.LT + 1e-9:
                feas_cnt += 1

        status = "OK" if feas_cnt > 0 else "IMPOSSIBLE"
        if feas_cnt == 0:
            print(f"  Batch {b.batch_id}: feas_paths={feas_cnt}/{len(paths)} | ET={b.ET:.1f}, LT={b.LT:.1f}, best_arr={best_arr:.1f} -> {status}")
            bad.append((b.batch_id, b.origin, b.destination, b.ET, b.LT, best_arr))

    if bad:
        print(f"\n[WARNING] {len(bad)} batches have ZERO time-feasible paths (ignoring capacity queueing).")
    else:
        print("\n[SUCCESS] All batches have at least one time-feasible path (ignoring capacity queueing).")
    print()
    return bad


# ========================
# 6. Evaluation (Queueing + Carbon + Border + Transshipment + ArcCap)
# ========================

def evaluate_individual(ind: Individual,
                        batches: List[Batch],
                        tt_dict,
                        node_caps: Dict[str, float],
                        region_group_of_node: Dict[str, str],
                        carbon_tax_usd: Dict[str, float],
                        theta: Dict[Tuple[str, str], int],
                        emis_factor_g_teu_km: Dict[Tuple[str, str], float],
                        transship: Dict[Tuple[str, str, str], Tuple[float, float]],
                        node_border: Dict[str, Dict[str, float]],
                        waiting_cost: Dict[str, float]):

    total_cost = 0.0
    total_emis_g = 0.0
    makespan = 0.0

    penalty = 0.0
    missing = False
    late = False

    # NEW: wait48 tracking (single waiting episode hard cap)
    wait48_any = False
    batch_wait48 = defaultdict(int)     # bid -> 0/1
    batch_wait48_first = {}             # bid -> first violation record

    def _flag_wait48(bid: int, where: str, node: str, wait_h: float, arc: Optional[Arc] = None) -> bool:
        nonlocal wait48_any, penalty
        if wait_h > WAIT_EVENT_CAP_H + WAIT48_EPS:
            wait48_any = True
            batch_wait48[bid] = 1
            if bid not in batch_wait48_first:
                batch_wait48_first[bid] = {
                    "where": where,
                    "node": str(node),
                    "wait_h": float(wait_h),
                    "from_node": (arc.from_node if arc else None),
                    "to_node": (arc.to_node if arc else None),
                    "mode": (arc.mode if arc else None),
                }
            penalty += WAIT48_HARD_PENALTY
            return True
        return False

    # waiting/processing emission factor (gCO2 per TEU-hour)
    wait_emis_g_per_teu_h = float(waiting_cost.get("emis_g_per_teu_h", 0.0))

    # Build tasks
    tasks = []
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if not allocs:
            penalty += 1e9
            missing = True
            continue

        for a in allocs:
            if a.share <= 1e-12:
                continue
            qty = a.share * b.quantity  # TEU
            tasks.append({
                "start_time": b.ET,
                "quantity": qty,
                "path": a.path,
                "batch": b
            })

    tasks.sort(key=lambda x: x["start_time"])

    # Queue states
    node_next_free_time: Dict[str, float] = defaultdict(float)
    arc_next_free_time: Dict[Tuple[str, str, str], float] = defaultdict(float)

    for task in tasks:
        curr_t = float(task["start_time"])
        qty = float(task["quantity"])
        path: Path = task["path"]
        batch: Batch = task["batch"]

        # base transport cost (distance-based) using stored path cost
        total_cost += float(path.base_cost_per_teu) * qty

        prev_mode = None

        for arc in path.arcs:
            # -------- Transshipment at arc.from_node if mode changes
            if prev_mode is not None and arc.mode != prev_mode:
                node_j = arc.from_node
                tc, tt_h = transship.get((node_j, prev_mode, arc.mode), (150.0, 3.0))
                # cost
                total_cost += float(tc) * qty
                # time + processing cost
                curr_t += float(tt_h)
                total_cost += float(tt_h) * qty * waiting_cost["proc"]
                # emission during transfer processing/waiting
                if wait_emis_g_per_teu_h > 0 and tt_h > 0:
                    total_emis_g += wait_emis_g_per_teu_h * qty * float(tt_h)

            prev_mode = arc.mode

            # -------- Timetable waiting (hold) at FROM node
            travel = arc.distance / max(arc.speed_kmh, 1.0)
            if arc.mode == "road":
                dep = curr_t
            else:
                dep = next_departure_time(curr_t, tt_dict.get((arc.from_node, arc.to_node, arc.mode), []))

            timetable_wait = dep - curr_t
            if timetable_wait > 0:
                total_cost += timetable_wait * qty * waiting_cost["hold"]
                if wait_emis_g_per_teu_h > 0:
                    total_emis_g += wait_emis_g_per_teu_h * qty * timetable_wait
                # NEW: hard cap check
                _flag_wait48(batch.batch_id, "TIMETABLE", arc.from_node, timetable_wait, arc=arc)

            # -------- Arc capacity queueing (CAP_{ijm}) at FROM node
            arc_key = (arc.from_node, arc.to_node, arc.mode)
            arc_free = arc_next_free_time[arc_key]
            dep2 = max(dep, arc_free)

            arc_wait = dep2 - dep
            if arc_wait > 0:
                total_cost += arc_wait * qty * waiting_cost["cong"]
                if wait_emis_g_per_teu_h > 0:
                    total_emis_g += wait_emis_g_per_teu_h * qty * arc_wait
                # NEW: hard cap check
                _flag_wait48(batch.batch_id, "ARC_QUEUE", arc.from_node, arc_wait, arc=arc)

            arc_cap = max(float(arc.capacity), 1e-9)  # TEU per time-bucket
            arc_service = qty / arc_cap  # hours (continuous service approximation)
            arc_next_free_time[arc_key] = dep2 + arc_service
            total_cost += arc_service * qty * waiting_cost["proc"]
            if wait_emis_g_per_teu_h > 0 and arc_service > 0:
                total_emis_g += wait_emis_g_per_teu_h * qty * arc_service

            # depart and travel
            arr_at_next = dep2 + travel

            # -------- Border/customs at arrival node (hold) at TO node
            info = node_border.get(arc.to_node)
            if info and int(info.get("is_border", 0)) == 1:
                cc = float(info.get("customs_h", 0.0))
                if arc.mode == "road":
                    bd = float(info.get("bd_road_h", 0.0))
                elif arc.mode == "rail":
                    bd = float(info.get("bd_rail_h", 0.0))
                else:
                    bd = float(info.get("bd_water_h", 0.0))
                border_time = cc + bd
                if border_time > 0:
                    total_cost += border_time * qty * waiting_cost["hold"]
                    arr_at_next += border_time
                    if wait_emis_g_per_teu_h > 0:
                        total_emis_g += wait_emis_g_per_teu_h * qty * border_time
                    # NEW: hard cap check
                    _flag_wait48(batch.batch_id, "BORDER", arc.to_node, border_time, arc=arc)

            # -------- Node capacity queueing at next node (cong + proc) at TO node
            next_node = arc.to_node
            proc_rate = max(node_caps.get(next_node, 1e18), 1e-9)  # TEU per bucket
            service_time = qty / proc_rate

            node_free = node_next_free_time[next_node]
            start_service_time = max(arr_at_next, node_free)
            node_wait = start_service_time - arr_at_next
            if node_wait > 0:
                total_cost += node_wait * qty * waiting_cost["cong"]
                if wait_emis_g_per_teu_h > 0:
                    total_emis_g += wait_emis_g_per_teu_h * qty * node_wait
                # NEW: hard cap check
                _flag_wait48(batch.batch_id, "NODE_QUEUE", arc.to_node, node_wait, arc=arc)

            finish_service_time = start_service_time + service_time
            node_next_free_time[next_node] = finish_service_time
            total_cost += service_time * qty * waiting_cost["proc"]
            if wait_emis_g_per_teu_h > 0 and service_time > 0:
                total_emis_g += wait_emis_g_per_teu_h * qty * service_time

            curr_t = finish_service_time

            # -------- Emission + carbon cost for this arc (regional heterogeneity)
            rg = region_group_of_node.get(arc.from_node, region_group_of_node.get(arc.to_node, "WE"))
            ef = emis_factor_g_teu_km.get((rg, arc.mode), 0.0)  # gCO2 per TEU-km
            emis_g = ef * arc.distance * qty
            total_emis_g += emis_g

            ct = carbon_tax_usd.get(rg, 0.0)  # USD/tonCO2
            th = int(theta.get((rg, arc.mode), 0))
            carbon_cost = (emis_g / 1e6) * ct * th  # g -> ton
            total_cost += carbon_cost

        final_arrival = curr_t
        makespan = max(makespan, final_arrival)

        if final_arrival > batch.LT + 1e-9:
            late = True
            penalty += (final_arrival - batch.LT) * 500.0

    total_emis_ton = total_emis_g / 1e6

    ind.objectives = (total_cost + penalty, total_emis_ton + penalty * 1e-6, makespan)
    ind.penalty = penalty
    ind.feasible = not (missing or late or wait48_any)
    ind.infeas_reason = {
        "missing_alloc": int(missing),
        "late": int(late),
        "wait48": int(wait48_any),
        "arc_cap": 0,
        "node_cap": 0
    }
    ind.debug_info["batch_wait48"] = dict(batch_wait48)
    ind.debug_info["batch_wait48_first"] = dict(batch_wait48_first)


# ========================
# 7. Repair & GA Operators
# ========================

def repair_individual(ind: Individual,
                      batches: List[Batch],
                      path_lib,
                      tt_dict,
                      transship,
                      node_border):
    for b in batches:
        od = (b.origin, b.destination)
        key = (b.origin, b.destination, b.batch_id)
        paths = path_lib.get(od, [])
        if not paths:
            continue

        allocs = ind.od_allocations.get(key, [])

        if not allocs:
            best = None
            best_arr = float("inf")
            best_feas = None
            best_feas_arr = float("inf")

            for p in paths:
                arr = b.ET + simulate_time_only(p, b.ET, tt_dict, transship, node_border)
                if arr < best_arr:
                    best_arr = arr
                    best = p
                if arr <= b.LT + 1e-9 and arr < best_feas_arr:
                    best_feas_arr = arr
                    best_feas = p

            chosen = best_feas if best_feas else best
            if chosen:
                ind.od_allocations[key] = [PathAllocation(path=chosen, share=1.0)]
            continue

        kept = []
        for a in allocs:
            if a.share <= 1e-12:
                continue
            arr = b.ET + simulate_time_only(a.path, b.ET, tt_dict, transship, node_border)
            if arr <= b.LT + 1e-9:
                kept.append(a)

        if kept:
            ind.od_allocations[key] = merge_and_normalize(kept)
        else:
            best = None
            best_arr = float("inf")
            best_feas = None
            best_feas_arr = float("inf")

            for p in paths:
                arr = b.ET + simulate_time_only(p, b.ET, tt_dict, transship, node_border)
                if arr < best_arr:
                    best_arr = arr
                    best = p
                if arr <= b.LT + 1e-9 and arr < best_feas_arr:
                    best_feas_arr = arr
                    best_feas = p

            chosen = best_feas if best_feas else best
            if chosen:
                ind.od_allocations[key] = [PathAllocation(path=chosen, share=1.0)]


def repair_wait48_post_eval(ind: Individual,
                            batches: List[Batch],
                            path_lib,
                            tt_dict,
                            transship,
                            node_border) -> bool:
    """
    Post-eval hard repair: only triggered when wait48 happened.
    Strategy:
    - For each violated batch, read first violation (where,node).
    - Prefer time-feasible paths that avoid the violated node.
    - If none, fall back to best time-feasible path (ignoring avoid).
    - If still none, keep original (let GA淘汰).
    """
    info = (ind.debug_info or {}).get("batch_wait48_first", {})
    if not info:
        return False

    changed = False
    bid2batch = {b.batch_id: b for b in batches}

    def arr_time(b: Batch, p: Path) -> float:
        return b.ET + simulate_time_only(p, b.ET, tt_dict, transship, node_border)

    for bid, v in info.items():
        b = bid2batch.get(int(bid))
        if b is None:
            continue

        od = (b.origin, b.destination)
        key = (b.origin, b.destination, b.batch_id)
        paths = path_lib.get(od, [])
        if not paths:
            continue

        avoid_node = str(v.get("node", ""))

        cand_avoid = [p for p in paths if (avoid_node not in p.nodes)]
        feas_avoid = [p for p in cand_avoid if arr_time(b, p) <= b.LT + 1e-9]

        feas_all = [p for p in paths if arr_time(b, p) <= b.LT + 1e-9]

        chosen = None
        if feas_avoid:
            chosen = min(feas_avoid, key=lambda p: arr_time(b, p))
        elif feas_all:
            chosen = min(feas_all, key=lambda p: arr_time(b, p))

        if chosen is not None:
            ind.od_allocations[key] = [PathAllocation(path=chosen, share=1.0)]
            changed = True

    return changed


def random_initial_individual(batches, path_lib, tt_dict, transship, node_border, max_paths=3) -> Individual:
    ind = Individual()
    for b in batches:
        od = (b.origin, b.destination)
        paths = path_lib.get(od, [])
        if not paths:
            continue

        feas = []
        for p in paths:
            arr = b.ET + simulate_time_only(p, b.ET, tt_dict, transship, node_border)
            if arr <= b.LT + 1e-9:
                feas.append(p)
        pool = feas if feas else paths

        k = random.randint(1, min(max_paths, len(pool)))
        chosen = random.sample(pool, k)
        key = (b.origin, b.destination, b.batch_id)
        ind.od_allocations[key] = merge_and_normalize([PathAllocation(path=p, share=random.random()) for p in chosen])

    repair_individual(ind, batches, path_lib, tt_dict, transship, node_border)
    return ind


def crossover_structural(ind1: Individual, ind2: Individual, batches: List[Batch]) -> Tuple[Individual, Individual]:
    c1 = Individual()
    c2 = Individual()
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1 = ind1.od_allocations.get(key, [])
        g2 = ind2.od_allocations.get(key, [])
        if not g1 and not g2:
            continue
        if not g1:
            c1.od_allocations[key] = [clone_gene(x) for x in g2]
            c2.od_allocations[key] = [clone_gene(x) for x in g2]
            continue
        if not g2:
            c1.od_allocations[key] = [clone_gene(x) for x in g1]
            c2.od_allocations[key] = [clone_gene(x) for x in g1]
            continue

        cut1 = random.randint(0, len(g1))
        cut2 = random.randint(0, len(g2))
        gg1 = [clone_gene(x) for x in g1[:cut1]] + [clone_gene(x) for x in g2[cut2:]]
        gg2 = [clone_gene(x) for x in g2[:cut2]] + [clone_gene(x) for x in g1[cut1:]]
        c1.od_allocations[key] = merge_and_normalize(gg1)
        c2.od_allocations[key] = merge_and_normalize(gg2)

    return c1, c2


# mutation caches
_ARC_LOOKUP: Dict[Tuple[str, str, str], Arc] = {}
_ARC_MODE_OPTIONS: Dict[Tuple[str, str], List[str]] = {}
_CACHE_BUILT = False


def _build_arc_caches_from_path_lib(path_lib: Dict[Tuple[str, str], List[Path]]):
    global _ARC_LOOKUP, _ARC_MODE_OPTIONS, _CACHE_BUILT
    if _CACHE_BUILT:
        return
    arc_lookup = {}
    mode_opts: Dict[Tuple[str, str], set] = {}
    for _, paths in path_lib.items():
        for p in paths:
            for a in p.arcs:
                arc_lookup[(a.from_node, a.to_node, a.mode)] = a
                mode_opts.setdefault((a.from_node, a.to_node), set()).add(a.mode)
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


def _mut_add_path(allocs: List[PathAllocation], od: Tuple[str, str], path_lib):
    paths = path_lib.get(od, [])
    if not paths:
        return allocs
    cur = {a.path for a in allocs}
    cand = [p for p in paths if p not in cur]
    if not cand:
        return allocs
    allocs.append(PathAllocation(path=random.choice(cand), share=0.2))
    return merge_and_normalize(allocs)


def _mut_delete_path(allocs: List[PathAllocation]):
    if len(allocs) <= 1:
        return allocs
    allocs.pop(random.randint(0, len(allocs) - 1))
    return merge_and_normalize(allocs)


def _mut_modify_share(allocs: List[PathAllocation]):
    if not allocs:
        return allocs
    a = random.choice(allocs)
    a.share *= random.uniform(0.5, 1.5)
    return merge_and_normalize(allocs)


def _mut_mode_single_arc(allocs: List[PathAllocation]):
    if not allocs:
        return allocs
    a = random.choice(allocs)
    p = a.path
    if not p.arcs:
        return allocs

    pos = random.randrange(len(p.arcs))
    old = p.arcs[pos]
    u, v = old.from_node, old.to_node
    old_mode = old.mode

    modes = _ARC_MODE_OPTIONS.get((u, v), [])
    if len(modes) <= 1:
        return allocs
    other = [m for m in modes if m != old_mode]
    if not other:
        return allocs
    new_mode = random.choice(other)
    new_arc = _ARC_LOOKUP.get((u, v, new_mode))
    if new_arc is None:
        return allocs

    new_arcs = list(p.arcs)
    new_arcs[pos] = new_arc
    new_modes = list(p.modes)
    new_modes[pos] = new_mode

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
        if allocs[i].path == a.path and abs(allocs[i].share - a.share) < 1e-12:
            allocs[i] = PathAllocation(path=new_path, share=a.share)
            break
    return merge_and_normalize(allocs)


def _allocs_signature(allocs: List[PathAllocation], share_round: int = 8) -> Tuple:
    items = []
    for a in allocs:
        p = a.path
        struct = (tuple(p.nodes), tuple(p.modes))
        items.append((struct, round(float(a.share), share_round)))
    items.sort(key=lambda x: (x[0], x[1]))
    return tuple(items)


@dataclass
class MutationTracker:
    records: List[Dict] = field(default_factory=list)

    def start_gen(self, gen: int):
        self.records.append({
            "gen": gen,
            "attempt_add": 0, "exec_add": 0,
            "attempt_del": 0, "exec_del": 0,
            "attempt_mod": 0, "exec_mod": 0,
            "attempt_mode": 0, "exec_mode": 0,
        })

    def _cur(self) -> Dict:
        if not self.records:
            raise RuntimeError("MutationTracker: start_gen() must be called before record().")
        return self.records[-1]

    def record(self, op: str, executed: bool):
        r = self._cur()
        key_a = f"attempt_{op}"
        key_e = f"exec_{op}"
        if key_a in r:
            r[key_a] += 1
        if executed and key_e in r:
            r[key_e] += 1

    def finalize_current_gen(self):
        r = self._cur()
        attempt_total = int(r["attempt_add"] + r["attempt_del"] + r["attempt_mod"] + r["attempt_mode"])
        exec_total = int(r["exec_add"] + r["exec_del"] + r["exec_mod"] + r["exec_mode"])
        r["attempt_total"] = attempt_total
        r["exec_total"] = exec_total

        def safe_div(a, b):
            return float(a) / float(b) if b > 0 else 0.0

        r["attempt_share_add"] = safe_div(r["attempt_add"], attempt_total)
        r["attempt_share_del"] = safe_div(r["attempt_del"], attempt_total)
        r["attempt_share_mod"] = safe_div(r["attempt_mod"], attempt_total)
        r["attempt_share_mode"] = safe_div(r["attempt_mode"], attempt_total)

        r["exec_share_add"] = safe_div(r["exec_add"], exec_total)
        r["exec_share_del"] = safe_div(r["exec_del"], exec_total)
        r["exec_share_mod"] = safe_div(r["exec_mod"], exec_total)
        r["exec_share_mode"] = safe_div(r["exec_mode"], exec_total)

        r["success_rate"] = safe_div(exec_total, attempt_total)
        r["success_add"] = safe_div(r["exec_add"], r["attempt_add"])
        r["success_del"] = safe_div(r["exec_del"], r["attempt_del"])
        r["success_mod"] = safe_div(r["exec_mod"], r["attempt_mod"])
        r["success_mode"] = safe_div(r["exec_mode"], r["attempt_mode"])


def mutate_structural(ind: Individual, batches: List[Batch], path_lib, tracker: Optional[MutationTracker] = None):
    _build_arc_caches_from_path_lib(path_lib)
    if not batches:
        return
    b = random.choice(batches)
    key = (b.origin, b.destination, b.batch_id)
    od = (b.origin, b.destination)

    allocs = ind.od_allocations.get(key, [])
    if not path_lib.get(od, []):
        return

    can_del = len(allocs) > 1
    op = _roulette_choice([
        ("add", W_ADD),
        ("del", W_DEL if can_del else 0.0),
        ("mod", W_MOD),
        ("mode", W_MODE),
    ])

    before_sig = _allocs_signature(allocs)

    if op == "add":
        allocs2 = _mut_add_path(allocs, od, path_lib)
    elif op == "del":
        allocs2 = _mut_delete_path(allocs)
    elif op == "mod":
        allocs2 = _mut_modify_share(allocs)
    else:
        allocs2 = _mut_mode_single_arc(allocs)

    after_sig = _allocs_signature(allocs2)
    executed = (after_sig != before_sig)

    if tracker is not None:
        tracker.record(op, executed)

    ind.od_allocations[key] = allocs2


# ========================
# 8. Metrics & timing
# ========================

class HypervolumeCalculator:
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


def unique_individuals_by_objectives(front: List[Individual], tol: float = 1e-6) -> List[Individual]:
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
    hv_time: float = 0.0
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


def _dominates_obj(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_filter_objectives(objs: List[Tuple[float, float, float]], tol: float = 1e-9) -> List[Tuple[float, float, float]]:
    if not objs:
        return []

    uniq = []
    for o in objs:
        ok = True
        for u in uniq:
            if (abs(o[0]-u[0]) <= tol and abs(o[1]-u[1]) <= tol and abs(o[2]-u[2]) <= tol):
                ok = False
                break
        if ok:
            uniq.append(o)

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


def normalise_by_Pstar(A_objs: List[Tuple[float, float, float]], Pstar: List[Tuple[float, float, float]]):
    A = np.array(A_objs, dtype=float)
    P = np.array(Pstar, dtype=float)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    A_norm = (A - mins) / ranges
    return A_norm, mins, maxs


def igd_plus(A_norm: np.ndarray, P_norm: np.ndarray) -> float:
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
# 9. NSGA-II core
# ========================

def dominates(a: Individual, b: Individual) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(x < y for x, y in zip(a.objectives, b.objectives))


def non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    S = {p: [] for p in pop}
    n = {p: 0 for p in pop}
    fronts = [[]]

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
        nxt = []
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
    d = {ind: 0.0 for ind in front}
    if l == 0:
        return d
    m = len(front[0].objectives)
    for i in range(m):
        front.sort(key=lambda x: x.objectives[i])
        d[front[0]] = float("inf")
        d[front[-1]] = float("inf")
        rng = front[-1].objectives[i] - front[0].objectives[i]
        if rng == 0:
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


def _count_infeas(pop: List[Individual]) -> Dict[str, int]:
    agg = {"missing_alloc": 0, "late": 0, "wait48": 0, "arc_cap": 0, "node_cap": 0}
    for ind in pop:
        if ind.feasible:
            continue
        r = ind.infeas_reason or {}
        for k in agg:
            agg[k] += int(r.get(k, 0))
    return agg


def run_nsga2_analytics(filename="data.xlsx", pop_size=60, generations=400, log_every=10):
    stats = RunStats()
    t_run0 = time.perf_counter()

    print("Loading data...")
    (node_names, node_caps, arcs, timetables, batches,
     region_group_of_node, carbon_tax_usd, theta,
     emis_factor_g_teu_km, transship, node_border, waiting_cost) = load_network_from_extended(filename)

    tt_dict = build_timetable_dict(timetables)

    # Path library
    path_lib = build_path_library(arcs, batches, tt_dict, K=100, repeats=40)
    _ = report_time_feasible_coverage(batches, path_lib, tt_dict, transship, node_border)

    print("Initializing Population...")
    t_init0 = time.perf_counter()
    population = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib, tt_dict, transship, node_border)
        t0 = time.perf_counter()
        evaluate_individual(ind, batches, tt_dict, node_caps,
                            region_group_of_node, carbon_tax_usd, theta,
                            emis_factor_g_teu_km, transship, node_border, waiting_cost)
        dt = time.perf_counter() - t0
        stats.evaluation_time += dt
        stats.evaluation_calls += 1
        population.append(ind)
    stats.init_time = time.perf_counter() - t_init0

    print(f"[INIT] FeasiblePop={sum(1 for x in population if x.feasible)}/{len(population)} | infeas={_count_infeas(population)}")

    history_fronts: List[Tuple[int, List[Tuple[float, float, float]]]] = []
    all_feasible_front0_objs: List[Tuple[float, float, float]] = []

    mut_tracker = MutationTracker()

    for gen in range(generations):
        t_gen0 = time.perf_counter()
        mut_tracker.start_gen(gen)

        t0 = time.perf_counter()
        fronts = non_dominated_sort(population)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        front0 = fronts[0]
        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-6)

        current_front_objs = [ind.objectives for ind in front0_unique]
        history_fronts.append((gen, current_front_objs))

        for ind in feasible_front0:
            all_feasible_front0_objs.append(ind.objectives)

        ranks = {}
        for r, f in enumerate(fronts):
            for ind in f:
                ranks[ind] = r

        t0 = time.perf_counter()
        dists = {}
        for f in fronts:
            dists.update(crowding_distance(f))
        dt = time.perf_counter() - t0
        stats.crowd_time += dt
        stats.crowd_calls += 1

        if gen % log_every == 0 or gen == generations - 1:
            print(f"Gen {gen}: Front0Unique={len(front0_unique)} | FeasiblePop={sum(1 for x in population if x.feasible)}/{len(population)}")

        mating = []
        while len(mating) < pop_size:
            mating.append(tournament_select(population, dists, ranks))

        offspring = []
        gen_crossover_time = 0.0
        gen_mutation_time = 0.0
        gen_eval_time = 0.0
        gen_eval_calls = 0
        gen_crossover_calls = 0
        gen_mutation_calls = 0

        while len(offspring) < pop_size:
            if random.random() < 0.7:
                p1, p2 = random.sample(mating, 2)
                t0 = time.perf_counter()
                c1, c2 = crossover_structural(p1, p2, batches)
                dt = time.perf_counter() - t0
                stats.crossover_time += dt
                stats.crossover_calls += 1
                gen_crossover_time += dt
                gen_crossover_calls += 1
            else:
                c1 = random_initial_individual(batches, path_lib, tt_dict, transship, node_border)
                c2 = random_initial_individual(batches, path_lib, tt_dict, transship, node_border)

            if random.random() < 0.3:
                t0 = time.perf_counter()
                mutate_structural(c1, batches, path_lib, tracker=mut_tracker)
                dt = time.perf_counter() - t0
                stats.mutation_time += dt
                stats.mutation_calls += 1
                gen_mutation_time += dt
                gen_mutation_calls += 1

            if random.random() < 0.3:
                t0 = time.perf_counter()
                mutate_structural(c2, batches, path_lib, tracker=mut_tracker)
                dt = time.perf_counter() - t0
                stats.mutation_time += dt
                stats.mutation_calls += 1
                gen_mutation_time += dt
                gen_mutation_calls += 1

            repair_individual(c1, batches, path_lib, tt_dict, transship, node_border)
            repair_individual(c2, batches, path_lib, tt_dict, transship, node_border)

            # ----- evaluate c1
            t0 = time.perf_counter()
            evaluate_individual(c1, batches, tt_dict, node_caps,
                                region_group_of_node, carbon_tax_usd, theta,
                                emis_factor_g_teu_km, transship, node_border, waiting_cost)
            dt = time.perf_counter() - t0
            stats.evaluation_time += dt
            stats.evaluation_calls += 1
            gen_eval_time += dt
            gen_eval_calls += 1

            # NEW: post-eval hard repair only when wait48 infeasible
            if (not c1.feasible) and int((c1.infeas_reason or {}).get("wait48", 0)) == 1:
                if repair_wait48_post_eval(c1, batches, path_lib, tt_dict, transship, node_border):
                    t0 = time.perf_counter()
                    evaluate_individual(c1, batches, tt_dict, node_caps,
                                        region_group_of_node, carbon_tax_usd, theta,
                                        emis_factor_g_teu_km, transship, node_border, waiting_cost)
                    dt = time.perf_counter() - t0
                    stats.evaluation_time += dt
                    stats.evaluation_calls += 1
                    gen_eval_time += dt
                    gen_eval_calls += 1

            # ----- evaluate c2
            t0 = time.perf_counter()
            evaluate_individual(c2, batches, tt_dict, node_caps,
                                region_group_of_node, carbon_tax_usd, theta,
                                emis_factor_g_teu_km, transship, node_border, waiting_cost)
            dt = time.perf_counter() - t0
            stats.evaluation_time += dt
            stats.evaluation_calls += 1
            gen_eval_time += dt
            gen_eval_calls += 1

            # NEW: post-eval hard repair only when wait48 infeasible
            if (not c2.feasible) and int((c2.infeas_reason or {}).get("wait48", 0)) == 1:
                if repair_wait48_post_eval(c2, batches, path_lib, tt_dict, transship, node_border):
                    t0 = time.perf_counter()
                    evaluate_individual(c2, batches, tt_dict, node_caps,
                                        region_group_of_node, carbon_tax_usd, theta,
                                        emis_factor_g_teu_km, transship, node_border, waiting_cost)
                    dt = time.perf_counter() - t0
                    stats.evaluation_time += dt
                    stats.evaluation_calls += 1
                    gen_eval_time += dt
                    gen_eval_calls += 1

            offspring.append(c1)
            offspring.append(c2)

        combined = population + offspring

        t0 = time.perf_counter()
        fronts2 = non_dominated_sort(combined)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        new_pop = []
        for f in fronts2:
            if len(new_pop) + len(f) <= pop_size:
                new_pop.extend(f)
            else:
                t0 = time.perf_counter()
                cd = crowding_distance(f)
                dt = time.perf_counter() - t0
                stats.crowd_time += dt
                stats.crowd_calls += 1
                f.sort(key=lambda x: cd[x], reverse=True)
                new_pop.extend(f[:pop_size - len(new_pop)])
                break

        population = new_pop

        mut_tracker.finalize_current_gen()

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

    final_fronts = non_dominated_sort(population)
    front0 = final_fronts[0]
    feasible_front0 = [ind for ind in front0 if ind.feasible]
    base_front = feasible_front0 if feasible_front0 else front0
    pareto_front = unique_individuals_by_objectives(base_front, tol=1e-6)

    final_feasible = [ind for ind in pareto_front if ind.feasible]
    final_feasible_nd_objs = [ind.objectives for ind in (final_feasible if final_feasible else pareto_front)]
    final_feasible_nd_objs = pareto_filter_objectives(final_feasible_nd_objs, tol=1e-6)

    stats.run_total_time = time.perf_counter() - t_run0

    print("\n========== TIMING SUMMARY (this run) ==========")
    s = stats.summary_dict()
    for k, v in s.items():
        print(f"{k}: {v}")
    print(f"Final Pareto size (unique, prefer feasible): {len(pareto_front)}")
    print("=============================================\n")

    return population, pareto_front, batches, history_fronts, stats, final_feasible_nd_objs, all_feasible_front0_objs, mut_tracker


# ========================
# 10. Output & plots
# ========================

def save_pareto_solutions(pareto: List[Individual], batches: List[Batch], filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-6)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (Unique, Feasible-first) =====\n\n")
        for i, ind in enumerate(pareto):
            cost, emit, time_ = ind.objectives
            f.write(f"===== Solution {i} =====\n")
            f.write(f"Objectives: Cost={cost:.6f}, Emission_ton={emit:.6f}, Time={time_:.6f}, Penalty={ind.penalty:.6f}\n")
            f.write(f"Feasible={ind.feasible}\n")
            f.write(f"InfeasReason={ind.infeas_reason}\n")
            if int((ind.infeas_reason or {}).get("wait48", 0)) == 1:
                f.write(f"wait48_first={ind.debug_info.get('batch_wait48_first', {})}\n")
            f.write("\n")

            for b in batches:
                key = (b.origin, b.destination, b.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if allocs:
                    f.write(f"Batch {b.batch_id}: {b.origin}->{b.destination} Q={b.quantity}\n")
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
    ax1.set_ylabel('Emission (ton CO2)')
    ax1.set_zlabel('Time')
    ax1.set_title('Pareto Front Evolution (Color=Generation)')
    ax1.legend()
    plt.tight_layout()
    plt.savefig("pareto_evolution_3d.png", dpi=300)
    plt.close()
    print("[Saved] pareto_evolution_3d.png")


def _sanitize_series(y: List[float], fallback: float = 0.0) -> np.ndarray:
    arr = np.array(y, dtype=float)
    arr[~np.isfinite(arr)] = np.nan

    if np.all(np.isnan(arr)):
        return np.full_like(arr, fallback, dtype=float)

    for i in range(len(arr)):
        if np.isnan(arr[i]):
            if i > 0:
                arr[i] = arr[i - 1]

    if np.isnan(arr[0]):
        first_valid = None
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                first_valid = i
                break
        if first_valid is None:
            return np.full_like(arr, fallback, dtype=float)
        arr[:first_valid] = arr[first_valid]

    arr = np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)
    return arr


def plot_single_run_line(series: List[float], title: str, ylabel: str, filename: str, marker_every: int = 1):
    y = _sanitize_series(series, fallback=0.0)
    gens = np.arange(len(y))

    if marker_every <= 0:
        marker_every = 1
    auto_me = max(1, len(gens) // 50)
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


def plot_mean_convergence_line(all_run_metrics: List[List[float]], title: str, ylabel: str, filename: str):
    if not all_run_metrics:
        print(f"[WARN] Empty metrics for {title}, skip.")
        return

    clean_runs = []
    for run in all_run_metrics:
        clean_runs.append(_sanitize_series(run, fallback=0.0))

    data = np.vstack(clean_runs)
    mean_curve = np.mean(data, axis=0)
    gens = np.arange(len(mean_curve))

    me = max(1, len(gens) // 50)
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


def save_mutation_stats_and_plots(tracker: MutationTracker,
                                 csv_name="mutation_stats_best_run.csv"):
    if not tracker.records:
        print("[WARN] No mutation records; skip mutation plots.")
        return

    df = pd.DataFrame(tracker.records)
    df.to_csv(csv_name, index=False)
    print(f"[Saved] {csv_name}")

    gens = df["gen"].values

    plt.figure(figsize=(10, 6))
    plt.stackplot(
        gens,
        df["attempt_share_add"].values,
        df["attempt_share_del"].values,
        df["attempt_share_mod"].values,
        df["attempt_share_mode"].values,
        labels=["add", "del", "mod", "mode"]
    )
    plt.xlabel("Generation")
    plt.ylabel("Share")
    plt.title("Mutation Attempt Share (roulette selected)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mutation_attempt_share_stacked.png", dpi=300)
    plt.close()
    print("[Saved] mutation_attempt_share_stacked.png")

    plt.figure(figsize=(10, 6))
    plt.stackplot(
        gens,
        df["exec_share_add"].values,
        df["exec_share_del"].values,
        df["exec_share_mod"].values,
        df["exec_share_mode"].values,
        labels=["add", "del", "mod", "mode"]
    )
    plt.xlabel("Generation")
    plt.ylabel("Share")
    plt.title("Mutation Execution Share (actually changed gene)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mutation_exec_share_stacked.png", dpi=300)
    plt.close()
    print("[Saved] mutation_exec_share_stacked.png")

    plt.figure(figsize=(10, 5))
    plt.plot(gens, df["success_rate"].values, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("exec_total / attempt_total")
    plt.title("Overall Mutation Success Rate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mutation_success_rate.png", dpi=300)
    plt.close()
    print("[Saved] mutation_success_rate.png")

    plt.figure(figsize=(10, 5))
    plt.plot(gens, df["success_add"].values, linewidth=2, label="add")
    plt.plot(gens, df["success_del"].values, linewidth=2, label="del")
    plt.plot(gens, df["success_mod"].values, linewidth=2, label="mod")
    plt.plot(gens, df["success_mode"].values, linewidth=2, label="mode")
    plt.xlabel("Generation")
    plt.ylabel("exec_op / attempt_op")
    plt.title("Per-Operator Mutation Success Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mutation_operator_success_rate.png", dpi=300)
    plt.close()
    print("[Saved] mutation_operator_success_rate.png")


# ========================
# 11. Main
# ========================

if __name__ == "__main__":
    filename = "data.xlsx"
    pop_size = 60
    generations = 400
    runs = 30

    print(f"\nStarting Experiment: Runs={runs}, Gen={generations}, Pop={pop_size}\n")

    cache_runs = []
    all_objs_for_Pstar: List[Tuple[float, float, float]] = []
    all_run_summaries: List[Dict] = []

    t_all0 = time.perf_counter()

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id}/{runs-1}, seed={seed} ==========\n")

        pop, pareto, batches, h_fronts, stats, final_feas_nd, feas_front0_allgens, mut_tracker = run_nsga2_analytics(
            filename=filename,
            pop_size=pop_size,
            generations=generations,
            log_every=10
        )

        all_objs_for_Pstar.extend(final_feas_nd)
        all_objs_for_Pstar.extend(pareto_filter_objectives(feas_front0_allgens, tol=1e-6))

        cache_runs.append({
            "run_id": run_id,
            "seed": seed,
            "population": pop,
            "pareto": pareto,
            "batches": batches,
            "history_fronts": h_fronts,
            "stats": stats,
            "mutation_tracker": mut_tracker
        })

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

    Pstar = build_reference_front_Pstar(all_objs_for_Pstar)
    print(f"\n[Global P*] Size: {len(Pstar)}")

    all_runs_igds = []
    all_runs_sps = []
    all_runs_hvs = []

    for item in cache_runs:
        h_fronts = item["history_fronts"]
        _, igds, sps, hvs = compute_gen_metrics_wrt_Pstar(h_fronts, Pstar, hv_norm_samples=20000)

        all_runs_igds.append(igds)
        all_runs_sps.append(sps)
        all_runs_hvs.append(hvs)

        item["final_igd_plus"] = float(igds[-1]) if igds else float("inf")
        item["final_spacing"] = float(sps[-1]) if sps else float("inf")
        item["final_hv_norm"] = float(hvs[-1]) if hvs else 0.0

    best_item = max(cache_runs, key=lambda x: x.get("final_hv_norm", 0.0))
    best_run_idx = best_item["run_id"]
    print(f"\n[Best Run Selection] by final HV_norm: Run={best_run_idx}, HV_norm={best_item['final_hv_norm']:.4f}, "
          f"IGD+={best_item['final_igd_plus']:.4f}, SP={best_item['final_spacing']:.4f}")

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

    save_mutation_stats_and_plots(best_item["mutation_tracker"], csv_name="mutation_stats_best_run.csv")

    best_pareto = best_item["pareto"]
    best_batches = best_item["batches"]

    plot_pareto_evolution_3d(best_h_fronts)
    save_pareto_solutions(best_pareto, best_batches, filename="result.txt")

    df_sum = pd.DataFrame(all_run_summaries)
    df_sum.to_csv("timing_summary_runs.csv", index=False)
    print("[Saved] timing_summary_runs.csv")

    total_all_time = time.perf_counter() - t_all0
    print(f"\n[TOTAL] All runs wall time = {total_all_time:.2f} seconds")
    print("\nAll done.\n")
