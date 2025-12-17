#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========================
# 全局设置
# ========================

TIME_BUCKET_H = 1.0  # 仿真时间粒度：1小时
CHINA_REGIONS = {"CN", "China"}
EUROPE_REGIONS = {"WE", "EE", "EU", "Europe"}
BIG_M = 1e6  # 容量惩罚的大常数

# ---- Mutation roulette weights (you can tune)
W_ADD = 0.25
W_DEL = 0.20
W_MOD = 0.35
W_MODE = 0.20


# ========================
# 1. 数据结构定义
# ========================

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float       # 单位：TEU / TimeBucket (例如 TEU/小时)
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

    # 忽略 ID，只比较结构
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

    # 关键：自定义 __eq__，基于内容比较，而非实例
    def __eq__(self, other):
        if not isinstance(other, PathAllocation):
            return NotImplemented
        # 比较 Path 对象（使用 Path 的 __eq__）和 share（考虑浮点误差）
        return (self.path == other.path and
                abs(self.share - other.share) < 1e-10)

    # 关键：自定义 __hash__，使对象可哈希，用于集合和字典键
    def __hash__(self):
        share_for_hash = round(self.share, 10)  # 限制精度以保持一致性
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
    # (origin, dest, batch_id)
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    # Objectives: Cost, Emission, Time
    objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    # 约束相关
    penalty: float = 0.0
    feasible: bool = False


# ========================
# 2. 核心逻辑：去重与归一化
# ========================

def clone_gene(alloc: PathAllocation) -> PathAllocation:
    return PathAllocation(path=alloc.path, share=alloc.share)


def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs:
        return []

    # 合并相同结构的路径
    merged_map: Dict[Path, float] = {}
    for a in allocs:
        if a.path not in merged_map:
            merged_map[a.path] = a.share
        else:
            merged_map[a.path] += a.share

    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged_map.items()]

    # 归一化
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

    # 过滤微小份额
    unique_allocs = [a for a in unique_allocs if a.share > 0.001]

    # 二次修正，保证总和 ≈ 1
    if unique_allocs:
        final_total = sum(a.share for a in unique_allocs)
        if abs(final_total - 1.0) > 1e-6:
            for a in unique_allocs:
                a.share /= final_total

    return unique_allocs


# ========================
# 3. 数据读取与图构建
# ========================

def load_network_from_extended(filename: str):
    try:
        xls = pd.ExcelFile(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit(1)

    # Nodes
    nodes_df = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).tolist()
    node_region = dict(zip(nodes_df["EnglishName"].astype(str),
                           nodes_df["Region"].astype(str)))

    # Arcs
    arcs_df = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []

    # 假设 Excel 中的 Capacity 是 TEU/Day，需要转换成 TEU/TimeBucket
    DAILY_HOURS = 24.0

    for _, row in arcs_df.iterrows():
        mode_raw = str(row["Mode"]).strip().lower()
        speed = 75.0 if mode_raw == "road" else (30.0 if mode_raw == "water" else 50.0)
        mode = "rail" if mode_raw == "rail" else mode_raw

        dist_str = str(row["Distance_km"])
        cleaned = "".join(ch for ch in dist_str if (ch.isdigit() or ch == "."))
        distance = float(cleaned) if cleaned else 0.0

        # 容量：从 TEU/Day 转换为 TEU/TimeBucket
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
    """将时刻表预处理为字典，加速查询。"""
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
            path_lib[od] = paths_for_od[:20]  # 保留前20条路径

    return path_lib


# ========================
# 4. 评估逻辑（目标 + 可行性检查）
# ========================

def simulate_path_time_capacity(path: Path, batch: Batch, flow: float, tt_dict,
                                arc_flow_map) -> float:
    """
    按批次ET出发，沿path仿真时间与容量占用。
    - Road: 随到随走 (不看时刻表)
    - Rail / Water: 按 Timetable 等下一班
    """
    t = batch.ET
    for arc in path.arcs:
        travel_time = arc.distance / max(arc.speed_kmh, 1.0)

        # 公路：强制不看时刻表，随到随走
        if arc.mode == "road":
            entries = []
        else:
            key = (arc.from_node, arc.to_node, arc.mode)
            entries = tt_dict.get(key, [])

        if not entries:
            dep = t
        else:
            # 简化：只用第一条时刻信息
            e = entries[0]
            if t <= e.first_departure_hour:
                dep = e.first_departure_hour
            else:
                waited = (t - e.first_departure_hour)
                n = math.ceil(waited / e.headway_hours)
                dep = e.first_departure_hour + n * e.headway_hours

        arr = dep + travel_time

        # 记录容量占用（以出发时间所在的 Time Bucket 为粒度）
        start_slot = int(dep)
        key = (arc.from_node, arc.to_node, arc.mode)
        slot_key = (key, start_slot)
        arc_flow_map[slot_key] = arc_flow_map.get(slot_key, 0) + flow

        t = arr

    # 返回总旅行时间（相对于ET）
    return t - batch.ET


def evaluate_individual(ind: Individual, batches, path_lib, arcs, tt_dict):
    total_cost = 0.0
    total_emission = 0.0
    makespan = 0.0

    arc_flow_map: Dict[Tuple[Tuple[str, str, str], int], float] = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    penalty = 0.0

    # 约束违背标记
    missing_alloc_violation = False
    late_violation = False
    capacity_violation = False

    for batch in batches:
        # 按 (O,D,batch_id) 取编码
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])

        if not allocs:
            # 没有为该批次分配路径 → 严重不可行
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

            # 软时间窗惩罚：这里也当成“可行性条件”
            if arrival_time > batch.LT:
                penalty += (arrival_time - batch.LT) * 1000.0
                late_violation = True

        makespan = max(makespan, batch_finish_time)

    # 容量约束惩罚：按（弧, 时间桶）检查
    for (key, slot), flow in arc_flow_map.items():
        cap = arc_caps.get(key, 1e9)
        if flow > cap:
            penalty += (flow - cap) * BIG_M
            capacity_violation = True

    ind.objectives = (total_cost + penalty,
                      total_emission + penalty,
                      makespan)
    ind.penalty = penalty
    # 所有约束都满足才算可行
    ind.feasible = not (missing_alloc_violation or late_violation or capacity_violation)


# ========================
# 5. 遗传算子
# ========================

def crossover_structural(ind1: Individual, ind2: Individual,
                         batches: List[Batch]) -> Tuple[Individual, Individual]:
    """结构+份额的片段交叉，交叉后用 merge_and_normalize 修复 share。
       现在以 (origin,destination,batch_id) 为 key，对每个批次单独交叉。
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


def crossover(ind1: 'Individual', ind2: 'Individual',
              batches: List['Batch']) -> Tuple['Individual', 'Individual']:
    """
    针对单个批次进行交叉，其他批次直接继承。
    策略：
    1. 随机打乱所有批次，遍历寻找可交叉的批次
    2. 找到第一个有共同节点的批次，进行单点交叉
    3. 如果找不到，随机选择一个批次进行路径交换
    4. 其余批次直接复制父代
    """
    # 创建子代，初始复制父代所有分配
    child1 = deepcopy(ind1)
    child2 = deepcopy(ind2)

    # 获取所有批次键并随机打乱
    batch_keys = [(batch.origin, batch.destination, batch.batch_id)
                  for batch in batches]
    random.shuffle(batch_keys)

    crossed = False  # 标记是否已执行交叉

    for key in batch_keys:
        # 获取该批次的路径分配
        allocs1 = ind1.od_allocations.get(key, [])
        allocs2 = ind2.od_allocations.get(key, [])

        # 检查是否有路径可交叉
        if not allocs1 or not allocs2:
            continue  # 至少一个个体没有该批次的分配

        # 检查是否基因完全相同（如果是则跳过）
        if are_individuals_identical(allocs1, allocs2):
            continue  # 基因相同，无法产生新基因

        # 寻找有共同中间节点的路径对
        common_node_candidates = []

        # 遍历所有路径组合
        for alloc1 in allocs1:
            for alloc2 in allocs2:
                common_nodes = find_common_intermediate_nodes(
                    alloc1.path.nodes,
                    alloc2.path.nodes
                )

                if common_nodes:
                    for node in common_nodes:
                        common_node_candidates.append({
                            'alloc1': alloc1,
                            'alloc2': alloc2,
                            'common_node': node
                        })

        # 如果找到共同节点，执行交叉
        if common_node_candidates:
            candidate = random.choice(common_node_candidates)

            new_alloc1, new_alloc2 = perform_single_point_crossover(
                candidate['alloc1'],
                candidate['alloc2'],
                candidate['common_node']
            )

            idx1 = find_allocation_index(allocs1, candidate['alloc1'])
            idx2 = find_allocation_index(allocs2, candidate['alloc2'])

            if idx1 is not None and idx2 is not None:
                new_allocs1 = deepcopy(allocs1)
                new_allocs2 = deepcopy(allocs2)

                new_allocs1[idx1] = new_alloc1
                new_allocs2[idx2] = new_alloc2

                new_allocs1[idx1] = basic_path_repair(new_allocs1[idx1])
                new_allocs2[idx2] = basic_path_repair(new_allocs2[idx2])

                child1.od_allocations[key] = new_allocs1
                child2.od_allocations[key] = new_allocs2

                print(f"✅ 批次 {key} 基于节点 '{candidate['common_node']}' 交叉")
                crossed = True
                break  # 只交叉一个批次

    if not crossed:
        if batch_keys:
            random_key = random.choice(batch_keys)
            if (random_key in child1.od_allocations and
                    random_key in child2.od_allocations):
                child1.od_allocations[random_key], child2.od_allocations[random_key] = \
                    child2.od_allocations[random_key], child1.od_allocations[random_key]
                print(f"⚠️  无共同节点，随机交换批次 {random_key}")
                crossed = True

    return child1, child2


def are_individuals_identical(genes1, genes2) -> bool:
    if len(genes1) != len(genes2):
        return False
    return all(g1 == g2 for g1, g2 in zip(genes1, genes2))


def find_allocation_index(allocations: List['PathAllocation'],
                          target: 'PathAllocation') -> int:
    for i, alloc in enumerate(allocations):
        if alloc == target:
            return i
    return None


def find_common_intermediate_nodes(path1_nodes, path2_nodes):
    """找到两条路径的非端点共同节点"""
    if not path1_nodes or not path2_nodes:
        return []

    intermediate1 = set(path1_nodes[1:-1])
    intermediate2 = set(path2_nodes[1:-1])
    return list(intermediate1.intersection(intermediate2))


def perform_single_point_crossover(alloc1, alloc2, common_node):
    """在共同节点处执行单点交叉"""
    path1 = alloc1.path
    path2 = alloc2.path

    idx1 = path1.nodes.index(common_node)
    idx2 = path2.nodes.index(common_node)

    new_nodes1 = path1.nodes[:idx1] + path2.nodes[idx2:]
    new_nodes2 = path2.nodes[:idx2] + path1.nodes[idx1:]

    new_modes1 = path1.modes[:idx1] + path2.modes[idx2:]
    new_modes2 = path2.modes[:idx2] + path1.modes[idx1:]

    new_path1 = Path(
        path_id=-1,
        origin=path1.origin,
        destination=path1.destination,
        nodes=new_nodes1,
        modes=new_modes1,
        arcs=[],
        base_cost_per_teu=0.0,
        base_emission_per_teu=0.0,
        base_travel_time_h=0.0
    )

    new_path2 = Path(
        path_id=-1,
        origin=path2.origin,
        destination=path2.destination,
        nodes=new_nodes2,
        modes=new_modes2,
        arcs=[],
        base_cost_per_teu=0.0,
        base_emission_per_teu=0.0,
        base_travel_time_h=0.0
    )

    return (PathAllocation(path=new_path1, share=alloc1.share),
            PathAllocation(path=new_path2, share=alloc2.share))


def basic_path_repair(alloc):
    """基础路径修复：去重和确保端点正确"""
    nodes = alloc.path.nodes
    modes = alloc.path.modes

    seen = set()
    unique_nodes = []
    unique_modes = []

    for i, node in enumerate(nodes):
        if node not in seen:
            seen.add(node)
            unique_nodes.append(node)
            if i < len(modes):
                unique_modes.append(modes[i])

    if unique_nodes[0] != alloc.path.origin:
        unique_nodes.insert(0, alloc.path.origin)
        if unique_modes:
            unique_modes.insert(0, unique_modes[0])

    if unique_nodes[-1] != alloc.path.destination:
        unique_nodes.append(alloc.path.destination)
        if unique_modes:
            unique_modes.append(unique_modes[-1])

    repaired_path = Path(
        path_id=alloc.path.path_id,
        origin=alloc.path.origin,
        destination=alloc.path.destination,
        nodes=unique_nodes,
        modes=unique_modes,
        arcs=alloc.path.arcs,
        base_cost_per_teu=alloc.path.base_cost_per_teu,
        base_emission_per_teu=alloc.path.base_emission_per_teu,
        base_travel_time_h=alloc.path.base_travel_time_h
    )

    return PathAllocation(path=repaired_path, share=alloc.share)


# ========================
# 5.1 变异（四算子轮盘赌）——仅改这一块
# ========================

# lazy caches built from path_lib (避免改动其它模块接口)
_ARC_LOOKUP: Dict[Tuple[str, str, str], Arc] = {}
_ARC_MODE_OPTIONS: Dict[Tuple[str, str], List[str]] = {}
_CACHE_BUILT = False


def _build_arc_caches_from_path_lib(path_lib: Dict[Tuple[str, str], List[Path]]):
    """从 path_lib 扫描可用弧，构建：
    - (u,v,mode)->Arc lookup
    - (u,v)->list of available modes
    """
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
    """只改一个路径中的一个弧的 mode，并替换 arc 对象、重算 base_*。"""
    if not allocs:
        return allocs

    # pick one allocation
    a = random.choice(allocs)
    p = a.path
    if not p.arcs or len(p.arcs) == 0:
        return allocs

    # pick one arc position
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

    # IMPORTANT: do NOT mutate Path in-place (hash changes) -> create a new Path
    new_arcs = list(p.arcs)
    new_arcs[pos] = new_arc
    new_modes = list(p.modes)
    if pos < len(new_modes):
        new_modes[pos] = new_mode
    else:
        # safety (should not happen if modes aligns)
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

    # replace in allocs (preserve share)
    for i in range(len(allocs)):
        if allocs[i] == a:
            allocs[i] = PathAllocation(path=new_path, share=a.share)
            break

    return merge_and_normalize(allocs)


def mutate_structural(ind: Individual, batches: List[Batch],
                      path_lib: Dict[Tuple[str, str], List[Path]],
                      p_add=0.2, p_del=0.3, p_mod=0.5):
    """
    ✅ 你原来的 mutate_structural 被替换为“四算子轮盘赌”版本：
    - add / del / mod-share / mode-change(单段)
    仍然：随机选一个批次，对该批次编码做变异，最后 merge_and_normalize 修复 share。
    注意：删除路径在 allocs 仅有1条时禁用（你要求的）。
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

    # ---- roulette weights (del disabled if not applicable)
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
# 6. 工具类：Hypervolume 计算 & 解去重
# ========================

class HypervolumeCalculator:
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
        """
        返回相对于 ref_point 定义的“归一化超体积”（0–1）。
        """
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


# ========================
# 6.1 统计/计时结构（新增）
# ========================

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


# ========================
# 7. NSGA-II 逻辑 (带可行性 & 记录)
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
        raw_allocs = [PathAllocation(path=p, share=random.random())
                      for p in chosen]

        key = (batch.origin, batch.destination, batch.batch_id)
        ind.od_allocations[key] = merge_and_normalize(raw_allocs)
    return ind


def dominates(a: Individual, b: Individual) -> bool:
    """
    约束支配：
    - 可行解总是优于不可行解；
    - 两者同为可行或不可行时，回到标准多目标支配关系。
    """
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
                        pop_size=50,
                        generations=100):
    stats = RunStats()
    t_run0 = time.perf_counter()

    print("Loading data...")
    node_names, arcs, timetables, batches = load_network_from_extended(filename)

    print("Pre-processing timetable...")
    tt_dict = build_timetable_dict(timetables)

    print("Building path library...")
    path_lib = build_path_library(node_names, arcs, batches, timetables)

    # 1. 初始种群
    t_init0 = time.perf_counter()
    population: List[Individual] = []
    for _ in range(pop_size):
        ind = random_initial_individual(batches, path_lib)

        t0 = time.perf_counter()
        evaluate_individual(ind, batches, path_lib, arcs, tt_dict)
        dt = time.perf_counter() - t0
        stats.evaluation_time += dt
        stats.evaluation_calls += 1
        if dt > stats.best_crossover_time:
            stats.eval_slower_than_best_crossover += 1

        population.append(ind)
        init_feasible = sum(1 for ind in population if ind.feasible)
        print(f"[INIT] FeasiblePop={init_feasible}/{len(population)}")

    stats.init_time = time.perf_counter() - t_init0

    # 2. HV 参考点：根据初始种群的最大值放大 1.2 倍
    all_objs = np.array([ind.objectives for ind in population], dtype=float)
    max_vals = np.max(all_objs, axis=0)
    ref_point = max_vals * 1.2
    hv_calc = HypervolumeCalculator(ref_point=tuple(ref_point),
                                    num_samples=10000)
    print(f"Reference Point for HV set to: {ref_point}")

    history_fronts: List[Tuple[int, List[Tuple[float, float, float]]]] = []
    hv_history: List[float] = []

    for gen in range(generations):
        t_gen0 = time.perf_counter()

        # ---- NDSort timing
        t0 = time.perf_counter()
        fronts = non_dominated_sort(population)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        front0 = fronts[0]

        # (1) Front0 可行解数
        feasible_front0 = [ind for ind in front0 if ind.feasible]
        base_front = feasible_front0 if feasible_front0 else front0

# (2) Front0 去重后的解数
        front0_unique = unique_individuals_by_objectives(base_front, tol=1e-3)

        current_front_objs = [ind.objectives for ind in front0_unique]
        history_fronts.append((gen, current_front_objs))

        t0 = time.perf_counter()
        current_hv = hv_calc.calculate(front0_unique)
        dt = time.perf_counter() - t0
        stats.hv_time += dt
        stats.hv_calls += 1

        hv_history.append(current_hv)

# (3) 种群整体可行解数
        feasible_pop = sum(1 for ind in population if ind.feasible)

        best_f1 = min(obj[0] for obj in current_front_objs)
        print(
            f"Gen {gen}: Best Cost={best_f1:.0f}, "
            f"Front0 Unique={len(front0_unique)}, "
            f"FeasibleInF0={len(feasible_front0)}, "
            f"FeasiblePop={feasible_pop}/{len(population)}, "
            f"HV={current_hv:.4f}"
        )

        # ranks
        ranks: Dict[Individual, int] = {}
        for r, front in enumerate(fronts):
            for ind in front:
                ranks[ind] = r

        # crowding timing (over all fronts)
        t0 = time.perf_counter()
        dists: Dict[Individual, float] = {}
        for front in fronts:
            dists.update(crowding_distance(front))
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

            # evaluation
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

        combined = population + offspring

        t0 = time.perf_counter()
        fronts = non_dominated_sort(combined)
        dt = time.perf_counter() - t0
        stats.ndsort_time += dt
        stats.ndsort_calls += 1

        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                t0 = time.perf_counter()
                d = crowding_distance(front)
                dt = time.perf_counter() - t0
                stats.crowd_time += dt
                stats.crowd_calls += 1

                front.sort(key=lambda x: d[x], reverse=True)
                new_pop.extend(front[:pop_size - len(new_pop)])
                break
        population = new_pop

        gen_total_time = time.perf_counter() - t_gen0
        stats.add_gen_record({
            "gen": gen,
            "front0_unique": int(len(front0_unique)),
            "feasible_in_front0": int(len(feasible_front0)),
            "hv": float(current_hv),
            "gen_total_time_s": float(gen_total_time),
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
    pareto_front = unique_individuals_by_objectives(base_front, tol=1e-3)

    stats.run_total_time = time.perf_counter() - t_run0

    print("\n========== TIMING SUMMARY (this run) ==========")
    s = stats.summary_dict()
    for k, v in s.items():
        print(f"{k}: {v}")
    print(f"Final Pareto size (unique, prefer feasible): {len(pareto_front)}")
    print("=============================================\n")
    print(f"[FINAL] Pareto size (unique, feasible-first) = {len(pareto_front)}")

    return population, pareto_front, batches, history_fronts, hv_history, stats


# ========================
# 8. 绘图与输出
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


def save_pareto_solutions(pareto: List[Individual],
                          batches: List[Batch],
                          filename: str = "result.txt"):
    pareto = unique_individuals_by_objectives(pareto, tol=1e-3)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Pareto Solutions (All, Unique & Mostly Feasible) =====\n\n")
        for i, ind in enumerate(pareto):
            cost, emit, time_ = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Objectives: Cost={cost:.2f}, "
                    f"Emission={emit:.2f}, Time={time_:.2f}, "
                    f"Penalty={ind.penalty:.2f}, Feasible={ind.feasible}\n\n")

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


def plot_results(history_fronts, hv_history):
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    num_gens = len(history_fronts)

    colors = cm.viridis(np.linspace(0, 1, num_gens))

    for gen, objs in history_fronts:
        if not objs:
            continue
        xs = [o[0] for o in objs]
        ys = [o[1] for o in objs]
        zs = [o[2] for o in objs]

        alpha = 0.2 + 0.8 * (gen / max(1, num_gens - 1))
        s = 20 + 30 * (gen / max(1, num_gens - 1))

        label = f"Gen {gen}" if gen in [0, num_gens - 1] else ""
        ax1.scatter(xs, ys, zs,
                    color=colors[gen],
                    alpha=alpha,
                    s=s,
                    label=label)

    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Emission (kg)')
    ax1.set_zlabel('Time (h)')
    ax1.set_title('Pareto Front Evolution (Color=Generation)')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(hv_history)),
             hv_history,
             marker='o',
             linestyle='-',
             markersize=4)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Hypervolume (0–1, Approx)')
    ax2.set_title('Convergence Metric (Hypervolume)')
    ax2.grid(True)

    plt.tight_layout()

    ax1.view_init(elev=30, azim=45)
    plt.savefig('nsga2_view_1_std.png', dpi=300)
    print("Saved view 1 (Standard): nsga2_view_1_std.png")

    ax1.view_init(elev=20, azim=135)
    plt.savefig('nsga2_view_2_side.png', dpi=300)
    print("Saved view 2 (Side): nsga2_view_2_side.png")

    ax1.view_init(elev=60, azim=45)
    plt.savefig('nsga2_view_3_top.png', dpi=300)
    print("Saved view 3 (Top-down): nsga2_view_3_top.png")

    ax1.view_init(elev=25, azim=210)
    plt.savefig('nsga2_view_4_other.png', dpi=300)
    print("Saved view 4 (Other): nsga2_view_4_other.png")

    plt.show()


# ========================
# 9. 典型 Pareto 解的可视化相关函数
# ========================

def normalise_objectives(pareto: List[Individual]):
    objs = np.array([ind.objectives for ind in pareto], dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (objs - mins) / ranges
    return norm, mins, maxs


def select_weighted_solution(pareto: List[Individual],
                             weights=(0.4, 0.3, 0.3)) -> Tuple[Individual, float]:
    norm_objs, mins, maxs = normalise_objectives(pareto)
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    scores = np.dot(norm_objs, w)
    idx = int(np.argmin(scores))
    return pareto[idx], float(scores[idx])


def analyse_solution_modes(ind: Individual, batches: List[Batch]) -> Dict[str, float]:
    mode_flow: Dict[str, float] = {}
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])
        for a in allocs:
            flow = a.share * batch.quantity
            for arc in a.path.arcs:
                k = arc.mode
                mode_flow[k] = mode_flow.get(k, 0.0) + flow * arc.distance
    return mode_flow


def plot_mode_share(ind: Individual, batches: List[Batch],
                    filename: str = "typical_solution_mode_share.png"):
    mode_flow = analyse_solution_modes(ind, batches)
    modes = list(mode_flow.keys())
    values = [mode_flow[m] for m in modes]

    plt.figure()
    plt.bar(modes, values)
    plt.xlabel("Mode")
    plt.ylabel("TEU·km in this solution")
    plt.title("Mode usage of selected Pareto solution")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved mode share figure: {filename}")


def plot_radar_for_solution(ind: Individual, pareto: List[Individual],
                            filename: str = "typical_solution_radar.png"):
    norm_objs, mins, maxs = normalise_objectives(pareto)
    objs = np.array([p.objectives for p in pareto], dtype=float)
    target = np.array(ind.objectives, dtype=float)
    idx = int(np.argmin(np.linalg.norm(objs - target, axis=1)))
    val = norm_objs[idx]

    labels = ["Cost", "Emission", "Time"]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)
    vals = np.concatenate([val, [val[0]]])

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Normalised objectives of selected solution")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved radar figure: {filename}")


# ========================
# 10. 多次实验：至少 30 组取平均
# ========================

if __name__ == "__main__":
    filename = "data.xlsx"
    pop_size = 60
    generations = 200
    runs = 30

    all_final_hv: List[float] = []
    all_run_summaries: List[Dict] = []
    t_all0 = time.perf_counter()

    best_run_idx = -1
    best_run_hv = -1.0

    best_pareto = None
    best_batches = None
    best_history_fronts = None
    best_hv_history = None
    best_stats = None

    for run_id in range(runs):
        seed = 1000 + run_id
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n========== RUN {run_id} / {runs-1}, seed={seed} ==========\n")
        pop, pareto, batches, h_fronts, h_hv, stats = run_nsga2_analytics(
            filename=filename,
            pop_size=pop_size,
            generations=generations
        )

        final_hv = h_hv[-1]
        all_final_hv.append(final_hv)
        print(f"[RUN {run_id}] Final HV = {final_hv:.4f}, "
              f"Pareto size = {len(pareto)}")

        run_summary = stats.summary_dict()
        run_summary["run_id"] = run_id
        run_summary["seed"] = seed
        run_summary["final_hv"] = float(final_hv)
        run_summary["final_pareto_size"] = int(len(pareto))
        run_summary["eval_over_crossover_ratio"] = float(
            (stats.evaluation_time / stats.crossover_time) if stats.crossover_time > 1e-12 else float("inf")
        )
        all_run_summaries.append(run_summary)

        print(f"[RUN {run_id}] Time total={stats.run_total_time:.2f}s | "
              f"eval={stats.evaluation_time:.2f}s | cross={stats.crossover_time:.2f}s | "
              f"mut={stats.mutation_time:.2f}s | hv={stats.hv_time:.2f}s | "
              f"pareto={len(pareto)}")

        if final_hv > best_run_hv:
            best_run_hv = final_hv
            best_run_idx = run_id
            best_pareto = pareto
            best_batches = batches
            best_history_fronts = h_fronts
            best_hv_history = h_hv
            best_stats = stats

    hv_mean = float(np.mean(all_final_hv))
    hv_std = float(np.std(all_final_hv))

    print("\n=========== SUMMARY OVER RUNS ===========")
    print(f"Runs           : {runs}")
    print(f"Final HV mean  : {hv_mean:.4f}")
    print(f"Final HV std   : {hv_std:.4f}")
    print(f"Best run index : {best_run_idx}")
    print(f"Best run HV    : {best_run_hv:.4f}")
    print("=========================================\n")

    total_all_time = time.perf_counter() - t_all0
    print(f"[TOTAL] All runs wall time = {total_all_time:.2f} seconds")

    df_sum = pd.DataFrame(all_run_summaries)
    df_sum.to_csv("timing_summary_runs.csv", index=False)
    print("[Saved] timing_summary_runs.csv")

    if best_pareto is not None:
        for i, ind in enumerate(best_pareto[:3]):
            print_pure_structure(ind, best_batches, f"BestRun Pareto Sol {i}")

        save_pareto_solutions(best_pareto, best_batches, filename="result.txt")

        plot_results(best_history_fronts, best_hv_history)

        typical_ind, typical_score = select_weighted_solution(
            best_pareto, weights=(0.4, 0.3, 0.3)
        )
        print("\n=== Selected 'Typical' Solution (w = 0.4, 0.3, 0.3) ===")
        print("Objectives:", typical_ind.objectives)
        print("Weighted score (normalised):", typical_score)
        print_pure_structure(typical_ind, best_batches, "Typical Solution")

        plot_mode_share(typical_ind, best_batches,
                        filename="typical_solution_mode_share.png")

        plot_radar_for_solution(typical_ind, best_pareto,
                                filename="typical_solution_radar.png")
