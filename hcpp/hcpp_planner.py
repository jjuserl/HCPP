"""
HCPP: 分层覆盖路径规划 (Hierarchy Coverage Path Planning)
     带主动极值预防 (Proactive Extremum Prevention)

严格遵循论文 "Hierarchy Coverage Path Planning With Proactive
Extremum Prevention in Unknown Environments" (IEEE RAL, 2025) 实现。

核心模块:
  (1) 任务管理 (Task Management) - Boustrophedon 单元分解 + 动态更新
  (2) 全局巡游规划 (GTP) - 邻接图 + 顶点分类(边缘/内部) + brim-first 巡游
  (3) 局部路径规划 (LPP) - 沿未覆盖区域边界移动，保持连通性
  (4) HCPP 主循环 (Algorithm 4) - 感知→更新CTS→GTP→LPP→执行→循环
"""

import numpy as np
from collections import deque
import heapq
import math
from .map_grid import GridMap


# =============================================================================
# 覆盖任务结构 (CTS: Coverage Task Structure)
# =============================================================================

class CoverageTask:
    """
    覆盖单元的四元组: {ID, px, pu, pd, adj}

    论文定义:
      ID  - 单元唯一标识
      px  - 单元所在列 x 坐标
      pu  - 单元上端点 y 坐标
      pd  - 单元下端点 y 坐标
      adj - 相邻单元 ID 列表 (共享公共边界)
    """
    __slots__ = ('id', 'px', 'pu', 'pd', 'adj', 'completed',
                 'visited_segments', 'split_flag',
                 '_obstructed_segments')

    def __init__(self, task_id, px, pu, pd, adj=None):
        self.id = task_id
        self.px = px
        self.pu = pu          # upper y
        self.pd = pd          # lower y
        self.adj = adj if adj is not None else []
        self.completed = False
        # 跟踪机器人实际访问过的该列 y 范围 [(y_start, y_end), ...]
        self.visited_segments = []
        self.split_flag = False  # 标记是否已被拆分
        # 该列上被障碍物占据的 y 范围（不需要覆盖）
        self._obstructed_segments = []

    def __repr__(self):
        return (f"CT(id={self.id}, px={self.px}, "
                f"pu={self.pu}, pd={self.pd}, adj={self.adj}, "
                f"done={self.completed})")

    @property
    def upper_endpoint(self):
        return (self.px, self.pu)

    @property
    def lower_endpoint(self):
        return (self.px, self.pd)

    @property
    def length(self):
        return abs(self.pu - self.pd)

    def y_range(self):
        return (min(self.pu, self.pd), max(self.pu, self.pd))

    def mark_visited_y(self, gy):
        """标记该列 y 坐标已被机器人访问"""
        y_min, y_max = self.y_range()
        if gy < y_min or gy > y_max:
            return
        # 扩展已有段或创建新段
        merged = False
        for i, (s, e) in enumerate(self.visited_segments):
            if s - 1 <= gy <= e + 1:
                self.visited_segments[i] = (min(s, gy), max(e, gy))
                merged = True
                break
        if not merged:
            self.visited_segments.append((gy, gy))
        # 合并重叠段
        self._merge_segments()
        # 检查是否完成
        self._check_completion()

    def _merge_segments(self):
        if len(self.visited_segments) <= 1:
            return
        sorted_segs = sorted(self.visited_segments)
        merged = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            last = merged[-1]
            if seg[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], seg[1]))
            else:
                merged.append(seg)
        self.visited_segments = merged

    def _check_completion(self):
        """当所有非障碍物的该列 y 都被访问过时，标记完成 (100% 覆盖)"""
        y_min, y_max = self.y_range()
        covered_y = 0
        for s, e in self.visited_segments:
            cs = max(s, y_min)
            ce = min(e, y_max)
            if ce >= cs:
                covered_y += (ce - cs + 1)
        total_y = y_max - y_min + 1
        # 排除障碍物占据的 y 坐标（这些不可能被覆盖）
        obstructed_y = 0
        if hasattr(self, '_obstructed_segments'):
            for os_s, os_e in self._obstructed_segments:
                obs_s = max(os_s, y_min)
                obs_e = min(os_e, y_max)
                if obs_e >= obs_s:
                    obstructed_y += (obs_e - obs_s + 1)
        effective_total = total_y - obstructed_y
        if effective_total <= 0:
            self.completed = True
        else:
            self.completed = (covered_y >= effective_total)

    def get_uncovered_segments(self):
        """返回单元内未覆盖的 y 范围列表 [(start, end), ...]"""
        y_min, y_max = self.y_range()
        if not self.visited_segments:
            return [(y_min, y_max)]
        gaps = []
        cur = y_min
        for s, e in sorted(self.visited_segments):
            if s > cur:
                gaps.append((cur, s - 1))
            cur = max(cur, e + 1)
        if cur <= y_max:
            gaps.append((cur, y_max))
        return gaps


# =============================================================================
# HCPP 规划器主类
# =============================================================================

class HCPPPlanner:
    """
    HCPP 规划器: 分层覆盖路径规划与主动极值预防

    严格遵循论文 Algorithm 1-4 实现。
    """

    def __init__(self, grid_map, global_map, sensor):
        self.grid_map = grid_map        # 机器人的认知地图
        self.global_map = global_map    # 全局真实地图（用于传感器仿真）
        self.sensor = sensor
        # 传感器覆盖半径 (格数), sensor里面没有用范围了（5*5），所以这里直接设置为2
        self.r = 2
        # self.r = int(sensor.range_max / grid_map.resolution)

        self.rx = self.ry = 0
        self.path = []
        self.extremum_count = 0
        self.comp_time = 0.0

        # CTS
        self.cts = []
        self._next_id = 0
        self._task_id_map = {}      # O(1) task lookup by id
        self._col_to_tasks = {}     # O(1) tasks lookup by column x
        self._cached_adj_dict = None  # cached adjacency graph
        self._global_tour = []
        self._local_path = []
        self._local_path_idx = 0
        self._last_completed_task_id = None

        self._step_count = 0
        self._initialized = False

        # 边界探索状态
        self._boundary_explored = False
        self._first_global = False
        self._boundary_path = []
        self._boundary_path_idx = 0
        self._in_boundary_explore = False

        # 绕障碍状态
        self._in_go_around = False
        self._go_around_path = []
        self._go_around_idx = 0
        self._explored_obstacle_keys = set()

        # 已访问集合 (用于覆盖判断)
        self._visited_cells = set()

        # 增量变化字典: {列x: set(y坐标)} —— 记录状态变为 OBSTACLE/COVERED 的格子
        self.changed_cells: dict[int, set[int]] = {}
        # 仅记录新障碍物涉及的列，供增量更新使用
        self._obstacle_changed_xs: set[int] = set()
        # 注册网格变化回调
        self.grid_map.on_cell_changed = self._on_cell_changed

        # 方向定义: 上右下左 (4邻域)
        self._dirs_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        # 8邻域
        self._dirs_8 = [(0, -1), (1, -1), (1, 0), (1, 1),
                        (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    # -------------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------------

    def initialize(self, sx, sy):
        """初始化机器人起始位置并执行初始感知，边界探索将在step中触发"""
        self.rx, self.ry = sx, sy
        self.grid_map.set_free(sx, sy)
        self.grid_map.set_covered(sx, sy)
        self.path = [(sx, sy)]
        self._visited_cells = {(sx, sy)}
        self._initialized = True
        self._boundary_explored = False
        self._in_boundary_explore = False
        self._in_go_around = False
        self._new_obstacle_flag = False
        self._explored_obstacle_keys = set()

        # 初始感知 (仅发现，不标记覆盖)
        self._sense()

        # 初始单元分解 + GTP 将在边界探索之后进行
        # (in _do_boundary_exploration completion)
        self.cts = []
        self._global_tour = []
        self._local_path = []
        self._local_path_idx = 0
        self._last_completed_task_id = None

    def _on_cell_changed(self, gx, gy, new_state):
        """网格状态变化回调: 跟踪 OBSTACLE/COVERED 变化，单独记录障碍物列"""
        if new_state in (GridMap.OBSTACLE, GridMap.COVERED):
            if gx not in self.changed_cells:
                self.changed_cells[gx] = set()
            self.changed_cells[gx].add(gy)
        if new_state == GridMap.OBSTACLE:
            self._obstacle_changed_xs.add(gx)
            self._new_obstacle_flag = True  # 增量标记: 有新障碍物

    # 四邻域偏移常量 (避免每次调用创建新列表)
    _N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def _sense(self):
        """
        传感器感知: 使用带遮挡检测的 simple_sense
        感知后处理: 被障碍物包围的格子自动赋值为障碍物
        优化: 通过 changed_cells 回调判断是否有新发现，无变化时跳过包围检测
        """
        gm = self.grid_map
        rx, ry = self.rx, self.ry

        # 记录感知前 changed_cells 的大小，用于判断是否有新发现
        changes_before = len(self.changed_cells)
        self.sensor.simple_sense(gm, self.global_map, rx, ry)
        changes_after = len(self.changed_cells)

        # 若无新格子被改变状态，跳过包围检测
        if changes_after == changes_before:
            return

        # 有新发现时才执行包围检测: 四邻域全是障碍/越界 → 标为障碍物
        r = self.r + 1
        w, h = gm.width, gm.height
        grid = gm.grid
        for dy in range(-r, r + 1):
            gy = ry + dy
            if gy < 0 or gy >= h:
                continue
            row = grid[gy]
            for dx in range(-r, r + 1):
                gx = rx + dx
                if gx < 0 or gx >= w:
                    continue
                s = row[gx]
                if s != GridMap.FREE and s != GridMap.UNKNOWN:
                    continue
                # 检查四邻域是否全是障碍或越界
                all_blocked = True
                for ndx, ndy in self._N4:
                    nx, ny = gx + ndx, gy + ndy
                    if gm.is_valid(nx, ny) and gm.grid[ny, nx] != GridMap.OBSTACLE:
                        all_blocked = False
                        break
                if all_blocked:
                    gm.set_cell(gx, gy, GridMap.OBSTACLE)

    def _cover_nearby(self):
        """
        标记当前格子为已覆盖 (单格工具宽度)。
        同时检查该单元内网格是否已全部覆盖，及时标记完成。
        """
        gm = self.grid_map
        rx, ry = self.rx, self.ry
        if gm.grid[ry, rx] == GridMap.FREE:
            gm.set_covered(rx, ry)
        # 检查该单元是否已全部覆盖
        col_tasks = self._col_to_tasks.get(rx, [])
        for task in col_tasks:
            if task.completed:
                continue
            y_min, y_max = task.y_range()
            if not (y_min <= ry <= y_max):
                continue
            # 找到当前位置所在的单元，检查该单元是否已全部覆盖
            all_covered = True
            for gy in range(y_min, y_max + 1):
                cell = gm.grid[gy, rx]
                if cell != GridMap.OBSTACLE and cell != GridMap.COVERED:
                    all_covered = False
                    break
            if all_covered:
                task.completed = True
                self._last_completed_task_id = task.id
                self._local_path = []
                self._local_path_idx = 0
                self._update_cell_decomposition()  # 刷新邻接关系，更新单元划分
                self._global_tour = self._generate_global_tour()  # 重新生成全局巡游序列
            # 找到目标单元就退出，每个位置只能属于一个单元
            break

    def _borders_unknown(self, gx, gy):
        """
        检查该格子是否与 UNKNOWN 格子邻接，即处于已探索/未探索的边界上。
        """
        gm = self.grid_map
        for dx, dy in self._dirs_4:
            nx, ny = gx + dx, gy + dy
            if gm.is_valid(nx, ny) and gm.is_unknown(nx, ny):
                return True
        return False

    # -------------------------------------------------------------------------
    # 边界探索 (Boundary Exploration)
    # 初始绕地图最外围一周，探索边界障碍情况，为单元划分做准备
    # -------------------------------------------------------------------------




    def _do_obstacle_exploration(self, clockwise=True):
        """
        执行障碍物探索：沿障碍物边界绕行。
        停在已覆盖格子附近。

        clockwise=True: 右手法则 (顺时针, 默认)，障碍物在右侧
        clockwise=False:  左手法则 (逆时针)
        """

        gm = self.grid_map
        rx, ry = self.rx, self.ry

        # 先执行初始单元分解,（放外面了）

        # 方向定义: 0=上(0,1), 1=右(1,0), 2=下(0,-1), 3=左(-1,0)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def _left_of(d):
            return (d - 1) % 4

        def _right_of(d):
            return (d + 1) % 4

        def _back_of(d):
            return (d + 2) % 4

        # 确定初始朝向

        if clockwise:
            cur_dir = 0
        else:
            cur_dir = 2

        start_x, start_y = rx, ry
        max_steps = 5000
        t = 0
        while True:

            t += 1
            # 读左格、前格、右格
            ldx, ldy = dirs[_left_of(cur_dir)]
            left_x, left_y = rx + ldx, ry + ldy

            rdx, rdy = dirs[_right_of(cur_dir)]
            right_x, right_y = rx + rdx, ry + rdy

            fdx, fdy = dirs[cur_dir]
            front_x, front_y = rx + fdx, ry + fdy

            bdx, bdy = dirs[_back_of(cur_dir)]
            back_x, back_y = rx + bdx, ry + bdy

            left_wall = gm.is_obstacle(left_x, left_y)
            front_wall = gm.is_obstacle(front_x, front_y)
            right_wall = gm.is_obstacle(right_x, right_y)

            if clockwise:
                # 右手法则: 保持墙在右侧 → 逆时针
                if right_wall:
                    if not front_wall:
                        # 右有墙，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_wall and not left_wall:
                        # 右有墙，前有墙 → 左转
                        cur_dir = _left_of(cur_dir)
                    else:
                        # 右、前、左都有墙 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 右侧丢了墙 → 右转，前进
                    cur_dir = _right_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_obstacle(front_x, front_y):
                        rx, ry = front_x, front_y

            else:
                # 左手法则: 保持墙在左侧 → 顺时针
                if left_wall:
                    if not front_wall:
                        # 左有墙，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_wall and not right_wall:
                        # 左有墙，前有墙 → 右转
                        cur_dir = _right_of(cur_dir)
                    else:
                        # 左、前、右都有墙 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 左侧丢了墙 → 左转，前进
                    cur_dir = _left_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_obstacle(front_x, front_y):
                        rx, ry = front_x, front_y


            # 绕障模式，检查规划的格子是否已经覆盖；若已覆盖则结束。
            if self.grid_map.is_covered(rx, ry):
                break

            # 更新机器人状态
            self.rx, self.ry = rx, ry
            self.grid_map.set_covered(rx, ry)
            self.path.append((rx, ry))
            self._visited_cells.add((rx, ry))
            self._sense()

            if len(self.path) > max_steps:
                break

        # 边界探索完成，增量更新单元划分
        self._update_cell_decomposition()
        self._global_tour = self._generate_global_tour()
        self._local_path = []
        self._local_path_idx = 0
        self._first_global = True
        self._boundary_explored = True
        # from hcpp.visualization import plot_hcpp_cells
        # import os
        # os.makedirs("results/test", exist_ok=True)
        # plot_hcpp_cells(self.grid_map, self.cts, self.path, 1,
        #                 title=f"HCPP Cell Decomposition - Step {self._step_count}",
        #                 save_path=f"results/test/cells_step_{self._step_count}.png")
        # print(f"[DEBUG] Global Tour IDs: {self._global_tour}")
        return (self.rx, self.ry)


    def _do_boundary_exploration(self, clockwise=True, target=None, ob=False):
        """
        执行边界探索：沿地图边界绕行一圈。
        实时感知，无需预知全地图，障碍物自动绕行。
        绕完一圈后停在起点附近。

        clockwise=True: 左手法则 (顺时针, 默认)边界在左侧
        clockwise=False:  右手法则 (逆时针)
        """
        gm = self.grid_map
        rx, ry = self.rx, self.ry

        # 先执行初始单元分解,（放外面了）

        # 方向定义: 0=上(0,1), 1=右(1,0), 2=下(0,-1), 3=左(-1,0)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def _left_of(d):
            return (d - 1) % 4

        def _right_of(d):
            return (d + 1) % 4

        def _back_of(d):
            return (d + 2) % 4

        # 确定初始朝向
        if ob :
            if not clockwise:
                cur_dir = 0
            else:
                cur_dir = 2
        else :
            if clockwise:

                cur_dir = 0
            else:
                cur_dir = 2

        start_x, start_y = rx, ry
        max_steps = 5000
        t = 0
        while True:
            t += 1
            # 读左格、前格、右格
            ldx, ldy = dirs[_left_of(cur_dir)]
            left_x, left_y = rx + ldx, ry + ldy

            rdx, rdy = dirs[_right_of(cur_dir)]
            right_x, right_y = rx + rdx, ry + rdy

            fdx, fdy = dirs[cur_dir]
            front_x, front_y = rx + fdx, ry + fdy

            bdx, bdy = dirs[_back_of(cur_dir)]
            back_x, back_y = rx + bdx, ry + bdy

            left_wall = gm.is_obstacle(left_x, left_y)
            front_wall = gm.is_obstacle(front_x, front_y)
            right_wall = gm.is_obstacle(right_x, right_y)

            if ob:
                if not clockwise:
                    # 左手法则: 保持墙在左侧 → 顺时针
                    if left_wall:
                        if not front_wall:
                            # 左有墙，前自由 → 前进
                            rx, ry = front_x, front_y
                        elif front_wall and not right_wall:
                            # 左有墙，前有墙 → 右转
                            cur_dir = _right_of(cur_dir)
                        else:
                            # 左、前、右都有墙 → 后退并转向
                            cur_dir = _back_of(cur_dir)
                            rx, ry = back_x, back_y
                    else:
                        # 左侧丢了墙 → 左转，前进
                        cur_dir = _left_of(cur_dir)
                        fdx, fdy = dirs[cur_dir]
                        front_x, front_y = rx + fdx, ry + fdy
                        if not gm.is_obstacle(front_x, front_y):
                            rx, ry = front_x, front_y
                else:
                    # 右手法则: 保持墙在右侧 → 逆时针
                    if right_wall:
                        if not front_wall:
                            # 右有墙，前自由 → 前进
                            rx, ry = front_x, front_y
                        elif front_wall and not left_wall:
                            # 右有墙，前有墙 → 左转
                            cur_dir = _left_of(cur_dir)
                        else:
                            # 右、前、左都有墙 → 后退并转向
                            cur_dir = _back_of(cur_dir)
                            rx, ry = back_x, back_y
                    else:
                        # 右侧丢了墙 → 右转，前进
                        cur_dir = _right_of(cur_dir)
                        fdx, fdy = dirs[cur_dir]
                        front_x, front_y = rx + fdx, ry + fdy
                        if not gm.is_obstacle(front_x, front_y):
                            rx, ry = front_x, front_y
            else :

                if clockwise:
                    # 左手法则: 保持墙在左侧 → 顺时针
                    if left_wall:
                        if not front_wall:
                            # 左有墙，前自由 → 前进
                            rx, ry = front_x, front_y
                        elif front_wall and not right_wall:
                            # 左有墙，前有墙 → 右转
                            cur_dir = _right_of(cur_dir)
                        else:
                            # 左、前、右都有墙 → 后退并转向
                            cur_dir = _back_of(cur_dir)
                            rx, ry = back_x, back_y
                    else:
                        # 左侧丢了墙 → 左转，前进
                        cur_dir = _left_of(cur_dir)
                        fdx, fdy = dirs[cur_dir]
                        front_x, front_y = rx + fdx, ry + fdy
                        if not gm.is_obstacle(front_x, front_y):
                            rx, ry = front_x, front_y
                else:
                    # 右手法则: 保持墙在右侧 → 逆时针
                    if right_wall:
                        if not front_wall:
                            # 右有墙，前自由 → 前进
                            rx, ry = front_x, front_y
                        elif front_wall and not left_wall:
                            # 右有墙，前有墙 → 左转
                            cur_dir = _left_of(cur_dir)
                        else:
                            # 右、前、左都有墙 → 后退并转向
                            cur_dir = _back_of(cur_dir)
                            rx, ry = back_x, back_y
                    else:
                        # 右侧丢了墙 → 右转，前进
                        cur_dir = _right_of(cur_dir)
                        fdx, fdy = dirs[cur_dir]
                        front_x, front_y = rx + fdx, ry + fdy
                        if not gm.is_obstacle(front_x, front_y):
                            rx, ry = front_x, front_y

            # 如果是绕障模式，检查规划的格子是否已经是覆盖了，若是则结束
            if ob:
                if self.grid_map.is_covered(rx, ry):
                    break

            # 更新机器人状态
            self.rx, self.ry = rx, ry
            self.grid_map.set_covered(rx, ry)
            self.path.append((rx, ry))
            self._visited_cells.add((rx, ry))
            self._sense()

            if len(self.path) > max_steps:
                break
            if target == None:
                # 回到起点附近 → 结束
                if len(self.path) > 10:
                    if abs(rx - start_x) + abs(ry - start_y) <= 1:
                        break
            else:
                # 到达目标 → 结束
                if (rx, ry) == target:
                    break

        # 边界探索完成，增量更新单元划分
        self._update_cell_decomposition()
        self._global_tour = self._generate_global_tour()
        self._local_path = []
        self._local_path_idx = 0
        self._boundary_explored = True
        return (self.rx, self.ry)


    def _is_obstacle_in_cell(self, gx, gy):
        """检查格子是否是已知的障碍物"""
        gm = self.grid_map
        if not gm.is_valid(gx, gy):
            return False
        return gm.is_obstacle(gx, gy)


    def _init_cell_decomposition(self):
        """
        初始单元划分:
        未探索前，将地图按列划分为初始单元，每列一个。
        每列创建一个覆盖整个地图高度的单元。
        真正的障碍物分割发生在后续的 _update_cell_decomposition 中。
        """
        gm = self.grid_map
        self.cts = []
        self._next_id = 0

        # 逐列创建单元，每列覆盖整列高度
        for px in range(gm.width):
            task = CoverageTask(
                self._next_id, px, gm.height - 1, 0
            )
            self.cts.append(task)
            self._next_id += 1

        # 构建相邻关系
        self._build_adjacency()

    def _build_adjacency(self):
        """构建单元间邻接关系: 相邻列 (px 差 1) 且 y 范围重叠视为相邻"""
        # Rebuild task ID map and column-to-tasks map
        self._task_id_map = {t.id: t for t in self.cts}
        self._col_to_tasks.clear()
        for t in self.cts:
            self._col_to_tasks.setdefault(t.px, []).append(t)
        # Invalidate adjacency cache
        self._cached_adj_dict = None

        for t in self.cts:
            t.adj = []

        sorted_tasks = sorted(self.cts, key=lambda t: (t.px, t.pd))

        for i, t1 in enumerate(sorted_tasks):
            for t2 in sorted_tasks[i + 1:]:
                if t2.px - t1.px > 1:
                    break
                if t2.px - t1.px == 1:
                    y1_min, y1_max = t1.y_range()
                    y2_min, y2_max = t2.y_range()
                    if y1_min <= y2_max and y2_min <= y1_max:
                        t1.adj.append(t2.id)
                        t2.adj.append(t1.id)

    def _update_cell_decomposition(self, columns=None):
        """
        增量更新单元划分：仅处理变化列。

        Args:
            columns: 可选，指定要更新的列集合。为 None 时使用 changed_cells 中所有列。
        """
        if columns is None:
            if not self.changed_cells:
                return False
            changed_cols = set(self.changed_cells.keys())
            self.changed_cells.clear()
        else:
            changed_cols = set(columns)
            self._obstacle_changed_xs.clear()
            if not changed_cols:
                return False

        gm = self.grid_map
        has_changes = False
        col_tasks: dict[int, list] = {}
        for task in self.cts:
            if task.completed:
                continue
            if task.px in changed_cols:
                col_tasks.setdefault(task.px, []).append(task)

        tasks_to_remove = set()
        new_tasks = []

        for px in changed_cols:
            # 扫描整列，提取连续自由段 (FREE)
            free_segments = []
            seg_start = None
            for gy in range(gm.height):
                state = gm.grid[gy, px]
                if state == GridMap.FREE or state == GridMap.UNKNOWN:
                    if seg_start is None:
                        seg_start = gy
                else:
                    if seg_start is not None:
                        free_segments.append((seg_start, gy - 1))
                        seg_start = None
            if seg_start is not None:
                free_segments.append((seg_start, gm.height - 1))

            # 该列的现有未完成任务
            existing = col_tasks.get(px, [])

            if len(free_segments) == 0:
                # 整列无自由空间，所有任务标完成
                for task in existing:
                    task.completed = True
                has_changes = True
                continue

            if len(free_segments) == len(existing):
                # 数量相同：尝试收缩/更新每个任务的范围
                for task, (fs, fe) in zip(
                    sorted(existing, key=lambda t: t.pd),
                    sorted(free_segments)
                ):
                    old_min, old_max = task.y_range()
                    if fs != old_min or fe != old_max:
                        task.pd, task.pu = fs, fe
                        has_changes = True
                if has_changes:
                    self._build_adjacency()
                continue

            # 数量不同：删除旧任务，按自由段创建新任务
            for task in existing:
                tasks_to_remove.add(task.id)
            for fs, fe in free_segments:
                new_task = CoverageTask(self._next_id, px, fe, fs)
                new_tasks.append(new_task)
                self._next_id += 1
            has_changes = True

        # 应用变更
        if tasks_to_remove or new_tasks:
            self.cts = [t for t in self.cts if t.id not in tasks_to_remove]
            self.cts.extend(new_tasks)
            self._build_adjacency()

        return has_changes

    def _is_task_done(self):
        return all(t.completed for t in self.cts)

    # -------------------------------------------------------------------------
    # Algorithm 2: 全局巡游规划 (GTP)
    # -------------------------------------------------------------------------

    def _build_adjacency_graph(self):
        """构建邻接图 G=(V,E) (使用缓存)"""
        if self._cached_adj_dict is not None:
            return self._cached_adj_dict
        adj_dict = {}
        for task in self.cts:
            if not task.completed:
                adj_dict[task.id] = [n for n in task.adj
                                     if self._get_task_by_id(n)
                                     and not self._get_task_by_id(n).completed]
        self._cached_adj_dict = adj_dict
        return adj_dict

    def _classify_vertices(self, adj_dict):
        """
        顶点分类:
        - Brim vertex (边缘顶点): degree ≤ 1, 位于未覆盖区域边界
        - Internal vertex (内部顶点): degree > 1, 移除会导致图分裂
        """
        brim_set = set()
        internal_set = set()
        for tid, neighbors in adj_dict.items():
            if len(neighbors) <= 1:
                brim_set.add(tid)
            else:
                internal_set.add(tid)
        return brim_set, internal_set

    def _get_task_by_id(self, task_id):
        return self._task_id_map.get(task_id)

    def _generate_global_tour(self):
        """
        GTP 全局巡游生成 (Algorithm 2):

        按 README/论文中的思想实现边缘优先删除:
          1. 先以机器人当前位置最近的单元作为候选点。
          2. 若候选点是内部点，则在剩余图中 BFS 找最近边缘点。
          3. 删除选中的边缘点，并更新剩余图的邻接度。
          4. 优先从刚删除点的邻居继续，否则跳到最近的边缘点。
        """
        active_tasks = [t for t in self.cts if not t.completed]
        if not active_tasks:
            self._components = []
            return []

        adj_dict: dict[int, set[int]] = {}
        for task in active_tasks:
            adj_dict[task.id] = {
                n for n in task.adj
                if self._get_task_by_id(n)
                and not self._get_task_by_id(n).completed
            }

        # 记录当前 CTS 的连通分量，供 LPP 判断两单元是否连通。
        components = []
        visited_comp = set()
        for tid in adj_dict:
            if tid in visited_comp:
                continue
            comp = self._get_graph_component(tid, {
                k: list(v) for k, v in adj_dict.items()
            })
            components.append(comp)
            visited_comp |= comp
        self._components = components

        remaining = set(adj_dict)
        # 度数按邻居所在列去重: 同列多个邻居只计1度
        degree = {}
        for tid, neighbors in adj_dict.items():
            cols = set()
            for nid in neighbors:
                nt = self._get_task_by_id(nid)
                if nt is not None:
                    cols.add(nt.px)
            degree[tid] = len(cols)

        tour = []
        cnode = min(
            remaining,
            key=lambda tid: self._task_distance_from_robot(
                self._get_task_by_id(tid))
        )

        while remaining:
            if cnode not in remaining:
                cnode = self._nearest_remaining_brim(remaining, degree)

            # 内部点不直接覆盖，先在剩余图内找最近边缘点。
            if degree.get(cnode, 0) > 1:
                nnode = self._find_brim_in_remaining(
                    cnode, remaining, adj_dict, degree)
                if nnode is None:
                    nnode = self._nearest_remaining_brim(remaining, degree)
            else:
                nnode = cnode

            if nnode is None:
                break

            tour.append(nnode)
            remaining.remove(nnode)

            next_candidates = []
            for nid in list(adj_dict.get(nnode, ())):
                adj_dict[nid].discard(nnode)
                # 更新度数: 剩余的邻居按列去重计数 (同列多个邻居只计1度)
                cols = set()
                for remain_id in (adj_dict[nid] & remaining):
                    nt = self._get_task_by_id(remain_id)
                    if nt is not None:
                        cols.add(nt.px)
                degree[nid] = len(cols)
                if nid in remaining:
                    next_candidates.append(nid)
            degree.pop(nnode, None)

            if next_candidates:
                cnode = min(
                    next_candidates,
                    key=lambda tid: self._task_distance(
                        self._get_task_by_id(nnode),
                        self._get_task_by_id(tid))
                )
            else:
                cnode = self._nearest_remaining_brim(remaining, degree)

        return tour

    def _task_distance_from_robot(self, task):
        """计算机器人当前位置到单元最近端点的曼哈顿距离。"""
        if task is None:
            return math.inf
        return abs(self.rx - task.px) + min(
            abs(self.ry - task.pu), abs(self.ry - task.pd))

    def _task_distance(self, task1, task2):
        """计算两个单元最近端点之间的近似距离，用于 GTP 跳转排序。"""
        if task1 is None or task2 is None:
            return math.inf
        ys1 = (task1.pu, task1.pd)
        ys2 = (task2.pu, task2.pd)
        return abs(task1.px - task2.px) + min(
            abs(y1 - y2) for y1 in ys1 for y2 in ys2)

    def _nearest_remaining_brim(self, remaining, degree):
        """从剩余图中选离机器人最近的边缘点；没有边缘点时选最近点兜底。"""
        if not remaining:
            return None
        brim = [tid for tid in remaining if degree.get(tid, 0) <= 1]
        candidates = brim if brim else list(remaining)
        return min(candidates,
                   key=lambda tid: self._task_distance_from_robot(
                       self._get_task_by_id(tid)))

    def _find_brim_in_remaining(self, start_id, remaining, adj_dict, degree):
        """在剩余图内 BFS 查找第一个边缘点。"""
        queue = deque([start_id])
        seen = {start_id}
        while queue:
            cur = queue.popleft()
            if cur in remaining and degree.get(cur, 0) <= 1:
                return cur
            for nid in adj_dict.get(cur, ()):
                if nid in remaining and nid not in seen:
                    seen.add(nid)
                    queue.append(nid)
        return None

    def _bfs_nearest_unvisited(self, start_id, visited, adj_dict):
        """BFS 搜索最近的未访问单元"""
        queue = deque([start_id])
        bfs_visited = {start_id}
        while queue:
            cur = queue.popleft()
            for nid in adj_dict.get(cur, []):
                if nid in bfs_visited:
                    continue
                bfs_visited.add(nid)
                if nid not in visited:
                    return nid
                queue.append(nid)
        return None

    def _bfs_find_brim(self, start_id, adj_dict, brim_set, visited):
        """BFS 搜索第一个边缘顶点"""
        queue = deque([start_id])
        bfs_visited = {start_id}
        while queue:
            cur = queue.popleft()
            for nid in adj_dict.get(cur, []):
                if nid in bfs_visited:
                    continue
                bfs_visited.add(nid)
                if nid in visited:
                    continue
                if nid in brim_set:
                    return nid
                queue.append(nid)
        return None

    def _get_graph_component(self, start_id, adj_dict):
        """BFS 获取从 start_id 出发在邻接图中的连通分量（所有可达节点ID集合）"""
        component = {start_id}
        queue = deque([start_id]) # 双端队列
        while queue:
            cur = queue.popleft()
            for nid in adj_dict.get(cur, []):
                if nid not in component:
                    component.add(nid)
                    queue.append(nid)
        return component

    # -------------------------------------------------------------------------
    # Algorithm 3: 局部路径规划 (LPP) — 两场景分发
    # -------------------------------------------------------------------------

    def _cells_connected(self, task1, task2):
        """检查两个单元是否在同一个连通分量中"""
        if not hasattr(self, '_components'):
            return False
        for comp in self._components:
            if task1.id in comp and task2.id in comp:
                return True
        return False

    # # ---- 场景1: 相邻直达 ----

    # def _plan_direct_path(self, current, next_task):
    #     """
    #     场景1: 相邻单元格直达

    #     执行逻辑:
    #     1. 找到目标单元格两个端点中距离机器人更近的端点 p_u
    #     2. 直接移动到 p_u (相邻单元格，直线不破坏连通性)
    #     3. 从 p_u 到远端端点 p_v 沿中线覆盖 (一条直线)
    #     如遇障碍物则用 A* 绕行后继续
    #     """
    #     gm = self.grid_map
    #     px = next_task.px

    #     # 选择较近端点
    #     dist_upper = abs(self.rx - px) + abs(self.ry - next_task.pu)
    #     dist_lower = abs(self.rx - px) + abs(self.ry - next_task.pd)

    #     if dist_upper <= dist_lower:
    #         target_endpoint = (px, next_task.pu)
    #         start_y, end_y = next_task.pu, next_task.pd
    #     else:
    #         target_endpoint = (px, next_task.pd)
    #         start_y, end_y = next_task.pd, next_task.pu

    #     path = []

    #     # 1. 移动到目标端点 (相邻直达，有障碍时 A* 绕行)
    #     if (gm.is_valid(px, start_y) and not gm.is_obstacle(px, start_y)
    #             and abs(self.rx - px) + abs(self.ry - start_y) <= 2):
    #         path.append((px, start_y))
    #     else:
    #         move_path = self._astar(self.rx, self.ry,
    #                                 target_endpoint[0], target_endpoint[1],
    #                                 allow_unknown=True)
    #         if move_path:
    #             path.extend(move_path[1:])  # skip current pos
    #         else:
    #             return self._fallback_frontier_path()

    #     # 2. 从中线一端到另一端覆盖 (直线)
    #     step_y = -1 if start_y > end_y else 1
    #     cur_y = start_y
    #     while cur_y != end_y:
    #         next_y = cur_y + step_y
    #         path.append((px, next_y))
    #         cur_y = next_y

    #     if not path:
    #         return self._fallback_frontier_path()
    #     return path

    # ----  未覆盖区域边界绕行辅助函数，初步判断走顺时针还是逆时针 ----

    def _plan_CWorCCW(self, target, clockwise=True):
        """
        判断沿未覆盖区边界顺时针(CW)还是逆时针(CCW)更短。
        """
        gm = self.grid_map
        rx, ry = self.rx, self.ry

        pathCW = []
        pathCCW = []

        # 方向定义: 0=上(0,1), 1=右(1,0), 2=下(0,-1), 3=左(-1,0)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def _left_of(d):
            return (d - 1) % 4

        def _right_of(d):
            return (d + 1) % 4

        def _back_of(d):
            return (d + 2) % 4

        # 确定初始朝向，在边缘，确定上下即可，后面自动纠偏
        if clockwise:
            cur_dir = 0
        else:
            cur_dir = 2  

        start_x, start_y = rx, ry
        max_steps = 5000

        while True:
            # 读左格、前格、右格
            ldx, ldy = dirs[_left_of(cur_dir)]
            left_x, left_y = rx + ldx, ry + ldy

            rdx, rdy = dirs[_right_of(cur_dir)]
            right_x, right_y = rx + rdx, ry + rdy

            fdx, fdy = dirs[cur_dir]
            front_x, front_y = rx + fdx, ry + fdy

            bdx, bdy = dirs[_back_of(cur_dir)]
            back_x, back_y = rx + bdx, ry + bdy

            left_frontier = gm.is_uncovered_frontier(left_x, left_y)
            front_frontier = gm.is_uncovered_frontier(front_x, front_y)
            right_frontier = gm.is_uncovered_frontier(right_x, right_y)


            if clockwise:
                # 左手法则: 保持覆盖区在左侧 → 顺时针
                if left_frontier:
                    if not front_frontier:
                        # 左有覆盖区，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_frontier and not right_frontier:
                        # 左有覆盖区，前有覆盖区 → 右转
                        cur_dir = _right_of(cur_dir)
                    else:
                        # 左、前、右都有覆盖区 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 左侧丢了墙 → 左转，前进
                    cur_dir = _left_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_uncovered_frontier(front_x, front_y):
                        rx, ry = front_x, front_y

                pathCW.append((rx, ry))
            else:
                # 右手法则: 保持墙在右侧 → 逆时针
                if right_frontier:
                    if not front_frontier:
                        # 右有墙，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_frontier and not left_frontier:
                        # 右有墙，前有墙 → 左转
                        cur_dir = _left_of(cur_dir)
                    else:
                        # 右、前、左都有墙 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 右侧丢了墙 → 右转，前进
                    cur_dir = _right_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_uncovered_frontier(front_x, front_y):
                        rx, ry = front_x, front_y

                pathCCW.append((rx, ry))



            if len(pathCW) > max_steps or len(pathCCW) > max_steps:
                break
            # 到达目标 → 结束
            if (rx, ry) == target:
                break
        if clockwise:
            return pathCW
        else:
            return pathCCW


    def _plan_boundary_path(self, current, next_task):
        """
        场景1: 两单元格不相邻但同属一个连通未覆盖区

        执行逻辑:
        1. 收集连通未覆盖区域的所有 FREE 格子
        2. 沿外围边界分别生成 CW / CCW 两条候选路径
        3. 选较短路径
        4. 走过的格子标记 COVERED 并收缩单元
        5. 到达目标后追加中线覆盖
        """
        gm = self.grid_map
        px = next_task.px

        pu = (px, next_task.pu)
        pd = (px, next_task.pd)

        # 生成顺逆时针路径
        cw_path = self._plan_CWorCCW(pu, clockwise=True)
        ccw_path = self._plan_CWorCCW(pd, clockwise=False)
        

        # 选择较短者
        # 优先选择顺时针路径
        if len(cw_path) <= len(ccw_path):
            gm = self.grid_map
            rx, ry = self.rx, self.ry

            # 方向定义: 0=上(0,1), 1=右(1,0), 2=下(0,-1), 3=左(-1,0)
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            def _left_of(d):
                return (d - 1) % 4

            def _right_of(d):
                return (d + 1) % 4

            def _back_of(d):
                return (d + 2) % 4

            cur_dir = 0 

            start_x, start_y = rx, ry
            max_steps = 5000

            while True:
                # 读左格、前格、右格
                ldx, ldy = dirs[_left_of(cur_dir)]
                left_x, left_y = rx + ldx, ry + ldy

                rdx, rdy = dirs[_right_of(cur_dir)]
                right_x, right_y = rx + rdx, ry + rdy

                fdx, fdy = dirs[cur_dir]
                front_x, front_y = rx + fdx, ry + fdy

                bdx, bdy = dirs[_back_of(cur_dir)]
                back_x, back_y = rx + bdx, ry + bdy

                left_frontier = gm.is_uncovered_frontier(left_x, left_y)
                front_frontier = gm.is_uncovered_frontier(front_x, front_y)
                right_frontier = gm.is_uncovered_frontier(right_x, right_y)

                # 左手法则: 保持覆盖区在左侧 → 顺时针
                if left_frontier:
                    if not front_frontier:
                        # 左有覆盖区，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_frontier and not right_frontier:
                        # 左有覆盖区，前有覆盖区 → 右转
                        cur_dir = _right_of(cur_dir)
                    else:
                        # 左、前、右都有覆盖区 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 左侧丢了墙 → 左转，前进
                    cur_dir = _left_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_uncovered_frontier(front_x, front_y):
                        rx, ry = front_x, front_y

                self.rx, self.ry = rx, ry
                self.grid_map.set_covered(rx, ry)
                self.path.append((rx, ry))
                self._visited_cells.add((rx, ry))
                self._sense()

                if len(self.path) > max_steps:
                    break

                # 到达目标 → 结束
                if (rx, ry) == pu:
                    break


        else:
            gm = self.grid_map
            rx, ry = self.rx, self.ry

            # 方向定义: 0=上(0,1), 1=右(1,0), 2=下(0,-1), 3=左(-1,0)
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            def _left_of(d):
                return (d - 1) % 4

            def _right_of(d):
                return (d + 1) % 4

            def _back_of(d):
                return (d + 2) % 4

            # 确定初始朝向，在边缘，确定上下即可，后面自动纠偏
            cur_dir = 2  

            start_x, start_y = rx, ry
            max_steps = 5000

            while True:
                # 读左格、前格、右格
                ldx, ldy = dirs[_left_of(cur_dir)]
                left_x, left_y = rx + ldx, ry + ldy

                rdx, rdy = dirs[_right_of(cur_dir)]
                right_x, right_y = rx + rdx, ry + rdy

                fdx, fdy = dirs[cur_dir]
                front_x, front_y = rx + fdx, ry + fdy

                bdx, bdy = dirs[_back_of(cur_dir)]
                back_x, back_y = rx + bdx, ry + bdy

                left_frontier = gm.is_uncovered_frontier(left_x, left_y)
                front_frontier = gm.is_uncovered_frontier(front_x, front_y)
                right_frontier = gm.is_uncovered_frontier(right_x, right_y)

                # 右手法则: 保持墙在右侧 → 逆时针
                if right_frontier:
                    if not front_frontier:
                        # 右有墙，前自由 → 前进
                        rx, ry = front_x, front_y
                    elif front_frontier and not left_frontier:
                        # 右有墙，前有墙 → 左转
                        cur_dir = _left_of(cur_dir)
                    else:
                        # 右、前、左都有墙 → 后退并转向
                        cur_dir = _back_of(cur_dir)
                        rx, ry = back_x, back_y
                else:
                    # 右侧丢了墙 → 右转，前进
                    cur_dir = _right_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    front_x, front_y = rx + fdx, ry + fdy
                    if not gm.is_uncovered_frontier(front_x, front_y):
                        rx, ry = front_x, front_y


                self.rx, self.ry = rx, ry
                self.grid_map.set_covered(rx, ry)
                self.path.append((rx, ry))
                self._visited_cells.add((rx, ry))
                self._sense()

                if len(self.path) > max_steps:
                    break

                # 到达目标 → 结束
                if (rx, ry) == pd:
                    break

        self._update_cell_decomposition()


    # ---- 场景2: A* + 直线覆盖 (相邻 / 不连通) ----

    def _plan_disconnected_path(self, current, next_task):
        """
        场景2: 相邻或不在同一连通区域 — A* + 直线覆盖

        执行逻辑:
        1. 选择目标单元较近的端点作为目标
        2. A* 导航到该端点 (优先走 COVERED/FREE, 不行则允许 UNKNOWN)
        3. 从该端点到远端端点沿中线逐格覆盖 (直线)
        """
        gm = self.grid_map
        px = next_task.px

        # 选择较近端点
        dist_upper = abs(self.rx - px) + abs(self.ry - next_task.pu)
        dist_lower = abs(self.rx - px) + abs(self.ry - next_task.pd)

        if dist_upper <= dist_lower:
            target = (px, next_task.pu)
            start_y, end_y = next_task.pu, next_task.pd
        else:
            target = (px, next_task.pd)
            start_y, end_y = next_task.pd, next_task.pu

        path = []

        # A* 穿过已覆盖区域 (allow_unknown=False)
        move_path = self._astar(self.rx, self.ry,
                                target[0], target[1],
                                allow_unknown=False)
        if move_path:
            path.extend(move_path[1:])  # skip current pos
        else:
            # A* 失败: 尝试 allow_unknown
            move_path = self._astar(self.rx, self.ry,
                                    target[0], target[1],
                                    allow_unknown=True)
            if move_path:
                path.extend(move_path[1:])
            else:
                return self._fallback_frontier_path()

        # 追加中线覆盖
        step_y = -1 if start_y > end_y else 1
        y = start_y + step_y
        while y != end_y + step_y:
            if gm.is_valid(px, y) and not gm.is_obstacle(px, y):
                path.append((px, y))
            y += step_y

        if not path:
            return self._fallback_frontier_path()
        return path

    # ---- 回退策略 ----

    def _fallback_frontier_path(self):
        """回退策略: 寻找最近前沿，A* 规划到那里 (允许穿越UNKNOWN)"""
        frontier = self._find_frontier()
        if frontier:
            path = self._astar(self.rx, self.ry, frontier[0], frontier[1],
                               allow_unknown=True)
            if path:
                return path
        # 没有前沿 → 找最近的可达 FREE 格子
        nearest_free = self._find_nearest_free()
        if nearest_free:
            path = self._astar(self.rx, self.ry,
                               nearest_free[0], nearest_free[1],
                               allow_unknown=True)
            if path:
                return path
        return []

    def _move_and_sense(self, gx, gy, update_task=True):
        """探索子过程中的原子移动：更新位姿、覆盖、保存路径并立即感知。"""
        if not self.grid_map.is_valid(gx, gy):
            return False
        if self.grid_map.is_obstacle(gx, gy):
            return False

        self.rx, self.ry = gx, gy
        if not self.path or self.path[-1] != (gx, gy):
            self.path.append((gx, gy))
        self._visited_cells.add((gx, gy))
        if update_task:
            self._cover_nearby()
        else:
            if self.grid_map.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                self.grid_map.set_covered(gx, gy)
        self._sense()
        return True

    def _run_wall_follow(self, clockwise=True, keep_wall="left",
                         stop_at_start=False, stop_on_covered=False,
                         max_steps=None, initial_dir=None):
        """
        连续式边界跟随。

        keep_wall:
          - "left": 让边界/障碍保持在机器人左侧，适合顺时针走外边界；
          - "right": 让边界/障碍保持在机器人右侧，适合绕内部障碍。
        """
        gm = self.grid_map
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def left_of(d):
            return (d - 1) % 4

        def right_of(d):
            return (d + 1) % 4

        def back_of(d):
            return (d + 2) % 4

        def blocked(x, y):
            return (not gm.is_valid(x, y)) or gm.is_obstacle(x, y)

        cur_dir = initial_dir if initial_dir is not None else (0 if clockwise else 2)
        start = (self.rx, self.ry)
        visited_states = set()
        if max_steps is None:
            max_steps = max(8, gm.width * gm.height * 2)

        for step_no in range(max_steps):
            state = (self.rx, self.ry, cur_dir)
            if step_no > 8 and state in visited_states:
                break
            visited_states.add(state)

            rx, ry = self.rx, self.ry
            fdx, fdy = dirs[cur_dir]
            ldx, ldy = dirs[left_of(cur_dir)]
            rdx, rdy = dirs[right_of(cur_dir)]
            bdx, bdy = dirs[back_of(cur_dir)]

            front = (rx + fdx, ry + fdy)
            left = (rx + ldx, ry + ldy)
            right = (rx + rdx, ry + rdy)
            back = (rx + bdx, ry + bdy)

            if keep_wall == "left":
                # 左侧有墙且前方可走则前进；前方堵住则右转；左侧空了则左转贴回去。
                if blocked(*left):
                    if not blocked(*front):
                        nxt = front
                    elif not blocked(*right):
                        cur_dir = right_of(cur_dir)
                        continue
                    else:
                        cur_dir = back_of(cur_dir)
                        nxt = back
                else:
                    cur_dir = left_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    nxt = (rx + fdx, ry + fdy)
            else:
                # 右手法则，内部障碍绕行时让障碍保持在右侧。
                if blocked(*right):
                    if not blocked(*front):
                        nxt = front
                    elif not blocked(*left):
                        cur_dir = left_of(cur_dir)
                        continue
                    else:
                        cur_dir = back_of(cur_dir)
                        nxt = back
                else:
                    cur_dir = right_of(cur_dir)
                    fdx, fdy = dirs[cur_dir]
                    nxt = (rx + fdx, ry + fdy)

            if blocked(*nxt):
                continue
            if stop_on_covered and step_no > 0 and gm.is_covered(*nxt):
                break
            if not self._move_and_sense(nxt[0], nxt[1], update_task=False):
                break
            if stop_at_start and step_no > 8:
                if abs(self.rx - start[0]) + abs(self.ry - start[1]) <= 1:
                    break

        self._local_path = []
        self._local_path_idx = 0
        self._update_tasks_after_sensing()

    def _run_initial_boundary_exploration(self):
        """首次先沿环境外围边界探索一圈，再生成 CTS 和 GTP。"""
        if not self.cts:
            self._init_cell_decomposition()
        self._run_wall_follow(clockwise=True, keep_wall="left",
                              stop_at_start=True,
                              max_steps=self.grid_map.width *
                              self.grid_map.height * 2)
        self._boundary_explored = True
        self._global_tour = self._generate_global_tour()

    def _obstacle_component_key(self, gx, gy):
        """返回当前已知的障碍连通块格子集合，用于避免重复绕同一障碍。"""
        gm = self.grid_map
        if not gm.is_valid(gx, gy) or not gm.is_obstacle(gx, gy):
            return None
        queue = deque([(gx, gy)])
        visited = {(gx, gy)}
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in gm.get_neighbors_4(cx, cy):
                if (nx, ny) not in visited and gm.is_obstacle(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def _current_vertical_direction(self):
        """根据当前局部路径判断机器人正在向上还是向下扫描。"""
        if self._local_path and self._local_path_idx < len(self._local_path):
            nx, ny = self._local_path[self._local_path_idx]
            dy = ny - self.ry
            if dy > 0:
                return 1
            if dy < 0:
                return -1
        if len(self.path) >= 2:
            dy = self.path[-1][1] - self.path[-2][1]
            if dy > 0:
                return 1
            if dy < 0:
                return -1
        return 0

    def _side_unknown_obstacle(self):
        """
        检查当前扫描列左右两侧是否有需要探索物理边界的障碍。

        当前 HCPP/GTP 可能让机器人从左侧推进，也可能绕到障碍右侧再回来，
        所以未知边界障碍既可能出现在 (rx + 1, ry)，也可能出现在
        (rx - 1, ry)。

        手法则按前进方向决定:
        - 障碍在 x+1:
          往上时障碍在前进方向右侧 -> 右手法则；
          往下时障碍在前进方向左侧 -> 左手法则。
        - 障碍在 x-1:
          往上时障碍在前进方向左侧 -> 左手法则；
          往下时障碍在前进方向右侧 -> 右手法则。
        """
        gm = self.grid_map
        vertical_dir = self._current_vertical_direction()
        moving_up = vertical_dir >= 0

        # 优先检查当前扫描前进侧。方向不明确时先看右侧，保持旧行为。
        side_order = (1, -1) if moving_up else (-1, 1)
        for side_dx in side_order:
            ox, oy = self.rx + side_dx, self.ry
            if not gm.is_valid(ox, oy) or not gm.is_obstacle(ox, oy):
                continue

            key = self._obstacle_component_key(ox, oy)
            if not key or key & self._explored_obstacle_keys:
                continue

            has_unknown = False
            for cx, cy in key:
                for nx, ny in gm.get_neighbors_4(cx, cy):
                    if gm.is_unknown(nx, ny):
                        has_unknown = True
                        break
                if has_unknown:
                    break
            if has_unknown:
                if side_dx > 0:
                    keep_wall = "right" if moving_up else "left"
                else:
                    keep_wall = "left" if moving_up else "right"
                initial_dir = 0 if moving_up else 2
                return ox, oy, key, keep_wall, initial_dir
        return None

    def _run_obstacle_boundary_exploration(self, obstacle_cells,
                                           keep_wall="right",
                                           initial_dir=0):
        """发现内部障碍后，先连续绕障探索其物理边界，再更新 CTS/GTP。"""
        self._run_wall_follow(clockwise=True, keep_wall=keep_wall,
                              stop_on_covered=True,
                              max_steps=self.grid_map.width *
                              self.grid_map.height,
                              initial_dir=initial_dir)
        if obstacle_cells:
            # 已绕过的障碍格保存为集合；后续同一连通块继续被感知扩大时可直接跳过。
            self._explored_obstacle_keys.update(obstacle_cells)
        self._update_tasks_after_sensing()
        self._global_tour = self._generate_global_tour()

    def _is_coverable_known(self, gx, gy):
        """已知且可覆盖的格子：FREE 或 COVERED。"""
        if not self.grid_map.is_valid(gx, gy):
            return False
        return self.grid_map.grid[gy, gx] in (GridMap.FREE, GridMap.COVERED)

    def _sync_completed_tasks(self):
        """根据当前认知地图同步 CTS 中各单元的完成状态。"""
        gm = self.grid_map
        for task in self.cts:
            if task.completed:
                continue
            y_min, y_max = task.y_range()
            done = True
            for gy in range(y_min, y_max + 1):
                state = gm.grid[gy, task.px]
                if state in (GridMap.FREE, GridMap.UNKNOWN):
                    done = False
                    break
            task.completed = done

    def _update_tasks_after_sensing(self):
        """
        感知后增量更新 CTS。

        返回 True 只表示“新障碍造成的结构变化”，用于打断当前局部路径并重建
        GTP；单纯 COVERED 引起的端点收缩不能打断 LPP，否则连通但不相邻的
        边界绕行会被拆成每步重规划。
        """
        structural_changed = False
        if self._obstacle_changed_xs:
            obstacle_cols = set(self._obstacle_changed_xs)
            structural_changed = self._update_cell_decomposition(
                columns=obstacle_cols)
            self._obstacle_changed_xs.clear()
            self._new_obstacle_flag = False
            for px in obstacle_cols:
                self.changed_cells.pop(px, None)
        elif self.changed_cells:
            # 覆盖造成的单元收缩仍更新 CTS，但不强制重规划。
            self._update_cell_decomposition()

        self._sync_completed_tasks()
        if structural_changed or not self._global_tour:
            self._global_tour = self._generate_global_tour()
        return structural_changed

    def _pop_next_task(self):
        """从全局巡游序列中取下一个未完成任务。"""
        while self._global_tour:
            tid = self._global_tour.pop(0)
            task = self._get_task_by_id(tid)
            if task is not None and not task.completed:
                return task
        self._global_tour = self._generate_global_tour()
        while self._global_tour:
            tid = self._global_tour.pop(0)
            task = self._get_task_by_id(tid)
            if task is not None and not task.completed:
                return task
        return None

    def _nearest_cell_in_task(self, task, prefer_uncovered=True):
        """返回单元内距离机器人最近的可用格子。"""
        gm = self.grid_map
        y_min, y_max = task.y_range()
        candidates = []
        for gy in range(y_min, y_max + 1):
            state = gm.grid[gy, task.px]
            if state == GridMap.OBSTACLE:
                continue
            if prefer_uncovered and state != GridMap.FREE:
                continue
            if state in (GridMap.FREE, GridMap.COVERED):
                dist = abs(task.px - self.rx) + abs(gy - self.ry)
                candidates.append((dist, gy))
        if not candidates and prefer_uncovered:
            return self._nearest_cell_in_task(task, prefer_uncovered=False)
        if not candidates:
            return None
        _, gy = min(candidates)
        return (task.px, gy)

    def _task_sweep_from(self, task, start_cell):
        """从进入点开始补齐该单元未覆盖格子，优先沿列方向扫描。"""
        gm = self.grid_map
        px, start_y = start_cell
        y_min, y_max = task.y_range()

        # 先覆盖从当前位置到较近端点的一侧，再折返覆盖另一侧。
        if abs(start_y - y_min) <= abs(start_y - y_max):
            order = list(range(start_y, y_min - 1, -1))
            order.extend(range(start_y + 1, y_max + 1))
        else:
            order = list(range(start_y, y_max + 1))
            order.extend(range(start_y - 1, y_min - 1, -1))

        path = []
        cur = (px, start_y)
        for gy in order:
            if not gm.is_valid(px, gy) or gm.is_obstacle(px, gy):
                continue
            if gm.grid[gy, px] != GridMap.FREE and (px, gy) != start_cell:
                continue
            target = (px, gy)
            if target == cur:
                if not path or path[-1] != target:
                    path.append(target)
                continue
            segment = self._astar_4(cur[0], cur[1], target[0], target[1],
                                    allow_unknown=False)
            if segment:
                path.extend(segment[1:])
                cur = target
        return path

    def _boundary_candidate_path(self, start, goal, clockwise=True):
        """
        在已知未覆盖区域边界上生成候选路径。
        这里用“贴着 COVERED/OBSTACLE 的 FREE 格子”近似 README 中的边界绕行。
        """
        gm = self.grid_map
        queue = deque([start])
        parent = {start: None}
        dirs = self._dirs_4 if clockwise else tuple(reversed(self._dirs_4))

        while queue:
            cur = queue.popleft()
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            ordered = []
            for dx, dy in dirs:
                nx, ny = cur[0] + dx, cur[1] + dy
                if (nx, ny) in parent:
                    continue
                if not gm.is_valid(nx, ny):
                    continue
                if gm.grid[ny, nx] not in (GridMap.FREE, GridMap.COVERED):
                    continue
                boundary_score = 0
                for bx, by in self._dirs_4:
                    ax, ay = nx + bx, ny + by
                    if not gm.is_valid(ax, ay) or gm.grid[ay, ax] in (
                            GridMap.COVERED, GridMap.OBSTACLE):
                        boundary_score += 1
                ordered.append((-boundary_score,
                                abs(nx - goal[0]) + abs(ny - goal[1]),
                                (nx, ny)))

            for _, _, nxt in sorted(ordered):
                parent[nxt] = cur
                queue.append(nxt)
        return None

    def _is_uncovered_boundary_cell(self, gx, gy):
        """判断格子是否是未覆盖区域边界格。"""
        gm = self.grid_map
        if not gm.is_valid(gx, gy) or gm.grid[gy, gx] != GridMap.FREE:
            return False
        for nx, ny in gm.get_neighbors_4(gx, gy):
            if gm.grid[ny, nx] in (
                    GridMap.COVERED, GridMap.OBSTACLE, GridMap.UNKNOWN):
                return True
        return gx in (0, gm.width - 1) or gy in (0, gm.height - 1)

    def _nearest_neighbor_boundary_cell(self):
        """从当前位置四邻域中选择最近的未覆盖边界格。"""
        candidates = []
        for nx, ny in self.grid_map.get_neighbors_4(self.rx, self.ry):
            if self._is_uncovered_boundary_cell(nx, ny):
                candidates.append((nx, ny))
        if not candidates:
            return None
        return min(candidates, key=lambda p: (abs(p[0] - self.rx) +
                                             abs(p[1] - self.ry), p[0], p[1]))

    def _boundary_path_between(self, start, goal):
        """
        沿未覆盖边界从 start 绕行到 goal。
        只允许经过 FREE 的未覆盖边界格，避免局部路径穿过已覆盖区域。
        """
        gm = self.grid_map
        queue = deque([start])
        parent = {start: None}

        def passable(x, y):
            if not gm.is_valid(x, y) or gm.is_obstacle(x, y):
                return False
            if (x, y) == goal:
                return gm.grid[y, x] == GridMap.FREE
            return self._is_uncovered_boundary_cell(x, y)

        while queue:
            cur = queue.popleft()
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            neighbors = []
            for nx, ny in gm.get_neighbors_4(cur[0], cur[1]):
                if (nx, ny) in parent or not passable(nx, ny):
                    continue
                dist = abs(nx - goal[0]) + abs(ny - goal[1])
                neighbors.append((dist, nx, ny))

            for _, nx, ny in sorted(neighbors):
                parent[(nx, ny)] = cur
                queue.append((nx, ny))
        return None

    def _follow_uncovered_boundary_path(self, start, goal, keep_wall="left",
                                        initial_dir=None):
        """
        方向性未覆盖边界跟随。

        keep_wall="left" 约等于顺时针候选，keep_wall="right" 约等于逆时针候选。
        只在 FREE 的未覆盖边界格之间移动，生成一条候选边界路径。
        """
        gm = self.grid_map
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def left_of(d):
            return (d - 1) % 4

        def right_of(d):
            return (d + 1) % 4

        def back_of(d):
            return (d + 2) % 4

        def passable(x, y):
            return ((x, y) == goal or
                    self._is_uncovered_boundary_cell(x, y))

        if not passable(*start) or not passable(*goal):
            return None

        if initial_dir is None:
            # 用目标方向给一个初始朝向；后续由左右手法则纠偏。
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            if abs(dx) > abs(dy):
                cur_dir = 1 if dx > 0 else 3
            else:
                cur_dir = 0 if dy >= 0 else 2
        else:
            cur_dir = initial_dir

        cur = start
        path = [start]
        seen = {(cur[0], cur[1], cur_dir)}
        max_steps = max(8, gm.width * gm.height * 2)

        for _ in range(max_steps):
            if cur == goal:
                return path

            rx, ry = cur
            def outside(x, y):
                if not gm.is_valid(x, y):
                    return True
                return gm.grid[y, x] in (
                    GridMap.COVERED, GridMap.OBSTACLE, GridMap.UNKNOWN)

            if keep_wall == "left":
                side_dir = left_of(cur_dir)
                away_dir = right_of(cur_dir)
            else:
                side_dir = right_of(cur_dir)
                away_dir = left_of(cur_dir)

            sx, sy = rx + dirs[side_dir][0], ry + dirs[side_dir][1]
            if outside(sx, sy):
                # Keep following the current boundary while the wall is still
                # on the requested side. Turn away only when the front is shut.
                candidate_dirs = [cur_dir, away_dir, back_of(cur_dir),
                                  side_dir]
            else:
                # The side wall disappeared; turn toward it to reattach.
                candidate_dirs = [side_dir, cur_dir, away_dir,
                                  back_of(cur_dir)]

            moved = False
            for ndir in candidate_dirs:
                nx, ny = rx + dirs[ndir][0], ry + dirs[ndir][1]
                can_probe_corner = (
                    ndir == cur_dir and outside(sx, sy) and
                    gm.is_valid(nx, ny) and gm.grid[ny, nx] == GridMap.FREE)
                if not passable(nx, ny) and not can_probe_corner:
                    continue
                # 指定绕向体现在候选方向顺序中；只要下一格仍是未覆盖边界格，
                # 拐角处允许短暂丢失侧墙，再由左右手规则重新贴回边界。
                state = (nx, ny, ndir)
                if state in seen:
                    continue
                seen.add(state)
                cur = (nx, ny)
                cur_dir = ndir
                path.append(cur)
                moved = True
                break

            if not moved:
                return None

        return None

    def _boundary_direction_paths(self, start, goal):
        """
        枚举从 start 出发的边界方向候选。

        未覆盖边界格通常形成一条或多条链/环。相比普通 BFS 直接从 start
        搜索，这里先固定 start 的第一个边界邻居，相当于分别沿顺/逆两个
        方向绕行，再比较哪条更短。
        """
        gm = self.grid_map
        if not self._is_uncovered_boundary_cell(*start):
            return []
        if not self._is_uncovered_boundary_cell(*goal):
            return []

        first_steps = []
        for nx, ny in gm.get_neighbors_4(start[0], start[1]):
            if self._is_uncovered_boundary_cell(nx, ny):
                first_steps.append((nx, ny))

        paths = []
        for first in first_steps:
            queue = deque([first])
            parent = {first: start}
            blocked_start = start

            while queue:
                cur = queue.popleft()
                if cur == goal:
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    path.reverse()
                    # 用“第一步在起点周围的左右侧关系”近似标注顺/逆绕向，
                    # 后续只把较短绕向传给连续贴边算法。
                    keep_wall = self._boundary_keep_wall_for_first_step(
                        start, first)
                    paths.append((keep_wall, path))
                    break

                for nx, ny in gm.get_neighbors_4(cur[0], cur[1]):
                    nxt = (nx, ny)
                    if nxt == blocked_start or nxt in parent:
                        continue
                    if not self._is_uncovered_boundary_cell(nx, ny):
                        continue
                    parent[nxt] = cur
                    queue.append(nxt)

        return paths

    def _boundary_keep_wall_for_first_step(self, start, first_step):
        """根据边界起点的第一步判断该候选应使用左手还是右手贴边。"""
        gm = self.grid_map
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        dx = first_step[0] - start[0]
        dy = first_step[1] - start[1]
        move_dir = dir_map.get((dx, dy))
        if move_dir is None:
            return "left"

        left_dir = (move_dir - 1) % 4
        right_dir = (move_dir + 1) % 4
        lx, ly = start[0] + dirs[left_dir][0], start[1] + dirs[left_dir][1]
        rx, ry = start[0] + dirs[right_dir][0], start[1] + dirs[right_dir][1]

        def outside(x, y):
            if not gm.is_valid(x, y):
                return True
            return gm.grid[y, x] in (
                GridMap.COVERED, GridMap.OBSTACLE, GridMap.UNKNOWN)

        left_outside = outside(lx, ly)
        right_outside = outside(rx, ry)
        if left_outside and not right_outside:
            return "left"
        if right_outside and not left_outside:
            return "right"
        return "left"

    def _plan_connected_nonadjacent_path(self, entry):
        """
        LPP 场景2：目标单元与当前已覆盖单元连通但不相邻。
        先进入当前位置四邻域中的未覆盖边界格，再沿未覆盖边界绕到目标入口。
        """
        boundary_start = self._nearest_neighbor_boundary_cell()
        if boundary_start is None:
            return None

        to_boundary = self._astar_4(self.rx, self.ry,
                                    boundary_start[0], boundary_start[1],
                                    allow_unknown=False)
        if not to_boundary:
            return None

        # 先快速用边界图估计顺/逆哪个方向更短，再用连续贴边法生成路径。
        estimate_paths = self._boundary_direction_paths(boundary_start, entry)
        boundary_path = None
        if estimate_paths:
            best_keep_wall, best_estimate = min(
                estimate_paths, key=lambda item: len(item[1]))
            first_step = best_estimate[1] if len(best_estimate) > 1 else None
            if first_step is not None:
                dx = first_step[0] - boundary_start[0]
                dy = first_step[1] - boundary_start[1]
                dir_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
                initial_dir = dir_map.get((dx, dy))
                follow_candidates = []
                for keep_wall in (best_keep_wall,
                                  "right" if best_keep_wall == "left"
                                  else "left"):
                    candidate = self._follow_uncovered_boundary_path(
                        boundary_start, entry, keep_wall=keep_wall,
                        initial_dir=initial_dir)
                    if candidate:
                        follow_candidates.append(candidate)
                if follow_candidates:
                    boundary_path = min(follow_candidates, key=len)
            if not boundary_path:
                boundary_path = best_estimate
        else:
            boundary_path = self._boundary_path_between(boundary_start, entry)
        if not boundary_path:
            return None

        path = to_boundary[1:]
        if path and path[-1] == boundary_path[0]:
            path.extend(boundary_path[1:])
        else:
            path.extend(boundary_path)
        return path

    def _plan_local_path_to_task(self, task):
        """
        LPP 局部路径规划:
        - 相邻/不连通场景用 A* 到达目标单元入口；
        - 连通但不相邻时尝试顺/逆两条边界候选，取较短者；
        - 到达后沿列中线覆盖目标单元剩余 FREE 格子。
        """
        if task is None or task.completed:
            return []

        entry = self._nearest_cell_in_task(task)
        if entry is None:
            task.completed = True
            return []

        direct = self._astar_4(self.rx, self.ry, entry[0], entry[1],
                               allow_unknown=False)
        if direct is None:
            direct = self._astar(self.rx, self.ry, entry[0], entry[1],
                                 allow_unknown=False)

        # 若 A* 不可达，说明已知区域断开；回退到最近前沿继续探索。
        if direct is None:
            return self._fallback_frontier_path()

        use_path = direct
        entry_is_adjacent = abs(self.rx - entry[0]) + abs(self.ry - entry[1]) <= 1
        same_column_continuation = self.rx == task.px
        if (not entry_is_adjacent and not same_column_continuation
                and len(direct) > 2):
            boundary_path = self._plan_connected_nonadjacent_path(entry)
            if boundary_path:
                use_path = boundary_path
        elif len(direct) > 2:
            cw = self._boundary_candidate_path((self.rx, self.ry), entry,
                                               clockwise=True)
            ccw = self._boundary_candidate_path((self.rx, self.ry), entry,
                                                clockwise=False)
            candidates = [p for p in (cw, ccw, direct) if p]
            use_path = min(candidates, key=len)

        path = use_path[1:] if use_path and use_path[0] == (self.rx, self.ry) else use_path
        sweep = self._task_sweep_from(task, entry)
        if sweep:
            if path and path[-1] == sweep[0]:
                path.extend(sweep[1:])
            else:
                path.extend(sweep)
        return path

    # -------------------------------------------------------------------------
    # Algorithm 4: HCPP 主循环
    # -------------------------------------------------------------------------

    def step(self):
        """
        HCPP 单步闭环:
        感知 → 增量维护 CTS → GTP 边缘优先排序 → LPP 生成局部路径 → 执行一步。
        """
        import time
        t0 = time.time()
        self._step_count += 1
        # if (self.rx,self.ry) == (11,28):
        #     print(f"[DEBUG] Step 824 - GTP Sequence: {self._global_tour}")
        #     print(f"[DEBUG] Step 824 - Next Planned Path: {self._local_path}")
        #     print(f"[DEBUG] Step 824 - Robot Position: ({self.rx}, {self.ry})")
        #     from hcpp.visualization import plot_hcpp_cells
        #     import os
        #     os.makedirs("results/debug", exist_ok=True)
        #     plot_hcpp_cells(self.grid_map, self.cts, self.path, 1,
        #                     title=f"HCPP Cells - Step {self._step_count}",
        #                     save_path=f"results/debug/cells_step_824.png")
        #     return None

        if not self._initialized:
            return None

        if not self._boundary_explored:
            self._run_initial_boundary_exploration()
            self.comp_time += time.time() - t0
            return (self.rx, self.ry)

        self._sense()
        self._update_tasks_after_sensing()

        obstacle_info = self._side_unknown_obstacle()
        if obstacle_info is not None:
            _, _, obstacle_key, keep_wall, initial_dir = obstacle_info
            self._run_obstacle_boundary_exploration(
                obstacle_key, keep_wall=keep_wall, initial_dir=initial_dir)
            self.comp_time += time.time() - t0
            return (self.rx, self.ry)

        if self._is_done():
            self.comp_time += time.time() - t0
            return None

        # 当前路径耗尽时，按全局巡游取下一个单元重新规划。
        while not self._local_path:
            target_task = self._pop_next_task()
            if target_task is None:
                self._local_path = self._fallback_frontier_path()
                break
            self._local_path = self._plan_local_path_to_task(target_task)
            self._local_path_idx = 0
            if not self._local_path:
                target_task.completed = True

        if not self._local_path:
            self.comp_time += time.time() - t0
            return None

        if self._local_path_idx >= len(self._local_path):
            self._local_path = []
            self._local_path_idx = 0
            self.comp_time += time.time() - t0
            return (self.rx, self.ry)

        gx, gy = self._local_path[self._local_path_idx]
        if not self._try_move(gx, gy):
            self._local_path = []
            self._local_path_idx = 0
        else:
            # 运动后立即感知和维护 CTS，发现新障碍即触发下一轮重规划。
            self._sense()
            if self._update_tasks_after_sensing():
                self._local_path = []
                self._local_path_idx = 0
            obstacle_info = self._side_unknown_obstacle()
            if obstacle_info is not None:
                _, _, obstacle_key, keep_wall, initial_dir = obstacle_info
                self._run_obstacle_boundary_exploration(
                    obstacle_key, keep_wall=keep_wall,
                    initial_dir=initial_dir)
                self._local_path = []
                self._local_path_idx = 0

        self.comp_time += time.time() - t0
        return (self.rx, self.ry)



    def _generate_column_width_path(self, task):
        """
        为单个单元生成该列的覆盖路径。

        单元划分已在障碍物处断开，列内均为 FREE 格子，
        直接从一端到另一端逐格生成路径即可。
        使用 A* 前往列的起始端点。
        """
        gm = self.grid_map
        col_x = task.px

        # 确定机器人靠近单元的上端还是下端
        dist_upper = abs(self.rx - col_x) + abs(self.ry - task.pu)
        dist_lower = abs(self.rx - col_x) + abs(self.ry - task.pd)

        if dist_upper <= dist_lower:
            start_y = task.pu
            end_y = task.pd
        else:
            start_y = task.pd
            end_y = task.pu

        col_step = -1 if start_y > end_y else 1

        # A* 前往列的起始端点
        path = []
        move_path = self._astar(self.rx, self.ry, col_x, start_y,
                                allow_unknown=True)
        if move_path:
            path.extend(move_path)

        # 列内逐格扫描，只添加未覆盖的格子
        cur_y = start_y
        while True:
            next_y = cur_y + col_step
            if (col_step > 0 and next_y > end_y) or \
               (col_step < 0 and next_y < end_y):
                break
            if gm.is_valid(col_x, next_y) and not gm.is_obstacle(col_x, next_y):
                if gm.grid[next_y, col_x] != GridMap.COVERED:
                    path.append((col_x, next_y))
            cur_y = next_y
            if cur_y == end_y:
                break

        return path

    def _generate_single_task_path(self, task):
        """为单个单元生成该列的覆盖路径。任务完成状态由 _cover_nearby 实时更新。"""
        if task.completed:
            return []
        return self._generate_column_width_path(task)

    def _is_done(self):
        """终止条件: 检查是否还有 FREE 或 UNKNOWN 格子"""
        gm = self.grid_map
        if not np.any(gm.grid == GridMap.FREE) and not np.any(gm.grid == GridMap.UNKNOWN):
            return True
        return False

    def _count_reachable_free(self):
        """BFS 计数从机器人位置可达的 FREE 格子"""
        gm = self.grid_map
        count = 0
        queue = deque([(self.rx, self.ry)])
        visited = {(self.rx, self.ry)}
        while queue:
            gx, gy = queue.popleft()
            if gm.grid[gy, gx] == GridMap.FREE:
                count += 1
            for nx, ny in gm.get_neighbors_4(gx, gy):
                if (nx, ny) in visited:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                visited.add((nx, ny))
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                    queue.append((nx, ny))
        return count

    def _count_reachable_unknown(self):
        """BFS 检查是否有UNKNOWN格子可通过FREE/COVERED区域接触到"""
        gm = self.grid_map
        queue = deque([(self.rx, self.ry)])
        visited = {(self.rx, self.ry)}
        while queue:
            gx, gy = queue.popleft()
            for nx, ny in gm.get_neighbors_4(gx, gy):
                if (nx, ny) in visited:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                visited.add((nx, ny))
                if gm.is_unknown(nx, ny):
                    return 1  # 至少一个未知格子可达
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                    queue.append((nx, ny))
        return 0

    def _find_frontier(self):
        """BFS 寻找最近前沿 (FREE/COVERED 邻接 UNKNOWN 的格子)"""
        gm = self.grid_map
        queue = deque([(self.rx, self.ry)])
        visited = {(self.rx, self.ry)}
        while queue:
            gx, gy = queue.popleft()
            for nx, ny in gm.get_neighbors_4(gx, gy):
                if (nx, ny) in visited:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                visited.add((nx, ny))
                if gm.is_unknown(nx, ny):
                    return (gx, gy)  # 返回父格子
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                    queue.append((nx, ny))
        return None

    def _find_nearest_free(self):
        """BFS 寻找最近的 FREE 格子 (优先于前沿探索)"""
        gm = self.grid_map
        queue = deque([(self.rx, self.ry)])
        visited = {(self.rx, self.ry)}
        while queue:
            gx, gy = queue.popleft()
            if gm.grid[gy, gx] == GridMap.FREE:
                return (gx, gy)
            for nx, ny in gm.get_neighbors_4(gx, gy):
                if (nx, ny) in visited:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                visited.add((nx, ny))
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                    queue.append((nx, ny))
        return None

    def _backtrack_path(self):
        """
        回溯路径: 当机器人陷入死胡同时，沿历史路径回退到有出口的位置。
        """
        if len(self.path) < 2:
            return []
        # 从当前位置沿路径历史向回查找，找到第一个有开阔出口的位置
        for i in range(len(self.path) - 2, -1, -3):
            hx, hy = self.path[i]
            open_count = 0
            for nx, ny in self.grid_map.get_neighbors_4(hx, hy):
                if not self.grid_map.is_obstacle(nx, ny):
                    open_count += 1
            if open_count >= 3:
                backtrack = list(reversed(self.path[i+1:]))
                if backtrack:
                    return backtrack
                return []
        # 找不到开阔的位置，回退到路径历史早期位置
        if len(self.path) > 10:
            return list(reversed(self.path[1:]))
        return []

    # -------------------------------------------------------------------------
    # 通用工具方法
    # -------------------------------------------------------------------------

    def _try_move(self, gx, gy):

        if not self.grid_map.is_valid(gx, gy):
            return False
        if self.grid_map.is_obstacle(gx, gy):
            return False

        # 执行移动
        self.rx, self.ry = gx, gy
        self.path.append((gx, gy))
        # if 320<len(self.path) < 330:
        #     print("局部规划")
        self._visited_cells.add((gx, gy))
        self._local_path_idx += 1
        # 标记为已覆盖
        self._cover_nearby()
        return True

    def _astar(self, sx, sy, gx, gy, allow_unknown=False):
        """A* 路径规划 (8邻域)
        allow_unknown=True 时允许穿越 UNKNOWN 格子 (用于前沿探索)
        """
        gm = self.grid_map
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        open_set = [(0, (sx, sy))]
        came_from = {}
        g_score = {(sx, sy): 0}
        closed = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == (gx, gy):
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            for nx, ny in gm.get_neighbors_8(cur[0], cur[1]):
                if (nx, ny) in closed:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                if not allow_unknown and not gm.is_explored(nx, ny):
                    continue

                move_cost = 1.414 if nx != cur[0] and ny != cur[1] else 1.0
                t = g_score[cur] + move_cost

                if (nx, ny) not in g_score or t < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = t
                    came_from[(nx, ny)] = cur
                    h = abs(nx - gx) + abs(ny - gy)
                    heapq.heappush(open_set, (t + h, (nx, ny)))

        return None

    def _astar_4(self, sx, sy, gx, gy, allow_unknown=False):
        """A* 路径规划 (4邻域，仅上下左右)
        用于边界跟随等需要沿网格线行走的场景
        allow_unknown=True 时允许穿越 UNKNOWN 格子
        """
        gm = self.grid_map
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        open_set = [(0, (sx, sy))]
        came_from = {}
        g_score = {(sx, sy): 0}
        closed = set()

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == (gx, gy):
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            for nx, ny in gm.get_neighbors_4(cur[0], cur[1]):
                if (nx, ny) in closed:
                    continue
                if gm.is_obstacle(nx, ny):
                    continue
                if not allow_unknown and not gm.is_explored(nx, ny):
                    continue

                move_cost = 1.0
                t = g_score[cur] + move_cost

                if (nx, ny) not in g_score or t < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = t
                    came_from[(nx, ny)] = cur
                    h = abs(nx - gx) + abs(ny - gy)
                    heapq.heappush(open_set, (t + h, (nx, ny)))

        return None

    def run(self, max_steps=30000):
        """运行规划器直到完成"""
        for _ in range(max_steps):
            if self.step() is None:
                break
        return self.path
