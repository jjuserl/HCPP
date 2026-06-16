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

        # 初始感知 (仅发现，不标记覆盖)
        self._sense()

        # 初始单元分解 + GTP 将在边界探索之后进行
        # (in _do_boundary_exploration completion)
        self.cts = []
        self._global_tour = []
        self._local_path = []
        self._local_path_idx = 0

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
        print("调用了绕障算法")
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


            # 绕障模式，检查规划的格子是否已经是覆盖了，若是则结束
            print(f"成功进入障碍边界检测，第{t}个格子:", self.rx, self.ry)
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
                # print(f"第{t}个格子:", self.rx, self.ry)
                if 320<len(self.path) < 330 :
                    print("绕障算法")
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

        每步从所有未完成单元中选:
          1. 边缘节点优先 — 按邻接数 (adj) 从小到大排序
          2. 同度数按距离当前列的列距离从小到大排序
        动态更新: 选完后移除该节点，邻居度数自动变化。
        """
        active_tasks = [t for t in self.cts if not t.completed]
        if not active_tasks:
            return []

        # 建立邻接图: id → [邻居id列表] (仅活跃邻居)
        adj_dict = {}
        for task in active_tasks:
            adj_dict[task.id] = [n for n in task.adj
                                 if self._get_task_by_id(n)
                                 and not self._get_task_by_id(n).completed]

        # 预计算连通分量（供 _cells_connected 使用）
        components = []
        visited_comp = set()
        for tid in adj_dict:
            if tid in visited_comp:
                continue
            comp = self._get_graph_component(tid, adj_dict)
            components.append(comp)
            visited_comp |= comp
        self._components = components

        # 当前列位置（初始为机器人所在列）
        cur_px = self.rx

        tour = []
        visited = set()

        while len(tour) < len(active_tasks):
            # 所有未访问单元
            candidates = [t for t in active_tasks if t.id not in visited]
            if not candidates:
                break

            # 按 (邻接数, 列距离) 选最优单元
            best = min(candidates,
                       key=lambda t: (len(adj_dict.get(t.id, [])),
                                      abs(t.px - self.rx) + min(abs(t.pu - self.ry), abs(t.pd - self.ry))))

            tour.append(best.id)
            visited.add(best.id)
            cur_px = best.px

            # 从邻接图中移除，更新邻居度数
            for nid in adj_dict.get(best.id, []):
                if nid in adj_dict:
                    adj_dict[nid] = [n for n in adj_dict[nid]
                                     if n != best.id]

        # self._first_global = True
        # from hcpp.visualization import plot_hcpp_cells
        # import os
        # os.makedirs("results/test", exist_ok=True)
        # print("GTP重新生成后打印",self._step_count)
        # plot_hcpp_cells(self.grid_map, self.cts, self.path, 1,
        #                 title=f"HCPP Cell Decomposition - Step {self._step_count}",
        #                 save_path=f"results/test/cells_step_{self._step_count}.png")
        # print(f"[DEBUG] Global Tour IDs: {self._global_tour}")
        return tour

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

    # -------------------------------------------------------------------------
    # Algorithm 4: HCPP 主循环
    # -------------------------------------------------------------------------

    def step(self):
        """
        执行单步规划 (重构后):

        1. 边界探索 (首次 / 绕障触发) — 含更新单元与序列生成、路径加入
        2. 感知环境 (每步)
        3. 生成局部路径 (含过滤已完成任务 + 去往单元 + 单元内部覆盖)
        4. 沿路径走一步 → _cover_nearby
        5. 新障碍物被感知 → 更新 CTS + 重新生成 GTP
        6. 检查终止条件 (每10步)
        7. 检测是否需绕障 → 若是，调用 1
        """
        import time
        t0 = time.time()
        self._step_count += 1

        # ---- 1. 边界探索 (首次) 并生成全局序列 ----
        if not self._boundary_explored:
            self._first_global = True
            self._init_cell_decomposition()
            # # 绘制初始单元划分 (未探索前，每列一个单元)
            # from hcpp.visualization import plot_hcpp_cells
            # import os
            # os.makedirs("results", exist_ok=True)
            # plot_hcpp_cells(self.grid_map, self.cts, None, 1,
            #                 title="HCPP Initial Cell Decomposition",
            #                 save_path="results/test/initial_cells.png")
            # print("[DEBUG] Initial cell decomposition saved to results/test/initial_cells.png")
            self._do_boundary_exploration(clockwise=True)
            self._step_count = len(self.path)
            self.comp_time += time.time() - t0

            # # ---- 调试: 边界探索完成后绘制三张图 ----
            # import copy
            # from hcpp.visualization import plot_map, plot_map_with_direction, plot_hcpp_cells
            # import os
            # os.makedirs("results", exist_ok=True)
            # merged_map = copy.deepcopy(self.global_map)
            # merged_map.grid[self.grid_map.grid == GridMap.COVERED] = GridMap.COVERED
            # plot_map(merged_map, self.path,
            #          title="HCPP - Boundary Exploration",
            #          save_path="results/test/boundary_explored_map.png")
            # plot_map_with_direction(merged_map, self.path,
            #                         title="HCPP - Path with Direction (After Boundary Exploration)",
            #                         save_path="results/test/boundary_explored_direction.png",
            #                         arrow_interval=5)
            # plot_hcpp_cells(merged_map, self.cts, self.path, 1,
            #                 title="HCPP Cell Decomposition - After Boundary Exploration",
            #                 save_path="results/test/boundary_explored_cells.png")
            # print("[DEBUG] Boundary exploration visualization saved to results/test/")

        # ---- 2. 感知环境 ----
        # 记录感知前障碍物数量，用于检测新障碍物
        obstacle_count_before = int(np.sum(self.grid_map.grid == GridMap.OBSTACLE))
        self._sense()
        new_obstacles_sensed = self._new_obstacle_flag

        # ---- 3. 生成局部路径 (两场景分发) ----
        if not self._local_path and self._global_tour:
            
            # 只剩最后一个单元，直接判断是否完成并返回
            if len(self._global_tour) <= 1 :
                c_id = self._global_tour[0]
                c_task = self._get_task_by_id(c_id)
                if c_task.completed:
                    return None
                    
            else :
                # 处理掉第一个未访问的序列
                if  self._first_global:
                    self._first_global = False
                    target_tid = self._global_tour[0]
                    target_task = self._get_task_by_id(target_tid)
                    self._local_path = self._plan_disconnected_path(None, target_task)

                # 第一个序列变为已完成，正常流程处理下一个序列
                else :
                    # 以防万一，处理成第一个已完成，第二个未完成的情况
                    first_tid = None
                    first_task = None
                    popped = False
                    while len(self._global_tour) > 2:
                        first_tid = self._global_tour[0]
                        first_task = self._get_task_by_id(first_tid)
                        next_tid = self._global_tour[1]
                        next_task = self._get_task_by_id(next_tid)
                        if first_task and first_task.completed:
                            if  next_task.completed:
                                self._global_tour.pop(0)
                            elif not next_task.completed:
                                break
                        else:
                            break

                    # target_tid = self._global_tour[1]
                    # target_task = self._get_task_by_id(target_tid)
                    target_tid = next_tid
                    target_task = next_task

                    if target_task and not target_task.completed:

                        # 刚覆盖完的单元 vs 目标单元，按 adj 分场景
                        if target_tid in first_task.adj:
                            # 场景1: 相邻 → A*到最近端点 + 直线覆盖
                            self._local_path = self._plan_disconnected_path(
                                None, target_task)
                                
                        else:
                            # 场景2: 不相邻但连通 → 边界绕行
                            # 检查四邻域是否有 FREE 格子，有则先移动过去
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = self.rx + dx, self.ry + dy
                                if self.grid_map.grid[nx, ny] == GridMap.FREE:
                                    self.rx, self.ry = nx, ny
                                    self.path.append((self.rx, self.ry))
                                    break
                            self._local_path = self._plan_boundary_path(
                                first_task, target_task)
                        # else:
                        #     # 初始: 没有上一个覆盖完的单元
                        #     self._local_path = self._plan_disconnected_path(
                        #         None, target_task)
                        if (self._local_path and self._local_path[0] == (self.rx, self.ry)):
                            self._local_path = self._local_path[1:]
                            self._local_path_idx = 0
                    # if next_tid == 8:
                    #     print("开始覆盖id==8的单元")
                    #     print("当前路径:", self._local_path)


        # ---- 4. 沿路径走一步，置为 COVERED ----
        if  self._local_path:
            old_x = self.rx
            if self._local_path_idx < len(self._local_path):
                target = self._local_path[self._local_path_idx]
                if self._try_move(target[0], target[1]):
                    self._local_path_idx += 0
                else:
                    # 移动失败: 清除当前路径，下次重新规划
                    self._local_path = []
                    self._local_path_idx = 0
            else:
                # 路径已耗尽但未被清空
                self._local_path = []
                self._local_path_idx = 0
        self._sense()
        # if (self.rx,self.ry) == (1, 28):
        #     from hcpp.visualization import plot_hcpp_cells
        #     import os
        #     os.makedirs("results/test", exist_ok=True)
        #     print("步数到达打印",self._step_count,(self.rx,self.ry))
        #     plot_hcpp_cells(self.grid_map, self.cts, self.path, 1,
        #                     title=f"HCPP Cell Decomposition - Step {self._step_count}",
        #                     save_path=f"results/test/cells_step_{self._step_count}.png")
        #     print(f"[DEBUG] Global Tour IDs: {self._global_tour}")
        #     task_1 = self._get_task_by_id(1)
        #     print(f"[DEBUG] Task 1 covered?: {task_1.completed}")    
        #     print(f"[DEBUG] Task 1 adj: {task_1.adj}")
        #     task_2 = self._get_task_by_id(2)
        #     print(f"[DEBUG] Task 2 covered?: {task_2.completed}")    
        #     print(f"[DEBUG] Task 2 adj: {task_2.adj}")

        # if (self.rx,self.ry) == (2, 28):
        #     from hcpp.visualization import plot_hcpp_cells
        #     import os
        #     os.makedirs("results/test", exist_ok=True)
        #     print("步数到达打印",self._step_count,(self.rx,self.ry))
        #     plot_hcpp_cells(self.grid_map, self.cts, self.path, 1,
        #                     title=f"HCPP Cell Decomposition - Step {self._step_count}",
        #                     save_path=f"results/test/cells_step_{self._step_count}.png")
        #     print(f"[DEBUG] Global Tour IDs: {self._global_tour}")
        #     task_1 = self._get_task_by_id(1)
        #     print(f"[DEBUG] Task 1 covered?: {task_1.completed}")    
        #     print(f"[DEBUG] Task 1 adj: {task_1.adj}")
        #     task_2 = self._get_task_by_id(2)
        #     print(f"[DEBUG] Task 2 covered?: {task_2.completed}")    
        #     print(f"[DEBUG] Task 2 adj: {task_2.adj}")
        #     # if task_13:
            #     print(f"[DEBUG] Task 13 adj: {task_13.adj}")
            # return None
        # if (self.rx,self.ry) == (3,28):
        #     return None
        # ---- 5. 新障碍物被感知 → 仅更新障碍物涉及的列 ----
        if  new_obstacles_sensed:
            has_changes = self._update_cell_decomposition(
                columns=self._obstacle_changed_xs)
            if has_changes:
                self._global_tour = self._generate_global_tour()
                self._local_path = []
                self._local_path_idx = 0

        # ---- 6. 检查终止条件 (每50步) ----
        if self._step_count % 50 == 0:
            td = time.time()
            if self._is_done():
                self.comp_time += time.time() - t0
                return None

        # ---- 7. 检测是否需绕障 (绕障调用 1) ----#####z这里还有bug#########---
        # if self._local_path and self._local_path_idx < len(self._local_path):
        rx, ry = self.rx, self.ry
        # 根据前进方向检查侧方是否有未知边界障碍
        if self.grid_map.is_obstacle(rx + 1, ry):
            print("进阶条件")
            print("绕障时局部路径数",len(self._local_path))
            next_idx = self._local_path.index((rx,ry)) + 1
            print("下一个目标索引:",next_idx)
            return None
            next_cell = self._local_path[next_idx]
            dx_move = next_cell[0] - rx
            dy_move = next_cell[1] - ry # 大于0往上，小于0往下

            #与绕边界相反，往下时，障碍物要在左侧，逆时针
            clockwiseP = False if dy_move < 0 else True 
            # print(f"  [Step {self._step_count}] {side_name} obstacle "
            #           f"({check_x},{check_y}) bordering unknown, "
            #           f"boundary following...")
            self._do_obstacle_exploration(clockwise=clockwiseP)
            self._first_global = True


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
