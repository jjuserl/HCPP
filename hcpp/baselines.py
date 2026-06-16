"""
对比基线算法 (Baseline Algorithms)

- EpsilonStar (ε*): 多分辨率势场在线覆盖 (Song & Gupta, 2018)
- BSA: 回溯蛇形覆盖 (Gonzalez et al., 2005)
- FS-STC: 在线生成树环绕覆盖 (Gabriely & Rimon, 2003)
- SP2E: 8邻域螺旋覆盖 (Li et al., 2023)

所有算法均在未知环境中在线运行: 感知→移动→覆盖，一步一格。
"""

import time
import numpy as np
from collections import deque
import heapq
from .map_grid import GridMap


# =============================================================================
# 通用工具函数
# =============================================================================

_DIRS4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # N, E, S, W
_DIRS8 = [(0, -1), (1, -1), (1, 0), (1, 1),
          (0, 1), (-1, 1), (-1, 0), (-1, -1)]  # N, NE, E, SE, S, SW, W, NW


def _is_passable(gm, gx, gy):
    return gm.is_valid(gx, gy) and not gm.is_obstacle(gx, gy)


def _is_uncovered(gm, gx, gy):
    if not gm.is_valid(gx, gy):
        return False
    return gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN)


def _astar(gm, sx, sy, gx, gy, diag=True, allow_unknown=False):
    """A* 路径规划，支持 4 或 8 邻域
    allow_unknown=True 时允许穿越 UNKNOWN 格子 (用于探索)"""
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]
    nbrs = gm.get_neighbors_8 if diag else gm.get_neighbors_4
    open_set = [(0, (sx, sy))]
    came = {}
    g = {(sx, sy): 0}
    closed = set()
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gx, gy):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path))
        for nx, ny in nbrs(cur[0], cur[1]):
            if (nx, ny) in closed or not _is_passable(gm, nx, ny):
                continue
            if not allow_unknown and not gm.is_explored(nx, ny):
                continue
            cost = 1.414 if (nx != cur[0] and ny != cur[1]) else 1.0
            t = g[cur] + cost
            if (nx, ny) not in g or t < g[(nx, ny)]:
                g[(nx, ny)] = t
                came[(nx, ny)] = cur
                h = max(abs(nx - gx), abs(ny - gy)) if diag else abs(nx - gx) + abs(ny - gy)
                heapq.heappush(open_set, (t + h, (nx, ny)))
    return None


def _bfs_path(gm, sx, sy, gx, gy):
    """BFS 最短路径 (4 邻域)"""
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]
    q = deque([(sx, sy)])
    par = {(sx, sy): None}
    while q:
        cur = q.popleft()
        if cur == (gx, gy):
            p = []
            c = cur
            while c is not None:
                p.append(c)
                c = par[c]
            return list(reversed(p))
        for dx, dy in _DIRS4:
            nx, ny = cur[0] + dx, cur[1] + dy
            if (nx, ny) in par or not _is_passable(gm, nx, ny):
                continue
            par[(nx, ny)] = cur
            q.append((nx, ny))
    return None


def _bfs_frontier(gm, sx, sy):
    """BFS 寻找最近的前沿格子 (已探索→未知的边界)"""
    q = deque([(sx, sy)])
    vis = {(sx, sy)}
    while q:
        gx, gy = q.popleft()
        for dx, dy in _DIRS4:
            nx, ny = gx + dx, gy + dy
            if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                continue
            vis.add((nx, ny))
            if gm.is_unknown(nx, ny):
                return (gx, gy)
            if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                q.append((nx, ny))
    return None


def _is_done(gm, rx, ry):
    """终止条件: 无 FREE 且无可达 UNKNOWN"""
    if np.any(gm.grid == GridMap.FREE):
        return False
    if not np.any(gm.grid == GridMap.UNKNOWN):
        return True
    q = deque([(rx, ry)])
    vis = {(rx, ry)}
    while q:
        gx, gy = q.popleft()
        for dx, dy in _DIRS4:
            nx, ny = gx + dx, gy + dy
            if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                continue
            vis.add((nx, ny))
            if gm.is_unknown(nx, ny):
                return False
            if gm.grid[ny, nx] == GridMap.COVERED:
                q.append((nx, ny))
    return True


def _sense(gm, global_map, rx, ry, sensor):
    """带遮挡检测的感知: 使用 sensor.simple_sense"""
    sensor.simple_sense(gm, global_map, rx, ry)


def _cover(gm, rx, ry):
    """走到才算覆盖: 仅标记当前格子"""
    if gm.is_valid(rx, ry) and gm.grid[ry, rx] == GridMap.FREE:
        gm.grid[ry, rx] = GridMap.COVERED


# =============================================================================
# Epsilon* — 多分辨率势场在线覆盖
# =============================================================================

class EpsilonStarPlanner:
    """
    ε* 规划器 (Song & Gupta, TRO 2018) — 行覆盖重构版

    双模态架构，严格切换，不混合:

    模态 1 — 直线覆盖模态 (LINE, 占 90%+ 路径)
      - 固定覆盖主方向 (Y 轴), 行间距 = 2r (覆盖宽度)
      - 机器人沿行方向走直线，批量标记覆盖带内已感知 FREE 格子
      - 不调用势场，直接沿行方向移动

    模态 2 — 行间跳转 / 极值逃逸 (TRANSITION, 短路径)
      - 行间跳转: 细尺度势场 + A* 规划到下一条未覆盖行
      - 极值逃逸: 粗尺度势场 (3/5/9 倍) 梯度逃离
    """

    def __init__(self, grid_map, global_map, sensor):
        self.grid_map = grid_map        # 机器人的认知地图
        self.global_map = global_map    # 全局真实地图（用于传感器仿真）
        self.sensor = sensor
        self.r = int(sensor.range_max / grid_map.resolution)
        self.robot_gx = 0
        self.robot_gy = 0
        self.path = []
        self.local_extremum_count = 0
        self.computation_time = 0.0
        self._step_count = 0
        self._stuck = 0
        # --- 双模态状态 ---
        self._mode = 'LINE'          # 'LINE' 或 'TRANSITION'
        self._line_dy = 1            # 行方向 (+1=Y+, -1=Y-)
        self._row_x = 0              # 当前行 X 坐标
        self._cached_path = []       # 跳转缓存路径
        self._row_step = max(1, 2 * self.r)  # 行间距 = 2r

    def initialize(self, start_gx, start_gy):
        self.robot_gx, self.robot_gy = start_gx, start_gy
        self.grid_map.set_free(start_gx, start_gy)
        self.grid_map.set_covered(start_gx, start_gy)
        self.path = [(start_gx, start_gy)]
        _sense(self.grid_map, self.global_map, start_gx, start_gy, self.sensor)
        self._row_x = start_gx
        self._mode = 'LINE'
        # 初始方向: 选空间更多的 Y 方向
        h = self.grid_map.height
        if start_gy > h // 2:
            self._line_dy = -1
        # （Epsilon* 已统一为单格覆盖，不再批量标记行带）

    def step(self):
        t0 = time.time()
        self._step_count += 1
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy

        # --- 感知 + 覆盖当前格 ---
        _sense(gm, self.global_map, rx, ry, self.sensor)
        _cover(gm, rx, ry)

        # --- 终止检查 ---
        if self._step_count % 10 == 0 and _is_done(gm, rx, ry):
            self.computation_time += time.time() - t0
            return None

        # ============================================================
        #  模态 1: 直线覆盖 (LINE)
        # ============================================================
        if self._mode == 'LINE':
            # --- 1a. 沿缓存绕行路径前进 (遇到障碍时使用) ---
            if self._cached_path and len(self._cached_path) > 1:
                nxt = self._cached_path[1]
                self._cached_path = self._cached_path[1:]
                if _is_passable(gm, nxt[0], nxt[1]):
                    self.robot_gx, self.robot_gy = nxt[0], nxt[1]
                    self.path.append((nxt[0], nxt[1]))
                    _cover(gm, nxt[0], nxt[1])
                    self._stuck = 0
                    if len(self._cached_path) == 1:
                        self._cached_path = []
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)
                self._cached_path = []

            # --- 1b. 直线前进 (沿列方向) ---
            ny = ry + self._line_dy
            if _is_passable(gm, rx, ny) and gm.grid[ny, rx] != GridMap.COVERED:
                self.robot_gy = ny
                self.path.append((rx, ny))
                _cover(gm, rx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

            # --- 1c. A* 绕行: 前方列带内还有未覆盖区域 (缓存完整路径, 禁用对角线) ---
            if self._has_uncovered_in_row(gm, rx, ry):
                target = self._find_row_target(gm, rx, ry)
                if target:
                    p = _astar(gm, rx, ry, target[0], target[1], diag=False, allow_unknown=True)
                    if p and len(p) > 1:
                        self._cached_path = p
                        nxt = p[1]
                        self._cached_path = self._cached_path[1:]
                        self.robot_gx, self.robot_gy = nxt
                        self.path.append(nxt)
                        _cover(gm, nxt[0], nxt[1])
                        self._stuck = 0
                        if len(self._cached_path) == 1:
                            self._cached_path = []
                        self.computation_time += time.time() - t0
                        return (self.robot_gx, self.robot_gy)

            # --- 1d. 行真的走完了 ---
            self._mode = 'TRANSITION'
            self._stuck = 0

        # ============================================================
        #  模态 2: 行间跳转 / 极值逃逸 (TRANSITION)
        # ============================================================

        # --- 2a. 沿缓存跳转路径前进 (走到终点才切回 LINE, 不在中途切换) ---
        if self._cached_path and len(self._cached_path) > 1:
            nxt = self._cached_path[1]
            self._cached_path = self._cached_path[1:]
            if _is_passable(gm, nxt[0], nxt[1]):
                self.robot_gx, self.robot_gy = nxt[0], nxt[1]
                self.path.append((nxt[0], nxt[1]))
                _cover(gm, nxt[0], nxt[1])
                self._stuck = 0
                # 只有走到路径终点才切回 LINE
                if len(self._cached_path) <= 1:
                    self._cached_path = []
                    self._mode = 'LINE'
                    self._line_dy = 1 if self.robot_gy < gm.height // 2 else -1
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)
            self._cached_path = []

        # --- 2b. 行间跳转: 细尺度势场 → A* 规划 (禁用对角线, 走到终点再切 LINE) ---
        target = self._select_target(gm, rx, ry)
        if target:
            p = _astar(gm, rx, ry, target[0], target[1], diag=False, allow_unknown=True)
            if p and len(p) > 1:
                self._cached_path = p
                nxt = p[1]
                self._cached_path = self._cached_path[1:]
                self.robot_gx, self.robot_gy = nxt
                self.path.append(nxt)
                _cover(gm, nxt[0], nxt[1])
                self._stuck = 0
                if len(self._cached_path) <= 1:
                    self._cached_path = []
                    self._mode = 'LINE'
                    self._line_dy = 1 if self.robot_gy < gm.height // 2 else -1
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 2c. 极值逃逸: 逐步加粗网格分辨率 (禁用对角线, 走到终点才切 LINE) ---
        self._stuck += 1
        self.local_extremum_count += 1

        if self._stuck >= 3:
            esc = self._coarse_escape()
            if esc:
                p = _astar(gm, rx, ry, esc[0], esc[1], diag=False, allow_unknown=True)
                if p and len(p) > 1:
                    self._cached_path = p
                    nxt = p[1]
                    self._cached_path = self._cached_path[1:]
                    self.robot_gx, self.robot_gy = nxt
                    self.path.append(nxt)
                    _cover(gm, nxt[0], nxt[1])
                    self._stuck = 0
                    if len(self._cached_path) <= 1:
                        self._cached_path = []
                        self._mode = 'LINE'
                        self._line_dy = 1 if self.robot_gy < gm.height // 2 else -1
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)

        # --- 2d. BFS 兜底 (禁用对角线, 走到终点才切 LINE) ---
        nearest = self._bfs_uncovered()
        if nearest:
            p = _astar(gm, rx, ry, nearest[0], nearest[1], diag=False, allow_unknown=True)
            if p and len(p) > 1:
                self._cached_path = p
                nxt = p[1]
                self._cached_path = self._cached_path[1:]
                self.robot_gx, self.robot_gy = nxt
                self.path.append(nxt)
                _cover(gm, nxt[0], nxt[1])
                self._stuck = 0
                if len(self._cached_path) <= 1:
                    self._cached_path = []
                    self._mode = 'LINE'
                    self._line_dy = 1 if self.robot_gy < gm.height // 2 else -1
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        if self._stuck > 500:
            self.computation_time += time.time() - t0
            return None

        self.computation_time += time.time() - t0
        return (self.robot_gx, self.robot_gy)

    # ---- 行覆盖判定 ----
    def _is_row_done(self, gm, rx, ry):
        """当前行是否已覆盖完: 前方行带内既无 FREE 也无 UNKNOWN"""
        r = self.r
        for dy_off in range(1, 4):
            cy = ry + self._line_dy * dy_off
            if not gm.is_valid(rx, cy):
                continue
            for dx in range(-r, r + 1):
                gx = rx + dx
                if not gm.is_valid(gx, cy):
                    continue
                if gm.is_obstacle(gx, cy):
                    continue
                s = gm.grid[cy, gx]
                if s == GridMap.FREE or s == GridMap.UNKNOWN:
                    return False
        return True

    def _has_uncovered_in_row(self, gm, rx, ry):
        """当前行前方是否还有未覆盖格子 (用于判断是否需要绕行)"""
        r = self.r
        for dy_off in range(1, r + 2):
            cy = ry + self._line_dy * dy_off
            if not gm.is_valid(rx, cy):
                continue
            for dx in range(-r, r + 1):
                gx = rx + dx
                if not gm.is_valid(gx, cy):
                    continue
                if gm.grid[cy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                    return True
        return False

    def _find_row_target(self, gm, rx, ry):
        """在当前行带内找最近的未覆盖格子 (用于 A* 绕行目标)"""
        r = self.r
        best, best_d = None, float('inf')
        for dy_off in range(1, r + 3):
            cy = ry + self._line_dy * dy_off
            if not gm.is_valid(rx, cy):
                continue
            for dx in range(-r, r + 1):
                gx = rx + dx
                if not gm.is_valid(gx, cy):
                    continue
                s = gm.grid[cy, gx]
                if s in (GridMap.FREE, GridMap.UNKNOWN) and not gm.is_obstacle(gx, cy):
                    d = abs(gx - rx) + abs(cy - ry)
                    if d < best_d:
                        best_d, best = d, (gx, cy)
        return best

    # ---- 行间跳转目标选择 ----
    def _select_target(self, gm, rx, ry):
        """基于细尺度势场找到最近未覆盖格子 (跳转目标)
        同时考虑 FREE 和 UNKNOWN 格子"""
        pot = self._potential_field(gm)

        best, best_s = None, float('inf')

        if pot:
            for (gx, gy), d in pot.items():
                if not _is_passable(gm, gx, gy):
                    continue
                s = gm.grid[gy, gx]
                if s == GridMap.FREE:
                    score = d
                elif s == GridMap.COVERED:
                    score = d + 50
                else:
                    continue
                if score < best_s:
                    best_s, best = score, (gx, gy)

        # 同时搜索 UNKNOWN 前沿格子 (FREE↔UNKNOWN 边界旁的 UNKNOWN)
        h, w = gm.grid.shape
        for gy in range(h):
            for gx in range(w):
                if not gm.is_unknown(gx, gy):
                    continue
                # 检查是否邻接 FREE/COVERED (前沿)
                is_frontier = False
                for dx, dy in _DIRS4:
                    nx, ny = gx + dx, gy + dy
                    if gm.is_valid(nx, ny) and gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                        is_frontier = True
                        break
                if not is_frontier:
                    continue
                d = abs(gx - rx) + abs(gy - ry)
                score = d + 5  # 略优于同距离的 FREE
                if score < best_s:
                    best_s, best = score, (gx, gy)

        if best is None:
            return self._bfs_nearest_uncovered()
        return best

    # ---- 势场 ----
    def _potential_field(self, gm):
        """多源 BFS 势场: 从 FREE↔UNKNOWN 边界向外扩展"""
        pot = {}
        q = deque()
        h, w = gm.grid.shape
        for gy in range(h):
            for gx in range(w):
                if gm.grid[gy, gx] != GridMap.FREE:
                    continue
                for dx, dy in _DIRS4:
                    nx, ny = gx + dx, gy + dy
                    if gm.is_valid(nx, ny) and gm.is_unknown(nx, ny):
                        pot[(gx, gy)] = 0
                        q.append((gx, gy))
                        break
        while q:
            gx, gy = q.popleft()
            d = pot[(gx, gy)]
            for dx, dy in _DIRS4:
                nx, ny = gx + dx, gy + dy
                if not gm.is_valid(nx, ny) or gm.is_obstacle(nx, ny):
                    continue
                if (nx, ny) in pot:
                    continue
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
                    pot[(nx, ny)] = d + 1
                    q.append((nx, ny))
        return pot

    def _bfs_nearest_uncovered(self):
        """BFS 寻找最近的未覆盖格子 (FREE 或 UNKNOWN)"""
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        q = deque([(rx, ry)])
        vis = {(rx, ry)}
        while q:
            gx, gy = q.popleft()
            s = gm.grid[gy, gx]
            if s == GridMap.FREE or s == GridMap.UNKNOWN:
                return (gx, gy)
            for dx, dy in _DIRS4:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                    continue
                vis.add((nx, ny))
                q.append((nx, ny))
        return None

    # ---- 粗分辨率逃离 ----
    def _coarse_escape(self):
        """多分辨率势场: 逐步加粗直到找到逃离方向"""
        for block in [3, 5, 9]:
            target = self._escape_at_resolution(block)
            if target:
                return target
        return None

    def _escape_at_resolution(self, block):
        """在粗分辨率下构建势场，找逃离目标"""
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        h, w = gm.grid.shape
        bw, bh = (w + block - 1) // block, (h + block - 1) // block
        rbx, rby = rx // block, ry // block

        # 粗网格状态
        cg = np.full((bh, bw), -1, dtype=np.int8)
        for cy in range(bh):
            for cx in range(bw):
                has_free, has_obs = False, False
                for dy in range(block):
                    for dx in range(block):
                        gx, gy = cx * block + dx, cy * block + dy
                        if not gm.is_valid(gx, gy):
                            has_obs = True
                            continue
                        s = gm.grid[gy, gx]
                        if s == GridMap.OBSTACLE:
                            has_obs = True
                        elif s in (GridMap.FREE, GridMap.COVERED):
                            has_free = True
                if has_obs and not has_free:
                    cg[cy, cx] = GridMap.OBSTACLE
                elif has_free:
                    cg[cy, cx] = GridMap.FREE

        # 粗前沿 + BFS
        cq = deque()
        cp = {}
        for cy in range(bh):
            for cx in range(bw):
                if cg[cy, cx] != GridMap.FREE:
                    continue
                for dx, dy in _DIRS4:
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < bw and 0 <= ny < bh):
                        continue
                    if cg[ny, nx] == -1:
                        cp[(cx, cy)] = 0
                        cq.append((cx, cy))
                        break

        if not cp:
            return None

        while cq:
            cx, cy = cq.popleft()
            d = cp[(cx, cy)]
            for dx, dy in _DIRS4:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < bw and 0 <= ny < bh):
                    continue
                if (nx, ny) in cp or cg[ny, nx] == GridMap.OBSTACLE:
                    continue
                if cg[ny, nx] == GridMap.FREE:
                    cp[(nx, ny)] = d + 1
                    cq.append((nx, ny))

        # 找最近的粗前沿块 (非当前块)
        best_block, best_d = None, float('inf')
        for (cx, cy), d in cp.items():
            if d == 0 and (cx, cy) != (rbx, rby) and d < best_d:
                best_d, best_block = d, (cx, cy)
        if best_block is None:
            for (cx, cy), d in cp.items():
                if d < best_d:
                    best_d, best_block = d, (cx, cy)
        if best_block is None:
            return None

        # 在该块中找最近未覆盖细格子
        bx, by = best_block
        best_fine, best_fd = None, float('inf')
        for dy in range(block):
            for dx in range(block):
                gx, gy = bx * block + dx, by * block + dy
                if not gm.is_valid(gx, gy) or gm.is_obstacle(gx, gy):
                    continue
                if gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                    d = abs(gx - rx) + abs(gy - ry)
                    if d < best_fd:
                        best_fd, best_fine = d, (gx, gy)
        return best_fine

    def _bfs_uncovered(self):
        """BFS 寻找最近未覆盖格子 (穿过覆盖区域)"""
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        q = deque([(rx, ry)])
        vis = {(rx, ry)}
        while q:
            gx, gy = q.popleft()
            if gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                return (gx, gy)
            for dx, dy in _DIRS8:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                    continue
                vis.add((nx, ny))
                q.append((nx, ny))
        return None

    # ---- 移动 ----
    def _move_to(self, gx, gy):
        gm = self.grid_map
        if not _is_passable(gm, gx, gy):
            return False
        if abs(gx - self.robot_gx) <= 1 and abs(gy - self.robot_gy) <= 1:
            self.robot_gx, self.robot_gy = gx, gy
            self.path.append((gx, gy))
            _cover(gm, gx, gy)
            return True
        p = _astar(gm, self.robot_gx, self.robot_gy, gx, gy, allow_unknown=True)
        if p and len(p) > 1:
            nx, ny = p[1]
            self.robot_gx, self.robot_gy = nx, ny
            self.path.append((nx, ny))
            _cover(gm, nx, ny)
            return True
        return False

    def run(self, max_steps=30000):
        for _ in range(max_steps):
            if self.step() is None:
                break
        return self.path


# =============================================================================
# BSA — 回溯蛇形覆盖
# =============================================================================

class BSAPlanner:
    """
    BSA 规划器 (Gonzalez et al., ICRA 2005)

    核心思想:
    1. 优先沿当前方向前进
    2. 无法前进时顺时针转向
    3. 四周均无法前进时，BFS 回溯到最近未覆盖格子
    4. 所有可通行格子均被覆盖时终止
    """

    def __init__(self, grid_map, global_map, sensor):
        self.grid_map = grid_map        # 机器人的认知地图
        self.global_map = global_map    # 全局真实地图（用于传感器仿真）
        self.sensor = sensor
        self.r = int(sensor.range_max / grid_map.resolution)
        self.robot_gx = 0
        self.robot_gy = 0
        self.path = []
        self.local_extremum_count = 0
        self.computation_time = 0.0
        self._step_count = 0
        self._dir = 0               # 当前方向索引 (0=N,1=E,2=S,3=W)
        self._stuck = 0

    def initialize(self, start_gx, start_gy):
        self.robot_gx, self.robot_gy = start_gx, start_gy
        self.grid_map.set_free(start_gx, start_gy)
        self.grid_map.set_covered(start_gx, start_gy)
        self.path = [(start_gx, start_gy)]
        _sense(self.grid_map, self.global_map, start_gx, start_gy, self.sensor)

    def step(self):
        t0 = time.time()
        self._step_count += 1
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy

        _sense(gm, self.global_map, rx, ry, self.sensor)
        _cover(gm, rx, ry)

        if self._step_count % 10 == 0 and _is_done(gm, rx, ry):
            self.computation_time += time.time() - t0
            return None

        # --- 1. 优先沿当前方向前进 ---
        for i in range(4):
            d = (self._dir + i) % 4
            dx, dy = _DIRS4[d]
            nx, ny = rx + dx, ry + dy
            if _is_passable(gm, nx, ny) and _is_uncovered(gm, nx, ny):
                self._dir = d
                self.robot_gx, self.robot_gy = nx, ny
                self.path.append((nx, ny))
                _cover(gm, nx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 2. 4 邻域都走不通，尝试 8 邻域未覆盖格子 ---
        for dx, dy in _DIRS8:
            nx, ny = rx + dx, ry + dy
            if _is_passable(gm, nx, ny) and _is_uncovered(gm, nx, ny):
                self.robot_gx, self.robot_gy = nx, ny
                self.path.append((nx, ny))
                _cover(gm, nx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 3. 死区: BFS 回溯到最近未覆盖格子 ---
        target = self._bfs_nearest_uncovered()
        if target:
            p = _astar(gm, rx, ry, target[0], target[1])
            if p and len(p) > 1:
                nx, ny = p[1]
                self.robot_gx, self.robot_gy = nx, ny
                self.path.append((nx, ny))
                _cover(gm, nx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 4. 完全无法移动 ---
        self._stuck += 1
        self.local_extremum_count += 1
        if self._stuck > 300:
            self.computation_time += time.time() - t0
            return None

        self.computation_time += time.time() - t0
        return (self.robot_gx, self.robot_gy)

    def _bfs_nearest_uncovered(self):
        """BFS 寻找最近未覆盖格子 (走过覆盖区域)"""
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        q = deque([(rx, ry)])
        vis = {(rx, ry)}
        while q:
            gx, gy = q.popleft()
            if gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                return (gx, gy)
            for dx, dy in _DIRS4:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                    continue
                vis.add((nx, ny))
                q.append((nx, ny))
        return None

    def run(self, max_steps=30000):
        for _ in range(max_steps):
            if self.step() is None:
                break
        return self.path


# =============================================================================
# FS-STC — 在线生成树环绕覆盖
# =============================================================================

class FSSTCPlanner:
    """
    FS-STC 规划器 (Gabriely & Rimon, 2003)

    核心思想:
    1. 将环境划分为 2×2 粗网格 (与覆盖宽度匹配)
    2. 在线 BFS 构建粗网格生成树
    3. 打开生成树边对应的内部边，添加跨边界边
    4. 沿子格连接图追踪 Hamiltonian 路径
    5. 发现新区域时动态更新生成树和路径

    子格编号 (CCW, grid y-up):
        3(TL) ── 2(TR)
          │        │
        0(BL) ── 1(BR)
    """

    def __init__(self, grid_map, global_map, sensor):
        self.grid_map = grid_map        # 机器人的认知地图
        self.global_map = global_map    # 全局真实地图（用于传感器仿真）
        self.sensor = sensor
        self.r = int(sensor.range_max / grid_map.resolution)
        self.robot_gx = 0
        self.robot_gy = 0
        self.path = []
        self.local_extremum_count = 0
        self.computation_time = 0.0
        self._step_count = 0
        self._cs = 2  # 粗网格大小
        self._tree_nodes = set()
        self._tree_parent = {}
        self._tree_children = {}
        self._cached_path = []
        self._path_idx = 0
        self._stuck = 0
        self._rebuild_interval = 100  # 每 N 步重建一次
        self._last_rebuild = 0

    def initialize(self, start_gx, start_gy):
        self.robot_gx, self.robot_gy = start_gx, start_gy
        self.grid_map.set_free(start_gx, start_gy)
        self.grid_map.set_covered(start_gx, start_gy)
        self.path = [(start_gx, start_gy)]
        _sense(self.grid_map, self.global_map, start_gx, start_gy, self.sensor)
        self._build_tree()
        self._compute_path()

    # ---- 粗网格工具 ----
    def _mc(self, gx, gy):
        return (gx // self._cs, gy // self._cs)

    def _sc(self, mc, i):
        """粗网格 mc 的第 i 个子格坐标"""
        cx, cy = mc
        x0, y0 = cx * self._cs, cy * self._cs
        return [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)][i]

    def _mc_passable(self, cx, cy):
        cs = self._cs
        for dx in range(cs):
            for dy in range(cs):
                if _is_passable(self.grid_map, cx * cs + dx, cy * cs + dy):
                    return True
        return False

    # ---- 生成树 ----
    def _build_tree(self):
        gm = self.grid_map
        cs = self._cs
        mcw = (gm.width + cs - 1) // cs
        mch = (gm.height + cs - 1) // cs

        start = self._mc(self.robot_gx, self.robot_gy)
        if not self._mc_passable(*start):
            for r in range(1, 8):
                found = False
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        c = (start[0] + dx, start[1] + dy)
                        if 0 <= c[0] < mcw and 0 <= c[1] < mch and self._mc_passable(*c):
                            start = c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        self._tree_nodes = {start}
        self._tree_parent = {start: None}
        self._tree_children = {start: []}

        q = deque([start])
        while q:
            node = q.popleft()
            for dx, dy in _DIRS4:
                nb = (node[0] + dx, node[1] + dy)
                if nb in self._tree_nodes:
                    continue
                if not (0 <= nb[0] < mcw and 0 <= nb[1] < mch):
                    continue
                if not self._mc_passable(*nb):
                    continue
                self._tree_nodes.add(nb)
                self._tree_parent[nb] = node
                self._tree_children[nb] = []
                self._tree_children[node].append(nb)
                q.append(nb)

    # ---- 构建 Hamiltonian 路径 ----
    def _compute_path(self):
        """
        STC 经典方法:
        1. 每个粗网格内部有 4 条 CCW 边 (0-1, 1-2, 2-3, 3-0)
        2. 生成树边打开边界: 删除 2 条内部边，添加 2 条跨边界边
        3. 追踪子格连接图中的 Hamiltonian 路径

        正确的跨边界边映射:
        - EAST/WEST: 删竖边 (1-2, 3-0)，加横边
        - NORTH/SOUTH: 删横边 (0-1, 2-3)，加竖边
        """
        gm = self.grid_map
        cs = self._cs

        def sc(mc, i):
            return self._sc(mc, i)

        # 构建子格邻接图
        adj = {}
        def add_e(a, b):
            adj.setdefault(a, [])
            adj.setdefault(b, [])
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
        def rm_e(a, b):
            if a in adj and b in adj[a]:
                adj[a].remove(b)
            if b in adj and a in adj[b]:
                adj[b].remove(a)

        def is_pass(*c):
            return _is_passable(gm, c[0], c[1])

        # 内部 CCW 边
        for mc in self._tree_nodes:
            cells = [sc(mc, i) for i in range(4)]
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if is_pass(*cells[i]) and is_pass(*cells[j]):
                    add_e(cells[i], cells[j])

        # 生成树边: 打开边界
        seen = set()
        for parent in self._tree_children:
            for child in self._tree_children[parent]:
                ek = tuple(sorted([parent, child]))
                if ek in seen:
                    continue
                seen.add(ek)
                dx, dy = child[0] - parent[0], child[1] - parent[1]
                p = [sc(parent, i) for i in range(4)]
                c = [sc(child, i) for i in range(4)]

                if dx == 1 and dy == 0:  # CHILD EAST
                    # 删除竖边, 加横边
                    rm_e(p[1], p[2]); rm_e(c[3], c[0])
                    if is_pass(*p[1]) and is_pass(*c[0]): add_e(p[1], c[0])
                    if is_pass(*p[2]) and is_pass(*c[3]): add_e(p[2], c[3])
                elif dx == -1 and dy == 0:  # CHILD WEST
                    rm_e(p[3], p[0]); rm_e(c[1], c[2])
                    if is_pass(*p[0]) and is_pass(*c[1]): add_e(p[0], c[1])
                    if is_pass(*p[3]) and is_pass(*c[2]): add_e(p[3], c[2])
                elif dx == 0 and dy == 1:  # CHILD NORTH (higher y)
                    # 共享边界在 parent 顶部(sub2,sub3) / child 底部(sub0,sub1)
                    # 删除横边: parent sub2-sub3, child sub0-sub1
                    rm_e(p[2], p[3]); rm_e(c[0], c[1])
                    # 加竖跨边: p3-c0, p2-c1
                    if is_pass(*p[3]) and is_pass(*c[0]): add_e(p[3], c[0])
                    if is_pass(*p[2]) and is_pass(*c[1]): add_e(p[2], c[1])
                elif dx == 0 and dy == -1:  # CHILD SOUTH (lower y)
                    rm_e(p[0], p[1]); rm_e(c[2], c[3])
                    if is_pass(*p[0]) and is_pass(*c[3]): add_e(p[0], c[3])
                    if is_pass(*p[1]) and is_pass(*c[2]): add_e(p[1], c[2])

        # 追踪所有路径/环 (度数 ≤ 2 的图 = 简单路径和环的集合)
        traced = set()
        segments = []

        def trace(start):
            seg = [start]
            if not adj.get(start):
                return seg
            prev, cur = start, adj[start][0]
            while cur != start and cur not in set(seg):
                seg.append(cur)
                nxt = None
                for nb in adj.get(cur, []):
                    if nb != prev:
                        nxt = nb
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
            return seg

        for node in sorted(adj.keys()):
            if node not in traced:
                seg = trace(node)
                for n in seg:
                    traced.add(n)
                segments.append(seg)

        # 收集所有需要覆盖的子格
        all_cells = set()
        for mc in self._tree_nodes:
            for i in range(4):
                c = sc(mc, i)
                if is_pass(*c):
                    all_cells.add(c)

        # 添加孤立子格
        for ic in sorted(all_cells - traced):
            segments.append([ic])

        # 从机器人位置开始，贪心连接各 segment
        self._cached_path = self._connect_segments(segments, adj, all_cells)
        self._path_idx = 0

    def _connect_segments(self, segments, adj, all_cells):
        """从起点贪心连接各 segment 为一条完整路径"""
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy

        def bfs_path(s, t):
            if s == t: return [s]
            q = deque([s])
            par = {s: None}
            while q:
                cur = q.popleft()
                if cur == t:
                    p = []
                    c = cur
                    while c is not None:
                        p.append(c)
                        c = par[c]
                    return list(reversed(p))
                for dx, dy in _DIRS4:
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in par or not _is_passable(gm, nb[0], nb[1]):
                        continue
                    par[nb] = cur
                    q.append(nb)
            return None

        visited = set()
        final = [(rx, ry)]
        visited.add((rx, ry))

        # 找最近的 segment 起点
        remaining = list(segments)
        while remaining:
            last = final[-1]
            best_si, best_ep, best_d = -1, None, 9999
            for si, seg in enumerate(remaining):
                for c in seg:
                    if c in visited:
                        continue
                    d = abs(last[0] - c[0]) + abs(last[1] - c[1])
                    if d < best_d:
                        best_d, best_si, best_ep = d, si, c
            if best_si < 0:
                break

            seg = remaining.pop(best_si)

            # BFS 走到 segment 入口
            if final[-1] != best_ep:
                conn = bfs_path(final[-1], best_ep)
                if conn:
                    for c in conn[1:]:
                        final.append(c)
                        visited.add(c)
                else:
                    final.append(best_ep)
                    visited.add(best_ep)

            # 沿 adj 追踪 segment
            seg_set = set(seg)
            curr, prev_n = best_ep, None
            local_vis = {best_ep}
            # 选覆盖更多的方向
            best_local = [best_ep]
            for first_nb in adj.get(best_ep, []):
                lv = [best_ep, first_nb]
                pv, cv = best_ep, first_nb
                lv_set = {best_ep, first_nb}
                while True:
                    found = False
                    for nb in adj.get(cv, []):
                        if nb != pv and nb in seg_set and nb not in lv_set:
                            lv.append(nb)
                            lv_set.add(nb)
                            pv, cv = cv, nb
                            found = True
                            break
                    if not found:
                        break
                if len(lv) > len(best_local):
                    best_local = lv

            for c in best_local[1:]:
                if abs(final[-1][0] - c[0]) + abs(final[-1][1] - c[1]) == 1:
                    final.append(c)
                    visited.add(c)
                else:
                    conn = bfs_path(final[-1], c)
                    if conn:
                        for cc in conn[1:]:
                            final.append(cc)
                            visited.add(cc)
                    else:
                        final.append(c)
                        visited.add(c)

            # 补充 segment 中未追踪的子格
            for c in seg:
                if c not in visited:
                    if abs(final[-1][0] - c[0]) + abs(final[-1][1] - c[1]) == 1:
                        final.append(c)
                        visited.add(c)
                    else:
                        conn = bfs_path(final[-1], c)
                        if conn:
                            for cc in conn[1:]:
                                final.append(cc)
                                visited.add(cc)

        # 补充漏掉的子格
        for c in sorted(all_cells - visited):
            conn = bfs_path(final[-1], c)
            if conn:
                for cc in conn[1:]:
                    final.append(cc)

        return final if final else [(rx, ry)]

    # ---- 主循环 ----
    def step(self):
        t0 = time.time()
        self._step_count += 1
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy

        _sense(gm, self.global_map, rx, ry, self.sensor)

        if self._step_count % 10 == 0 and _is_done(gm, rx, ry):
            self.computation_time += time.time() - t0
            return None

        # 定期重建生成树 (发现新区域)
        if self._step_count - self._last_rebuild > self._rebuild_interval:
            self._last_rebuild = self._step_count
            old_count = len(self._tree_nodes)
            self._build_tree()
            if len(self._tree_nodes) > old_count:
                self._compute_path()

        # 沿缓存路径前进
        if self._cached_path and self._path_idx < len(self._cached_path):
            target = self._cached_path[self._path_idx]
            if target == (rx, ry):
                self._path_idx += 1
                if self._path_idx < len(self._cached_path):
                    target = self._cached_path[self._path_idx]

            if target != (rx, ry):
                d = abs(target[0] - rx) + abs(target[1] - ry)
                if d == 1 and _is_passable(gm, target[0], target[1]):
                    self.robot_gx, self.robot_gy = target
                    self.path.append(target)
                    _cover(gm, target[0], target[1])
                    self._path_idx += 1
                    self._stuck = 0
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)
                elif d > 1 or not _is_passable(gm, target[0], target[1]):
                    # 重新对齐
                    p = _astar(gm, rx, ry, target[0], target[1])
                    if p and len(p) > 1:
                        nx, ny = p[1]
                        self.robot_gx, self.robot_gy = nx, ny
                        self.path.append((nx, ny))
                        _cover(gm, nx, ny)
                        if (nx, ny) == target:
                            self._path_idx += 1
                        self._stuck = 0
                        self.computation_time += time.time() - t0
                        return (self.robot_gx, self.robot_gy)

        # 缓存路径走完或失效: 找最近未覆盖
        target = self._bfs_uncovered()
        if target:
            p = _astar(gm, rx, ry, target[0], target[1])
            if p and len(p) > 1:
                nx, ny = p[1]
                self.robot_gx, self.robot_gy = nx, ny
                self.path.append((nx, ny))
                _cover(gm, nx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # 前沿探索
        front = _bfs_frontier(gm, rx, ry)
        if front:
            p = _astar(gm, rx, ry, front[0], front[1])
            if p and len(p) > 1:
                nx, ny = p[1]
                self.robot_gx, self.robot_gy = nx, ny
                self.path.append((nx, ny))
                _cover(gm, nx, ny)
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        self._stuck += 1
        self.local_extremum_count += 1
        if self._stuck > 300:
            self.computation_time += time.time() - t0
            return None

        self.computation_time += time.time() - t0
        return (self.robot_gx, self.robot_gy)

    def _bfs_uncovered(self):
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        q = deque([(rx, ry)])
        vis = {(rx, ry)}
        while q:
            gx, gy = q.popleft()
            if gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                return (gx, gy)
            for dx, dy in _DIRS4:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                    continue
                vis.add((nx, ny))
                if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED, GridMap.UNKNOWN):
                    q.append((nx, ny))
        return None

    def run(self, max_steps=30000):
        for _ in range(max_steps):
            if self.step() is None:
                break
        return self.path


# =============================================================================
# SP2E — 8 邻域螺旋覆盖
# =============================================================================

class SP2EPlanner:
    """
    SP2E 规划器 (Li et al., 2023)

    核心思想:
    1. 以螺旋方式向外扩展覆盖
    2. 检测前方障碍或已覆盖区域时，主动切换螺旋方向 (CW↔CCW)
    3. 8 邻域移动支持斜向行进，减少转向和重复覆盖
    4. 无可扩展区域时判定覆盖完成
    """

    # CW 和 CCW 方向环 (8 方向)
    _CW = [(0, -1), (1, -1), (1, 0), (1, 1),
           (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    _CCW = [(0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0), (1, -1)]

    def __init__(self, grid_map, global_map, sensor):
        self.grid_map = grid_map        # 机器人的认知地图
        self.global_map = global_map    # 全局真实地图（用于传感器仿真）
        self.sensor = sensor
        self.r = int(sensor.range_max / grid_map.resolution)
        self.robot_gx = 0
        self.robot_gy = 0
        self.path = []
        self.local_extremum_count = 0
        self.computation_time = 0.0
        self._step_count = 0
        self._is_cw = True        # 当前螺旋方向
        self._dir_idx = 0         # 当前方向在环中的索引
        self._last_dir = None     # 上一步移动方向 (dx, dy)
        self._stuck = 0

    def initialize(self, start_gx, start_gy):
        self.robot_gx, self.robot_gy = start_gx, start_gy
        self.grid_map.set_free(start_gx, start_gy)
        self.grid_map.set_covered(start_gx, start_gy)
        self.path = [(start_gx, start_gy)]
        _sense(self.grid_map, self.global_map, start_gx, start_gy, self.sensor)

    def step(self):
        t0 = time.time()
        self._step_count += 1
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy

        _sense(gm, self.global_map, rx, ry, self.sensor)
        _cover(gm, rx, ry)

        if self._step_count % 10 == 0 and _is_done(gm, rx, ry):
            self.computation_time += time.time() - t0
            return None

        dirs = self._CW if self._is_cw else self._CCW

        # --- 1. 优先沿上一步方向继续前进 (惯性) ---
        if self._last_dir:
            dx, dy = self._last_dir
            nx, ny = rx + dx, ry + dy
            if _is_passable(gm, nx, ny) and _is_uncovered(gm, nx, ny):
                self._move_one(nx, ny, (dx, dy))
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 2. 从当前方向开始，沿螺旋方向搜索未覆盖邻居 ---
        for i in range(8):
            idx = (self._dir_idx + i) % 8
            dx, dy = dirs[idx]
            nx, ny = rx + dx, ry + dy
            if _is_passable(gm, nx, ny) and _is_uncovered(gm, nx, ny):
                self._dir_idx = idx
                self._move_one(nx, ny, (dx, dy))
                self._stuck = 0
                self.computation_time += time.time() - t0
                return (self.robot_gx, self.robot_gy)

        # --- 3. 切换螺旋方向再试 ---
        self._stuck += 1
        self.local_extremum_count += 1

        if self._stuck >= 2:
            self._is_cw = not self._is_cw
            dirs2 = self._CW if self._is_cw else self._CCW
            for i in range(8):
                dx, dy = dirs2[i]
                nx, ny = rx + dx, ry + dy
                if _is_passable(gm, nx, ny) and _is_uncovered(gm, nx, ny):
                    self._dir_idx = i
                    self._move_one(nx, ny, (dx, dy))
                    self._stuck = 0
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)

        # --- 4. BFS 寻找最近未覆盖格子 (导航过去) ---
        if self._stuck >= 4:
            target = self._bfs_uncovered()
            if target:
                p = _astar(gm, rx, ry, target[0], target[1])
                if p and len(p) > 1:
                    nx, ny = p[1]
                    dx, dy = nx - rx, ny - ry
                    self._move_one(nx, ny, (dx, dy))
                    self._stuck = 0
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)

        # --- 5. 前沿探索 ---
        if self._stuck >= 8:
            front = _bfs_frontier(gm, rx, ry)
            if front:
                p = _astar(gm, rx, ry, front[0], front[1])
                if p and len(p) > 1:
                    nx, ny = p[1]
                    dx, dy = nx - rx, ny - ry
                    self._move_one(nx, ny, (dx, dy))
                    self._stuck = 0
                    self.computation_time += time.time() - t0
                    return (self.robot_gx, self.robot_gy)

        if self._stuck > 300:
            self.computation_time += time.time() - t0
            return None

        self.computation_time += time.time() - t0
        return (self.robot_gx, self.robot_gy)

    def _move_one(self, nx, ny, direction):
        """移动一步"""
        self.robot_gx, self.robot_gy = nx, ny
        self.path.append((nx, ny))
        self._last_dir = direction
        _cover(self.grid_map, nx, ny)

    def _bfs_uncovered(self):
        gm = self.grid_map
        rx, ry = self.robot_gx, self.robot_gy
        q = deque([(rx, ry)])
        vis = {(rx, ry)}
        while q:
            gx, gy = q.popleft()
            if gm.grid[gy, gx] in (GridMap.FREE, GridMap.UNKNOWN):
                return (gx, gy)
            for dx, dy in _DIRS8:
                nx, ny = gx + dx, gy + dy
                if (nx, ny) in vis or not _is_passable(gm, nx, ny):
                    continue
                vis.add((nx, ny))
                q.append((nx, ny))
        return None

    def run(self, max_steps=30000):
        for _ in range(max_steps):
            if self.step() is None:
                break
        return self.path
