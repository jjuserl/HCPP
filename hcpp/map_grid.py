"""
网格地图表示模块 (Grid Map Representation)

将环境表示为二维占用栅格地图 (2D Occupancy Grid Map)。
每个栅格的状态可以是：
  - UNKNOWN (0): 未知区域
  - FREE (1): 自由空间（已发现但未覆盖）
  - OBSTACLE (2): 障碍物
  - COVERED (3): 已覆盖区域
  - FRONTIER (4): 前沿边界（用于 Boustrophedon 分解）
"""

import numpy as np
from collections import deque #  双端队列


class GridMap:
    """二维占用栅格地图，用于覆盖路径规划"""

    # 栅格状态常量
    UNKNOWN = 0   # 未知
    FREE = 1      # 自由空间
    OBSTACLE = 2  # 障碍物
    COVERED = 3   # 已覆盖
    FRONTIER = 4  # 前沿边界（已发现但不在当前覆盖单元内）

    def __init__(self, width, height, resolution=1.0):
        """
        初始化网格地图

        参数:
            width: 地图宽度（米）
            height: 地图高度（米）
            resolution: 网格分辨率（米/格）
        """
        self.resolution = resolution
        self.width = int(width / resolution)
        self.height = int(height / resolution)
        # 初始化为全未知状态
        self.grid = np.full((self.height, self.width), self.UNKNOWN, dtype=np.int8)
        self.origin_x = 0.0  # 地图原点 X 坐标
        self.origin_y = 0.0  # 地图原点 Y 坐标
        # 格子状态变化回调: callback(gx, gy, new_state)
        self.on_cell_changed = None

    def world_to_grid(self, wx, wy):
        """世界坐标转网格坐标"""
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """网格坐标转世界坐标（返回格子中心点）"""
        wx = (gx + 0.5) * self.resolution + self.origin_x
        wy = (gy + 0.5) * self.resolution + self.origin_y
        return wx, wy

    def is_valid(self, gx, gy):
        """检查网格坐标是否在地图范围内"""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def is_free(self, gx, gy):
        """检查格子是否为自由空间"""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] == self.FREE

    def is_unknown(self, gx, gy):
        """检查格子是否为未知状态"""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] == self.UNKNOWN

    def is_covered(self, gx, gy):
        """检查格子是否已被覆盖"""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] == self.COVERED

    def is_obstacle(self, gx, gy):
        """检查格子是否为障碍物"""
        if not self.is_valid(gx, gy):
            return True
        return self.grid[gy, gx] == self.OBSTACLE

    def is_uncovered_frontier(self, gx, gy):
        """检查格子是否为未覆盖前沿边界"""
        if not self.is_valid(gx, gy):
            return False
        if (self.is_obstacle(gx, gy) or self.is_covered(gx, gy)):
            return True

    def is_explored(self, gx, gy):
        """检查格子是否已被探索（非 UNKNOWN）"""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] != self.UNKNOWN

    def is_traversable(self, gx, gy):
        """检查格子是否可通行（FREE, COVERED, FRONTIER）"""
        if not self.is_valid(gx, gy):
            return False
        return self.grid[gy, gx] in (self.FREE, self.COVERED, self.FRONTIER)

    def set_cell(self, gx, gy, value):
        """设置格子状态值，并触发变化回调"""
        if self.is_valid(gx, gy):
            old = self.grid[gy, gx]
            self.grid[gy, gx] = value
            if old != value and self.on_cell_changed is not None:
                self.on_cell_changed(gx, gy, value)

    def set_free(self, gx, gy):
        """标记格子为自由空间"""
        self.set_cell(gx, gy, self.FREE)

    def set_obstacle(self, gx, gy):
        """标记格子为障碍物"""
        self.set_cell(gx, gy, self.OBSTACLE)

    def set_covered(self, gx, gy):
        """标记格子为已覆盖"""
        self.set_cell(gx, gy, self.COVERED)

    def set_frontier(self, gx, gy):
        """标记格子为前沿边界"""
        self.set_cell(gx, gy, self.FRONTIER)

    def get_neighbors_4(self, gx, gy):
        """获取 4 邻域（上下左右）邻居坐标"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = gx + dx, gy + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_neighbors_8(self, gx, gy):
        """获取 8 邻域（包含对角线）邻居坐标"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if self.is_valid(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def get_free_neighbors(self, gx, gy):
        """获取非障碍物的 4 邻域邻居"""
        return [(nx, ny) for nx, ny in self.get_neighbors_4(gx, gy)
                if self.is_free(nx, ny)]

    def get_traversable_neighbors(self, gx, gy):
        """获取可通行的 4 邻域邻居"""
        return [(nx, ny) for nx, ny in self.get_neighbors_4(gx, gy)
                if self.is_traversable(nx, ny)]

    def get_unknown_frontiers(self):
        """获取所有与自由格子相邻的未知格子（前沿）"""
        frontiers = set()
        for gy in range(self.height):
            for gx in range(self.width):
                if self.grid[gy, gx] == self.FREE:
                    for nx, ny in self.get_neighbors_4(gx, gy):
                        if self.grid[ny, nx] == self.UNKNOWN:
                            frontiers.add((nx, ny))
        return list(frontiers)

    def get_uncovered_cells(self):
        """获取所有尚未覆盖的格子（自由或未知）"""
        mask = (self.grid == self.FREE) | (self.grid == self.UNKNOWN)
        return list(zip(*np.where(mask)[::-1]))

    def get_coverage_ratio(self):
        """计算当前覆盖率"""
        free_count = np.sum(self.grid == self.FREE) + np.sum(self.grid == self.COVERED)
        if free_count == 0:
            return 0.0
        covered_count = np.sum(self.grid == self.COVERED)
        return covered_count / free_count

    def get_explored_ratio(self):
        """计算探索率（已探索区域/总自由区域）"""
        total_free = np.sum(self.grid != self.OBSTACLE)
        if total_free == 0:
            return 0.0
        explored = np.sum((self.grid == self.FREE) | (self.grid == self.COVERED) | (self.grid == self.FRONTIER))
        return explored / total_free

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham 直线算法，用于射线投射 (ray casting)"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def flood_fill(self, start_gx, start_gy, target_state, connected_states=None):
        """
        洪泛填充：从起点开始，填充所有与 connected_states 相连的格子

        参数:
            start_gx, start_gy: 起始坐标
            target_state: 目标状态（如 UNKNOWN）
            connected_states: 可连通的格子状态集合，默认 [FREE, COVERED, FRONTIER]

        返回:
            set: 所有填充的格子坐标集合
        """
        if connected_states is None:
            connected_states = {self.FREE, self.COVERED, self.FRONTIER}
        if not self.is_valid(start_gx, start_gy):
            return set()
        if self.grid[start_gy, start_gx] != target_state:
            return set()

        fill_set = set()
        queue = deque([(start_gx, start_gy)])
        visited = set()

        while queue:
            gx, gy = queue.popleft()
            if (gx, gy) in visited:
                continue
            visited.add((gx, gy))
            if self.grid[gy, gx] == target_state:
                fill_set.add((gx, gy))
            for nx, ny in self.get_neighbors_4(gx, gy):
                if (nx, ny) not in visited:
                    if self.grid[ny, nx] in connected_states or self.grid[ny, nx] == target_state:
                        queue.append((nx, ny))
        return fill_set

    def get_connected_component(self, start_gx, start_gy, states=None):
        """
        获取从起点可达的连通区域

        参数:
            start_gx, start_gy: 起始坐标
            states: 可通行的状态集合，默认 {FREE, COVERED, FRONTIER}

        返回:
            set: 连通区域坐标集合
        """
        if states is None:
            states = {self.FREE, self.COVERED, self.FRONTIER}
        component = set()
        queue = deque([(start_gx, start_gy)])
        visited = {(start_gx, start_gy)}
        while queue:
            gx, gy = queue.popleft()
            component.add((gx, gy))
            for nx, ny in self.get_neighbors_4(gx, gy):
                if (nx, ny) not in visited and self.grid[ny, nx] in states:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return component

    def load_from_array(self, arr):
        """从 NumPy 数组加载地图数据"""
        self.height, self.width = arr.shape
        self.grid = np.where(arr == 1, self.OBSTACLE, self.UNKNOWN).astype(np.int8)