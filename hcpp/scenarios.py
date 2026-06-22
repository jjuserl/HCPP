"""
实验场景生成模块 (Experiment Scenarios)

基于论文中的设置，创建多种仿真环境：
- 8 个基础场景：不同障碍物配置
- 岛屿场景 (Island, 100m x 100m)
- 室内场景 (Indoor, 房屋布局)
"""

import numpy as np
from .map_grid import GridMap
# from map_grid import GridMap


class ScenarioGenerator:
    """生成实验评估场景"""

    @staticmethod
    def _find_valid_perimeter_start(grid, seed=42):
        """
        在地图最外围寻找一个有效的非障碍物起点
        优先寻找角落，然后是边的中点
        """
        width, height = grid.width, grid.height
        rng = np.random.RandomState(seed)

        # 定义外围候选点：角落→边中点→随机外围点
        candidates = []

        # 角落点
        corners = [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]
        for gx, gy in corners:
            if grid.is_valid(gx, gy) and not grid.is_obstacle(gx, gy):
                candidates.append((gx, gy))

        # 边中点
        midpoints = [(width//2, 0), (width//2, height-1), 
                    (0, height//2), (width-1, height//2)]
        for gx, gy in midpoints:
            if grid.is_valid(gx, gy) and not grid.is_obstacle(gx, gy):
                candidates.append((gx, gy))

        # 随机外围点
        perimeter = []
        for gx in range(width):
            perimeter.append((gx, 0))
            perimeter.append((gx, height-1))
        for gy in range(1, height-1):
            perimeter.append((0, gy))
            perimeter.append((width-1, gy))

        # 打乱顺序
        rng.shuffle(perimeter)
        for gx, gy in perimeter:
            if grid.is_valid(gx, gy) and not grid.is_obstacle(gx, gy):
                candidates.append((gx, gy))

        # 去重（保持顺序以维护优先级：角落→边中点→随机外围点）
        candidates = list(dict.fromkeys(candidates))
        if candidates:
            return candidates[0]

        # 极端情况：所有外围都是障碍物
        # 寻找任意非障碍物点
        for gx in range(width):
            for gy in range(height):
                if grid.is_valid(gx, gy) and not grid.is_obstacle(gx, gy):
                    return (gx, gy)

        return (width//2, height//2)  # 最后退路

    @staticmethod
    def scenario1_random(width=30, height=30, num_obstacles=3):
        """
        场景 1：随机障碍物
        随机放置矩形障碍物（格子组合），
        边界区域保留为自由空间
        """
        grid = GridMap(width, height, resolution=1.0)
        # 边界区域保留为自由空间（地图的 10%）
        grid.grid[0:int(0.1*height), :] = GridMap.FREE
        grid.grid[int(0.9*height):, :] = GridMap.FREE
        grid.grid[:, 0:int(0.1*width)] = GridMap.FREE
        grid.grid[:, int(0.9*width):] = GridMap.FREE

        # 放置随机方块障碍物（以 (ox,oy) 为中心，对称扩展）
        rng = np.random.RandomState(42)
        for _ in range(num_obstacles):
            ox = rng.randint(int(0.15*width), int(0.85*width))
            oy = rng.randint(int(0.15*height), int(0.85*height))
            size = rng.randint(1, 3)   # 方块半边长 (2*size+1) 格
            for dx in range(-size, size):
                for dy in range(-size, size):
                    if grid.is_valid(ox + dx, oy + dy):
                        grid.set_obstacle(ox + dx, oy + dy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
        return grid, start_pos

    @staticmethod
    def scenario2_sparse(width=10, height=10, num_obstacles=1):
        """
        场景 2：稀疏障碍物
        随机放置少量方块障碍物
        """
        grid = GridMap(width, height, resolution=1.0)
        rng = np.random.RandomState(123)
        for _ in range(num_obstacles):
            ox = rng.randint(int(0.1*width), int(0.9*width))
            oy = rng.randint(int(0.1*height), int(0.9*height))
            size = rng.randint(2, 3)   # 方块半边长 (2*size+1) 格
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if grid.is_valid(ox + dx, oy + dy):
                        grid.set_obstacle(ox + dx, oy + dy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=123)
        return grid, start_pos

    @staticmethod
    def scenario3_dense(width=30, height=30, num_obstacles=5):
        """
        场景 3：密集障碍物
        随机放置大量小型障碍物
        """
        grid = GridMap(width, height, resolution=1.0)
        rng = np.random.RandomState(456)
        for _ in range(num_obstacles):
            ox = rng.randint(int(0.1*width), int(0.9*width))
            oy = rng.randint(int(0.1*height), int(0.9*height))
            o_size = rng.randint(1, 3)
            for dx in range(-o_size, o_size + 1):
                for dy in range(-o_size, o_size + 1):
                    if grid.is_valid(ox + dx, oy + dy):
                        grid.set_obstacle(ox + dx, oy + dy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=456)
        return grid, start_pos

    @staticmethod
    def scenario4_columns(width=30, height=30):
        """
        场景 4：柱状障碍物
        在网格状排列的方形柱体（3x3格子）
        """
        grid = GridMap(width, height, resolution=1.0)
        col_spacing_x = 10  # 列间距
        col_spacing_y = 10  # 行间距
        size = 1  # 柱子半边长 (3x3 方块)

        for gx in range(col_spacing_x, width - col_spacing_x, col_spacing_x):
            for gy in range(col_spacing_y, height - col_spacing_y, col_spacing_y):
                for dx in range(-size, size + 1):
                    for dy in range(-size, size + 1):
                        if grid.is_valid(gx + dx, gy + dy):
                            grid.set_obstacle(gx + dx, gy + dy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=789)
        return grid, start_pos

    @staticmethod
    def scenario5_walls(width=30, height=30):
        """
        场景 5：墙壁障碍物
        水平和垂直墙壁形成走廊结构，带有通行缺口
        """
        grid = GridMap(width, height, resolution=1.0)
        wall_thickness = 1

        # 水平墙壁（带缺口）
        for wy in [10, 20]:
            for gx in range(3, width - 3):
                gap_positions = [10, 20]
                # 只有当 gx 不在任何缺口范围内时才设置障碍物
                in_gap = any(abs(gx - gap) <= 2 for gap in gap_positions)
                if not in_gap:
                    for dy in range(wall_thickness):
                        if grid.is_valid(gx, wy + dy):
                            grid.set_obstacle(gx, wy + dy)

        # 垂直墙壁（带缺口）
        for wx in [10, 20]:
            for gy in range(3, height - 3):
                gap_positions = [10, 20]
                in_gap = any(abs(gy - gap) <= 2 for gap in gap_positions)
                if not in_gap:
                    for dx in range(wall_thickness):
                        if grid.is_valid(wx + dx, gy):
                            grid.set_obstacle(wx + dx, gy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=987)
        return grid, start_pos

    @staticmethod
    def scenario6_maze(width=30, height=30):
        """
        场景 6：迷宫结构
        简单的迷宫式布局
        """
        grid = GridMap(width, height, resolution=1.0)
        wall_thickness = 1

        walls_h = [(0, 8), (12, 20), (24, width)]
        walls_v = [(0, 8), (12, 20), (24, height)]

        for y_start, y_end in walls_h:
            for gx in range(y_start, y_end):
                for dy in range(wall_thickness):
                    for y_pos in [10, 20]:
                        if grid.is_valid(gx, y_pos + dy):
                            grid.set_obstacle(gx, y_pos + dy)

        for y_start, y_end in walls_v:
            for gy in range(y_start, y_end):
                for dx in range(wall_thickness):
                    for x_pos in [10, 20]:
                        if grid.is_valid(x_pos + dx, gy):
                            grid.set_obstacle(x_pos + dx, gy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=654)
        return grid, start_pos

    @staticmethod
    def scenario7_corner(width=30, height=30):
        """
        场景 7：角落障碍物
        障碍物集中在四个角落和边缘区域
        """
        grid = GridMap(width, height, resolution=1.0)
        size = 2  # 角落障碍物半边长 (5x5 方块)

        # 四个角落的障碍物
        corners = [
            (size + 1, size + 1),
            (width - size - 2, size + 1),
            (size + 1, height - size - 2),
            (width - size - 2, height - size - 2),
        ]
        for cx, cy in corners:
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if grid.is_valid(cx + dx, cy + dy):
                        grid.set_obstacle(cx + dx, cy + dy)

        # 边缘障碍物（1x1 方块间隔放置）
        for gx in range(0, width, 6):
            if grid.is_valid(gx, 0):
                grid.set_obstacle(gx, 0)
            if grid.is_valid(gx, height - 1):
                grid.set_obstacle(gx, height - 1)

        for gy in range(0, height, 6):
            if grid.is_valid(0, gy):
                grid.set_obstacle(0, gy)
            if grid.is_valid(width - 1, gy):
                grid.set_obstacle(width - 1, gy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=321)
        return grid, start_pos

    @staticmethod
    def scenario8_rooms(width=30, height=30):
        """
        场景 8：房间结构
        多个房间，通过门道连接
        """
        grid = GridMap(width, height, resolution=1.0)
        wall_thickness = 1

        # 外墙
        for gx in range(width):
            for dy in range(wall_thickness):
                grid.set_obstacle(gx, dy)
                grid.set_obstacle(gx, height - 1 - dy)
        for gy in range(height):
            for dx in range(wall_thickness):
                grid.set_obstacle(dx, gy)
                grid.set_obstacle(width - 1 - dx, gy)

        # 内墙，形成房间（带门洞）
        room_walls = [
            (width // 3, 2, width // 3, height // 2 - 3),  # 竖墙左上
            (width // 3, height // 2 + 3, width // 3, height - 2),  # 竖墙左下
            (2 * width // 3, 2, 2 * width // 3, height // 2 - 3),  # 竖墙右上
            (2 * width // 3, height // 2 + 3, 2 * width // 3, height - 2),  # 竖墙右下
            (wall_thickness + 1, height // 3, width // 3 - 3, height // 3),  # 横墙左上
            (width // 3 + 3, height // 3, 2 * width // 3 - 3, height // 3),  # 横墙中上
            (2 * width // 3 + 3, height // 3, width - wall_thickness - 1, height // 3),  # 横墙右上
            (wall_thickness + 1, 2 * height // 3, width // 2 - 3, 2 * height // 3),  # 横墙左下
            (width // 2 + 3, 2 * height // 3, width - wall_thickness - 1, 2 * height // 3),  # 横墙右下
        ]

        for x0, y0, x1, y1 in room_walls:
            for gx in range(min(x0, x1), max(x0, x1)):
                for gy in range(min(y0, y1), max(y0, y1)):
                    if grid.is_valid(gx, gy):
                        grid.set_obstacle(gx, gy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=135)
        return grid, start_pos

    @staticmethod
    def scenario_island(width=30, height=30):
        """
        岛屿场景：中心有一个大型不规则障碍物（格子组合），周围是自由空间
        30m x 30m 地图
        """
        grid = GridMap(width, height, resolution=1.0)
        island_cx = width // 2
        island_cy = height // 2
        size = 5  # 岛屿半边长：先铺 11x11 方块，再削角

        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                gx = island_cx + dx
                gy = island_cy + dy
                if not grid.is_valid(gx, gy):
                    continue
                # 削去四角，制造不规则形状
                if abs(dx) == size and abs(dy) == size:
                    continue
                if abs(dx) >= size - 1 and abs(dy) >= size - 1:
                    continue
                grid.set_obstacle(gx, gy)

        # 添加三个突出的格子（1x1 方块），制造不规则外观
        bumps = [
            (island_cx - size - 1, island_cy),
            (island_cx + size + 1, island_cy),
            (island_cx, island_cy - size - 1),
        ]
        for gx, gy in bumps:
            if grid.is_valid(gx, gy):
                grid.set_obstacle(gx, gy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=246)
        return grid, start_pos

    @staticmethod
    def scenario_indoor(width=30, height=30):
        """
        室内场景：房屋/公寓布局，包含多个房间和走廊
        30m x 30m 地图
        """
        grid = GridMap(width, height, resolution=1.0)
        wall_thickness = 1

        # 外墙
        for gx in range(width):
            for dy in range(wall_thickness):
                grid.set_obstacle(gx, dy)
                grid.set_obstacle(gx, height - 1 - dy)
        for gy in range(height):
            for dx in range(wall_thickness):
                grid.set_obstacle(dx, gy)
                grid.set_obstacle(width - 1 - dx, gy)

        # 房间布局
        self_add_wall = ScenarioGenerator._add_wall
        self_add_wall(grid, 10, wall_thickness, 10, 15)  # 竖墙分隔
        self_add_wall(grid, wall_thickness + 1, 10, 8, 10)  # 横墙分隔
        self_add_wall(grid, 12, 10, 18, 10)  # 横墙续

        # 厨房（右上）
        self_add_wall(grid, 18, wall_thickness, 18, 15)

        # 卧室 1（左下）
        self_add_wall(grid, 8, 18, 15, 18)
        self_add_wall(grid, 8, 18, 8, height - wall_thickness - 1)

        # 卧室 2（右下）
        self_add_wall(grid, 18, 15, 18, height - wall_thickness - 1)

        # 门洞（墙壁缺口）
        for gx, gy in [(10, 8), (10, 15), (18, 15), (8, 12), (15, 12)]:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if grid.is_valid(gx + dx, gy + dy):
                        grid.set_free(gx + dx, gy + dy)

        start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=864)
        return grid, start_pos

    @staticmethod
    def _add_wall(grid, x0, y0, x1, y1):
        """辅助函数：添加一段墙壁"""
        for gx in range(min(x0, x1), max(x0, x1)):
            for gy in range(min(y0, y1), max(y0, y1)):
                if grid.is_valid(gx, gy):
                    grid.set_obstacle(gx, gy)

    @staticmethod
    def get_all_scenarios():
        """Get all standard (8) scenarios with start positions."""
        scenarios = [
            ("Scenario1_Random", ScenarioGenerator.scenario1_random),
            ("Scenario2_Sparse", ScenarioGenerator.scenario2_sparse),
            ("Scenario3_Dense", ScenarioGenerator.scenario3_dense),
            ("Scenario4_Columns", ScenarioGenerator.scenario4_columns),
            ("Scenario5_Walls", ScenarioGenerator.scenario5_walls),
            ("Scenario6_Maze", ScenarioGenerator.scenario6_maze),
            ("Scenario7_Corner", ScenarioGenerator.scenario7_corner),
            ("Scenario8_Rooms", ScenarioGenerator.scenario8_rooms),
        ]
        return scenarios

    @staticmethod
    def get_complex_scenarios():
        """Get complex scenarios (island, indoor)."""
        return [
            ("Scenario_Island", ScenarioGenerator.scenario_island),
            ("Scenario_Indoor", ScenarioGenerator.scenario_indoor),
        ]


# ============================================================
# 自定义场景模板
# ============================================================

def custom_scenario1(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左下角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=3, y=18, w=3, h=2)      # 左侧矩形
    add_rect(grid, x=3, y=12, w=1, h=6)    # 右上方矩形
    add_rect(grid, x=3, y=10, w=3, h=2)     # 右下角方块

    add_rect(grid, x=25, y=18, w=5, h=2)      # 左侧矩形
    add_rect(grid, x=25, y=10, w=5, h=2)    # 右上方矩形


    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos


def custom_scenario2(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=3, y=11, w=3, h=8)
    add_rect(grid, x=18, y=0, w=1, h=5)
    add_rect(grid, x=18, y=13, w=1, h=6)
    add_rect(grid, x=18, y=14, w=12, h=1)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos

def custom_scenario3(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=14, y=7, w=1, h=14)
    add_rect(grid, x=10, y=21, w=5, h=1)
    add_rect(grid, x=9, y=6, w=6, h=1)
    add_rect(grid, x=15, y=14, w=3, h=1)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos
def custom_scenario4(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=5, y=24, w=4, h=2)
    add_rect(grid, x=4, y=8, w=4, h=2)
    add_rect(grid, x=24, y=20, w=2, h=4)
    add_rect(grid, x=25, y=6, w=5, h=2)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos
def custom_scenario5(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=10, y=10, w=8, h=8)
    # add_rect(grid, x=12, y=18, w=2, h=8)
    # add_rect(grid, x=20, y=5, w=4, h=4)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos
def custom_scenario6(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=0, y=9, w=12, h=2)
    add_rect(grid, x=0, y=22, w=12, h=2)
    add_rect(grid, x=6, y=0, w=8, h=3)
    add_rect(grid, x=14, y=0, w=8, h=1)
    add_rect(grid, x=10, y=15, w=20, h=2)
    add_rect(grid, x=28, y=11, w=2, h=4)
    add_rect(grid, x=22, y=9, w=8, h=2)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos
def custom_scenario7(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=0, y=18, w=2, h=1)
    add_rect(grid, x=10, y=18, w=1, h=12)
    add_rect(grid, x=6, y=18, w=4, h=1)
    add_obstacle(grid, 18, 29)
    add_rect(grid, x=18, y=16, w=1, h=6)
    add_rect(grid, x=19, y=18, w=11, h=1)
    add_rect(grid, x=18, y=0, w=1, h=10)
    add_rect(grid, x=5, y=6, w=2, h=3)
    add_rect(grid, x=7, y=7, w=3, h=1)
    add_rect(grid, x=10, y=6, w=2, h=3)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos
def custom_scenario8(width=30, height=30):
    """
    自定义场景模板: 在此函数中自由放置障碍物。

    使用方式:
      1. 修改下面的障碍物列表
      2. 终端运行: python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"

    障碍物添加方式:
      - add_rect(grid, x, y, w, h)         矩形障碍物 (x,y)=左上角, w=宽度, h=高度
      - add_obstacle(grid, x, y)           单个障碍物格子
      - add_line(grid, x1, y1, x2, y2)     线段障碍物
    """
    grid = GridMap(width, height, resolution=1.0)

    # ========== 在此区域自由添加障碍物 ==========

    # 示例1: 矩形障碍物
    add_rect(grid, x=5, y=5, w=3, h=5)
    add_rect(grid, x=12, y=18, w=2, h=8)
    add_rect(grid, x=20, y=5, w=4, h=4)

    # 示例2: 单个障碍物格子
    # add_obstacle(grid, 10, 10)
    # add_obstacle(grid, 11, 10)

    # 示例3: 线段障碍物
    # add_line(grid, 2, 15, 28, 15)          # 水平线
    # add_line(grid, 15, 2, 15, 28)          # 竖直线

    # ========== 障碍物定义结束 ==========

    start_pos = ScenarioGenerator._find_valid_perimeter_start(grid, seed=42)
    return grid, start_pos

# ---- 辅助函数: 帮助你在自定义场景中快速添加障碍物 ----

def add_rect(grid, x, y, w, h):
    """添加矩形障碍物: (x,y)=左上角, w=宽度, h=高度"""
    for dx in range(w):
        for dy in range(h):
            gx, gy = x + dx, y + dy
            if grid.is_valid(gx, gy):
                grid.set_obstacle(gx, gy)


def add_obstacle(grid, x, y):
    """添加单个障碍物格子"""
    if grid.is_valid(x, y):
        grid.set_obstacle(x, y)


def add_line(grid, x1, y1, x2, y2):
    """添加线段障碍物 (Bresenham 直线)"""
    cells = grid.bresenham_line(x1, y1, x2, y2)
    for gx, gy in cells:
        if grid.is_valid(gx, gy):
            grid.set_obstacle(gx, gy)


# ---- 可视化 ----

def show_scenario(grid, start_pos=None):
    """
    可视化自定义场景: 只显示障碍物(黑)和非障碍物(白)，带坐标轴数字。

    用法:
      from hcpp.scenarios import custom_scenario, show_scenario
      grid, start = custom_scenario()
      show_scenario(grid, start)

    或直接终端运行:
      python -c "from hcpp.scenarios import custom_scenario, show_scenario; show_scenario(*custom_scenario())"
    """
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 二值化: 障碍物=1(黑), 其他=0(白)
    data = (grid.grid == GridMap.OBSTACLE).astype(int)
    w, h = grid.width, grid.height

    fig, ax = plt.subplots(figsize=(8, 8))

    # 黑白配色
    cmap = plt.cm.colors.ListedColormap(['#ffffff', '#000000'])

    ax.imshow(data, cmap=cmap, origin='lower',
              extent=[0, w, 0, h], aspect='equal')

    # 坐标轴数字 (每1格显示一个数字)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xticks(np.arange(0, w + 1, 1))
    ax.set_yticks(np.arange(0, h + 1, 1))

    # 网格线 (每1格一条线)
    ax.grid(True, color='#cccccc', linewidth=0.3, alpha=0.5)

    # 起点
    if start_pos is not None:
        ax.plot(start_pos[0] + 0.5, start_pos[1] + 0.5, 'ro', markersize=8,
                markeredgecolor='darkred', markeredgewidth=2, zorder=5)

    # 图例
    legend_patches = [
        mpatches.Patch(color='#ffffff', label='FREE'),
        mpatches.Patch(color='#000000', label='OBSTACLE'),
    ]
    if start_pos is not None:
        legend_patches.append(mpatches.Patch(color='red', label='Start'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax.set_title(f'Custom Scenario ({w}x{h})', fontsize=14, fontweight='bold')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    plt.tight_layout()
    plt.show()


# ---- 如果直接运行此文件, 预览自定义场景 ----
if __name__ == "__main__":
    g, s = custom_scenario7()
    show_scenario(g, s)
