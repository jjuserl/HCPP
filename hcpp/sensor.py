"""
模拟传感器模块 (Sensor Model)

模拟360度LiDAR测距传感器，每度发射一条射线。
使用Bresenham直线算法检测障碍物遮挡。
"""

import numpy as np
from .map_grid import GridMap


class SensorModel:
    """模拟测距传感器 (360° LiDAR with ray casting)"""

    def __init__(self, range_max=8.0, angle_resolution=1.0, fov=360.0):
        """
        初始化传感器模型

        参数:
            range_max: 传感器最大量程 (米)
            angle_resolution: 角度分辨率 (度)，默认1度=360条射线
            fov: 视场角 (度)，默认360度
        """
        self.range_max = range_max
        self.angle_resolution = angle_resolution
        self.fov = fov
        self.num_beams = int(fov / angle_resolution)

    def sense(self, grid_map, robot_x, robot_y, robot_theta=0.0):
        """
        执行传感器扫描 (带遮挡检测)

        使用Bresenham直线算法模拟每条激光束:
        - 光束路径上的格子标记为FREE
        - 遇到障碍物时停止当前射线
        - 障碍物后的格子保持UNKNOWN (被遮挡)

        参数:
            grid_map: GridMap对象
            robot_x, robot_y: 机器人世界坐标 (米)
            robot_theta: 机器人朝向 (弧度)

        返回:
            GridMap: 更新后的网格地图
        """
        r_gx, r_gy = grid_map.world_to_grid(robot_x, robot_y)
        max_range_cells = int(self.range_max / grid_map.resolution)

        for i in range(self.num_beams):
            angle = robot_theta + np.radians(i * self.angle_resolution)
            end_x = r_gx + max_range_cells * np.cos(angle)
            end_y = r_gy + max_range_cells * np.sin(angle)
            end_x = int(round(end_x))
            end_y = int(round(end_y))

            line = grid_map.bresenham_line(r_gx, r_gy, end_x, end_y)

            for gx, gy in line[1:]:  # 跳过机器人自身
                if not grid_map.is_valid(gx, gy):
                    break
                if grid_map.grid[gy, gx] == GridMap.OBSTACLE:
                    # 遇到障碍物: 标记障碍物，停止当前射线
                    break
                if grid_map.grid[gy, gx] == GridMap.UNKNOWN:
                    grid_map.set_free(gx, gy)

        return grid_map

    def simple_sense(self, robot_map, global_map, robot_gx, robot_gy, debug=False):
        """
        统一5x5感知模型 (带 Bresenham 逐格遮挡检测):
        - 感知范围: 5x5 网格 (机器人在中心)
        - 对所有格子，从机器人画 Bresenham 直线
        - 遇障碍物即停止该射线，障碍物后方格子保持 UNKNOWN
        - 所有方向探测深度统一，按距离排序处理

        坐标系: (0,0) 在左下角，x向右，y向上
        """
        if debug:
            print(f"  [Sensor] 机器人位置: ({robot_gx},{robot_gy}), 开始5x5 Bresenham感知")

        # 收集5x5范围内所有待检测格子（排除机器人自身），按距离排序
        cells_to_check = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                gx, gy = robot_gx + dx, robot_gy + dy
                if global_map.is_valid(gx, gy):
                    dist = dx * dx + dy * dy  # 平方距离
                    cells_to_check.append((dist, gx, gy))

        cells_to_check.sort(key=lambda x: x[0])

        blocked = set()  # 已被障碍物遮挡的格子

        for dist, gx, gy in cells_to_check:
            if (gx, gy) in blocked:
                if debug:
                    print(f"  [Sensor]   ({gx},{gy}) dist={dist:.0f} - 被遮挡, 保持UNKNOWN")
                continue

            # 跳过 robot_map 中已检测过的格子
            if robot_map.grid[gy, gx] != GridMap.UNKNOWN:
                continue

            # Bresenham 直线从机器人到目标格子
            line = global_map.bresenham_line(robot_gx, robot_gy, gx, gy)

            for lgx, lgy in line[1:]:  # 跳过机器人自身
                if not global_map.is_valid(lgx, lgy):
                    break

                if (lgx, lgy) == (gx, gy):
                    # 到达目标格子
                    gs = global_map.grid[lgy, lgx]
                    if gs == GridMap.OBSTACLE:
                        robot_map.set_cell(lgx, lgy, GridMap.OBSTACLE)
                        if debug:
                            print(f"  [Sensor]   ({lgx},{lgy}) dist={dist:.0f} - OBSTACLE")
                    else:
                        robot_map.set_free(lgx, lgy)
                        if debug:
                            print(f"  [Sensor]   ({lgx},{lgy}) dist={dist:.0f} - FREE")
                    break

                # 路径中间格子
                rm_state = robot_map.grid[lgy, lgx]
                if rm_state == GridMap.OBSTACLE:
                    blocked.add((gx, gy))
                    if debug:
                        print(f"  [Sensor]   ({gx},{gy}) - 被({lgx},{lgy})处障碍物遮挡")
                    break
                elif rm_state == GridMap.UNKNOWN:
                    gs = global_map.grid[lgy, lgx]
                    if gs == GridMap.OBSTACLE:
                        robot_map.set_cell(lgx, lgy, GridMap.OBSTACLE)
                        blocked.add((gx, gy))
                        if debug:
                            print(f"  [Sensor]   ({lgx},{lgy}) dist=* - OBSTACLE(途中), 遮挡({gx},{gy})")
                        break
                    else:
                        robot_map.set_free(lgx, lgy)
        
