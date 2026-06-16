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

    def simple_sense(self, robot_map, global_map, robot_gx, robot_gy):
        """
        带遮挡检测的感知模型:
        - 感知范围: 5x5 网格 (机器人在中心)
        - 8邻域: 直接复制全局地图
        - 外围: 如果对应方向的8邻域无障碍，则直接复制
        
        坐标系: (0,0) 在左下角，x向右，y向上
        """
        # 8邻域格子（直接列举）
        neighbors_8 = [
            (-1, -1), (0, -1), (1, -1),  # 下边
            (-1,  0),          (1,  0),  # 左右
            (-1,  1), (0,  1), (1,  1),  # 上边
        ]
        
        direction_extensions = {
            0: [(-1, -1), (-2, -2), (-2, -1), (-1, -2)],  # 左下角，延伸3个
            1: [(0, -1), (0, -2)],                          # 正下方，延伸1个
            2: [(1, -1), (2, -2), (2, -1), (1, -2)],       # 右下角，延伸3个
            3: [(-1, 0), (-2, 0)],                          # 正左方，延伸1个
            4: [(1, 0), (2, 0)],                            # 正右方，延伸1个
            5: [(-1, 1), (-2, 2), (-2, 1), (-1, 2)],       # 左上角，延伸3个
            6: [(0, 1), (0, 2)],                            # 正上方，延伸1个
            7: [(1, 1), (2, 2), (2, 1), (1, 2)],           # 右上角，延伸3个
        }

        # 第一步：8邻域直接复制全局地图
        for id, (dx, dy) in enumerate(neighbors_8):
            target_gx = robot_gx + dx
            target_gy = robot_gy + dy
            
            if not global_map.is_valid(target_gx, target_gy):
                continue
            
            if robot_map.grid[target_gy, target_gx] == GridMap.UNKNOWN:
                global_state = global_map.grid[target_gy, target_gx]
                if global_state == GridMap.OBSTACLE:
                    robot_map.set_cell(target_gx, target_gy, GridMap.OBSTACLE)
                elif global_state == GridMap.FREE:
                    robot_map.set_free(target_gx, target_gy)
                    
                    # 8邻域free才进行其后面格子赋值，否则视为看不到后面格子

                    for ex_dx, ex_dy in direction_extensions[id][1:]:
                        check_gx = robot_gx + ex_dx
                        check_gy = robot_gy + ex_dy
                        if not global_map.is_valid(check_gx, check_gy):
                            continue
                        global_state = global_map.grid[check_gy, check_gx]
                        if global_state == GridMap.OBSTACLE:
                            robot_map.set_cell(check_gx, check_gy, GridMap.OBSTACLE)
                        elif global_state == GridMap.FREE:
                            robot_map.set_free(check_gx, check_gy)
        
