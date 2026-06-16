"""
实验运行主模块 (Experiment Runner)

在所有场景上运行所有算法，收集并对比实验结果。
"""

import os
import time
import json
import numpy as np
from datetime import datetime

from .map_grid import GridMap
from .sensor import SensorModel
from .hcpp_planner import HCPPPlanner
from .baselines import BSAPlanner, FSSTCPlanner, SP2EPlanner, EpsilonStarPlanner
from .scenarios import ScenarioGenerator
from .visualization import (
    plot_map, plot_coverage_progress, plot_comparison_table,
    plot_scenario_comparison, plot_hcpp_cells
)


def create_planner(algo_name, grid_map, global_map, sensor):
    """根据算法名称创建对应的规划器实例"""
    if algo_name == "HCPP":
        return HCPPPlanner(grid_map, global_map, sensor)
    elif algo_name == "BSA":
        return BSAPlanner(grid_map, global_map, sensor)
    elif algo_name == "FS-STC":
        return FSSTCPlanner(grid_map, global_map, sensor)
    elif algo_name == "SP2E":
        return SP2EPlanner(grid_map, global_map, sensor)
    elif algo_name == "Epsilon*" or algo_name == "Epsilon":
        return EpsilonStarPlanner(grid_map, global_map, sensor)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def calculate_coverage_ratio(grid, global_map):
    """
    正确计算覆盖率:

    覆盖率 = 已覆盖格子 / (已覆盖 + 可达FREE格子)
    不包括UNKNOWN (可能被障碍物阻挡无法到达)

    这样100%覆盖率是可达的。
    """
    # 统计已覆盖格子
    covered = np.sum(grid.grid == GridMap.COVERED)

    # 统计可达的FREE格子 (从机器人位置BFS)
    # gm = grid
    # from collections import deque
    # reachable_free = 0
    # queue = deque([(robot_gx, robot_gy)])
    # visited = {(robot_gx, robot_gy)}

    # while queue:
    #     gx, gy = queue.popleft()
    #     if gm.grid[gy, gx] == GridMap.FREE:
    #         reachable_free += 1
    #     for nx, ny in gm.get_neighbors_4(gx, gy):
    #         if (nx, ny) in visited:
    #             continue
    #         if gm.is_obstacle(nx, ny):
    #             continue
    #         visited.add((nx, ny))
    #         if gm.grid[ny, nx] in (GridMap.FREE, GridMap.COVERED):
    #             queue.append((nx, ny))

    # total = covered + reachable_free
    total = np.sum(global_map.grid != GridMap.UNKNOWN)
    if total == 0:
        return 0.0
    return covered / total


def run_single_experiment(algo_name, grid_map, sensor, start_gx, start_gy,
                          max_steps=30000):
    """
    在网格地图副本上运行单个算法的单次实验
    """
    import copy
    global_map = grid_map  # 全局真实地图（保持不变）
    
    # 创建机器人的认知地图（初始化为全 UNKNOWN）
    robot_map = GridMap(grid_map.width * grid_map.resolution, 
                        grid_map.height * grid_map.resolution,
                        grid_map.resolution)
    
    # 创建新的传感器实例
    sensor_copy = SensorModel(sensor.range_max, sensor.angle_resolution, sensor.fov)

    # 创建并初始化规划器
    planner = create_planner(algo_name, robot_map, global_map, sensor_copy)
    planner.initialize(start_gx, start_gy)

    # 运行规划器
    path = planner.run(max_steps)

    # 获取最终机器人位置
    if algo_name == "HCPP":
        final_rx, final_ry = planner.rx, planner.ry
    else:
        final_rx = getattr(planner, 'robot_gx', start_gx)
        final_ry = getattr(planner, 'robot_gy', start_gy)

    # 计算覆盖率 (基于机器人自己的认知地图)
    coverage_ratio = calculate_coverage_ratio(robot_map, global_map)

    path_length = len(path) if path else 0

    local_extremums = getattr(planner, 'local_extremum_count',
                              getattr(planner, 'extremum_count', 0))
    computation_time = getattr(planner, 'computation_time',
                               getattr(planner, 'comp_time', 0.0))

    # 计算转弯次数
    num_turns = 0
    if path and len(path) > 2:
        for i in range(1, len(path) - 1):
            dx1 = path[i][0] - path[i-1][0]
            dy1 = path[i][1] - path[i-1][1]
            dx2 = path[i+1][0] - path[i][0]
            dy2 = path[i+1][1] - path[i][1]
            if (dx1, dy1) != (dx2, dy2):
                num_turns += 1

    return {
        "coverage_ratio": coverage_ratio,
        "path_length": path_length,
        "local_extremums": local_extremums,
        "computation_time": computation_time,
        "num_turns": num_turns,
        "path": path,
        "grid": robot_map,
        "cts": getattr(planner, 'cts', None),
        "r1": getattr(planner, 'r1', None),
    }


def run_experiments(scenarios, algorithms, max_steps=30000, save_dir="results/test"):
    """
    对所有场景和算法运行实验
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}

    for scenario_name, scenario_func in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_name}")
        print(f"{'='*60}")

        grid_map, (start_gx, start_gy) = scenario_func()
        sensor = SensorModel(range_max=3.0, angle_resolution=1.0, fov=360.0)

        scenario_results = {}

        for algo_name in algorithms:
            print(f"  Running {algo_name}...")
            t0 = time.time()

            result = run_single_experiment(
                algo_name, grid_map, sensor, start_gx, start_gy, max_steps
            )
            result["wall_time"] = time.time() - t0

            scenario_results[algo_name] = result

            print(f"    Coverage: {result['coverage_ratio']:.2%}")
            print(f"    Path length: {result['path_length']}")
            print(f"    Local extremums: {result['local_extremums']}")
            print(f"    Computation time: {result['computation_time']:.3f}s")
            print(f"    Wall time: {result['wall_time']:.2f}s")
            print(f"    Turns: {result['num_turns']}")

        all_results[scenario_name] = scenario_results

        # 保存中间结果
        save_results(all_results, os.path.join(save_dir, "results.json"))

    return all_results


def save_results(results, filepath):
    """将结果保存为 JSON 文件（排除网格数据以减少文件大小）"""
    serializable = {}
    for scenario, algo_results in results.items():
        serializable[scenario] = {}
        for algo, metrics in algo_results.items():
            serializable[scenario][algo] = {
                k: v for k, v in metrics.items()
                if k not in ['path', 'grid', 'cts', 'r1']
            }
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


def print_summary_table(results):
    """打印结果汇总表格"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    scenarios = list(results.keys())
    algorithms = list(results[scenarios[0]].keys())

    # 表头
    header = f"{'Scenario':<20}"
    for algo in algorithms:
        header += f"{algo:>12}"
    print(header)
    print("-" * len(header))

    # 覆盖率
    print("\n--- Coverage Ratio ---")
    for scenario in scenarios:
        line = f"{scenario:<20}"
        for algo in algorithms:
            line += f"{results[scenario][algo]['coverage_ratio']:>12.2%}"
        print(line)

    # 路径长度
    print("\n--- Path Length ---")
    for scenario in scenarios:
        line = f"{scenario:<20}"
        for algo in algorithms:
            line += f"{results[scenario][algo]['path_length']:>12d}"
        print(line)

    # 局部极值数
    print("\n--- Local Extremums ---")
    for scenario in scenarios:
        line = f"{scenario:<20}"
        for algo in algorithms:
            line += f"{results[scenario][algo]['local_extremums']:>12d}"
        print(line)

    # 计算时间
    print("\n--- Computation Time (s) ---")
    for scenario in scenarios:
        line = f"{scenario:<20}"
        for algo in algorithms:
            line += f"{results[scenario][algo]['computation_time']:>12.3f}"
        print(line)


def generate_visualizations(results, save_dir="results/test"):
    """生成所有可视化图表"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 场景对比图: 传递每个算法的 grid+path 字典
    scenario_results = []
    for scenario_name, algo_results in results.items():
        # 构建 {algo: {"grid": ..., "path": ...}} 结构
        algo_grids_paths = {
            algo: {"grid": res["grid"], "path": res["path"]}
            for algo, res in algo_results.items()
        }
        scenario_results.append((scenario_name, algo_grids_paths))

    plot_scenario_comparison(scenario_results, save_dir)

    # 2. 覆盖率进展曲线
    for scenario_name, algo_results in results.items():
        paths = {algo: res['path'] for algo, res in algo_results.items()}
        grid_map = algo_results[list(algo_results.keys())[0]]['grid']
        plot_coverage_progress(
            paths, grid_map,
            title=f"Coverage Progress - {scenario_name}",
            save_path=os.path.join(save_dir, f"{scenario_name}_progress.png")
        )

    # 3. 对比汇总表
    plot_comparison_table(
        results,
        save_path=os.path.join(save_dir, "comparison_table.png")
    )

    # 4. HCPP 单元分解图
    for scenario_name, algo_results in results.items():
        if 'HCPP' in algo_results:
            hcpp_res = algo_results['HCPP']
            cts = hcpp_res.get('cts')
            r1 = hcpp_res.get('r1')
            if cts is not None and r1 is not None:
                plot_hcpp_cells(
                    hcpp_res['grid'], cts, hcpp_res['path'], r1,
                    title=f"HCPP Cell Decomposition - {scenario_name}",
                    save_path=os.path.join(
                        save_dir, f"{scenario_name}_HCPP_cells.png")
                )

    print(f"\nVisualizations saved to {save_dir}/")


def main():
    """主入口：运行所有实验"""
    print("="*60)
    print("HCPP Experiment Reproduction")
    print("="*60)
    print(f"Start time: {datetime.now()}")

    algorithms = ["HCPP", "BSA", "FS-STC", "SP2E", "Epsilon*"]

    # 获取所有场景
    scenarios = ScenarioGenerator.get_all_scenarios()
    complex_scenarios = ScenarioGenerator.get_complex_scenarios()

    # 运行标准场景实验 (8个场景)
    print("\n\nRunning standard scenarios (8 scenarios)...")
    results = run_experiments(scenarios, algorithms, max_steps=30000,
                              save_dir="results")

    # 运行复杂场景实验
    print("\n\nRunning complex scenarios...")
    complex_results = run_experiments(
        complex_scenarios, algorithms, max_steps=50000,
        save_dir="results/complex"
    )

    # 合并结果
    all_results = {**results, **complex_results}

    # 打印汇总表
    print_summary_table(all_results)

    # 生成可视化图表
    print("\n\nGenerating visualizations...")
    generate_visualizations(all_results)

    print(f"\nEnd time: {datetime.now()}")
    print("All experiments complete!")

    return all_results


if __name__ == "__main__":
    main()
