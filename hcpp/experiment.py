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
from .scenarios import ScenarioGenerator, custom_scenario1, custom_scenario2, custom_scenario3, custom_scenario4, custom_scenario5, custom_scenario6, custom_scenario7
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


def calculate_coverage_ratio(grid, global_free_count):
    """
    计算覆盖率:

    覆盖率 = robot_map 中已覆盖格子 / global_map 中 FREE 格子总数

    参数:
        grid: 机器人的认知地图 (robot_map)
        global_free_count: 全局真实地图中 FREE 格子的总数 (预计算, 不变)
    """
    covered = np.sum(grid.grid == GridMap.COVERED)
    if global_free_count is None or global_free_count == 0:
        return 0.0
    return covered / global_free_count


def run_single_experiment(algo_name, grid_map, global_map, sensor, start_gx, start_gy,
                          max_steps=30000, global_free_count=None):
    """
    在网格地图副本上运行单个算法的单次实验

    参数:
        grid_map: 原始场景地图 (用于创建 robot_map 的尺寸)
        global_map: 全局真实地图 (非障碍物已标记为 FREE)
        global_free_count: 全局 FREE 格子总数 (预计算)
    """
    import copy
    # global_map 已经是标记了 FREE 的真实地图（保持不变）
    
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

    # 计算覆盖率 (基于机器人自己的认知地图 / 全局 FREE 格子数)
    coverage_ratio = calculate_coverage_ratio(robot_map, global_free_count)

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

        # 创建全局真实地图: 非障碍物格子标记为 FREE
        import copy
        global_map = copy.deepcopy(grid_map)
        global_map.grid[global_map.grid != GridMap.OBSTACLE] = GridMap.FREE
        global_free_count = int(np.sum(global_map.grid == GridMap.FREE))

        scenario_results = {}

        for algo_name in algorithms:
            print(f"  Running {algo_name}...")
            t0 = time.time()

            result = run_single_experiment(
                algo_name, grid_map, global_map, sensor, start_gx, start_gy,
                max_steps, global_free_count
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

    # 自定义场景列表（7个场景）
    custom_scenarios = [
        ("Custom1_分散矩形", custom_scenario1),
        ("Custom2_L形组合", custom_scenario2),
        ("Custom3_十字分散", custom_scenario3),
        ("Custom4_四角分散", custom_scenario4),
        ("Custom5_中央大方块", custom_scenario5),
        ("Custom6_走廊分割", custom_scenario6),
        ("Custom7_复杂组合", custom_scenario7),
    ]

    # 运行自定义场景实验 (7个场景)
    print("\n\nRunning custom scenarios (7 scenarios)...")
    results = run_experiments(custom_scenarios, algorithms, max_steps=30000,
                              save_dir="results/all_test")

    # 打印汇总表
    print_summary_table(results)

    # 生成可视化图表
    print("\n\nGenerating visualizations...")
    generate_visualizations(results, "results/all_test")

    print(f"\nEnd time: {datetime.now()}")
    print("All experiments complete!")

    return results


if __name__ == "__main__":
    main()
