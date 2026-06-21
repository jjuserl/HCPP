"""
运行 HCPP 实验 (Run HCPP Experiments)

用法:
    python run_experiment.py              # 运行所有实验 (8标准 + 2复杂场景)
    python run_experiment.py --quick      # 快速测试 (2个场景)
    python run_experiment.py --single     # 单场景测试 (HCPP算法, Scenario1)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hcpp.experiment import main as run_all
from hcpp.experiment import (run_single_experiment, print_summary_table,
                              generate_visualizations, calculate_coverage_ratio)
from hcpp.scenarios import ScenarioGenerator
from hcpp.sensor import SensorModel
from hcpp.visualization import plot_map, plot_map_with_direction, plot_coverage_progress
from hcpp.map_grid import GridMap


def quick_test():
    """
    快速测试: 仅运行2个场景，用于快速验证代码正确性
    """
    print("Running quick test (2 scenarios)...")
    algorithms = ["HCPP", "BSA", "FS-STC", "SP2E", "Epsilon*"]
    scenarios = ScenarioGenerator.get_all_scenarios()[:2]

    from hcpp.experiment import run_experiments
    results = run_experiments(scenarios, algorithms, max_steps=15000,
                              save_dir="results/quick_test")
    print_summary_table(results)
    generate_visualizations(results, "results/quick_test")
    return results


def single_test():
    """
    单场景测试: 仅运行 Scenario1_Random + HCPP 算法
    用于调试和验证算法实现
    """
    print("Running single test (Scenario1_Random + HCPP)...")
    grid_map, (start_gx, start_gy) = ScenarioGenerator.scenario1_random()
    # global_map: 全局真实地图（用于传感器仿真）
    # robot_map: 机器人的认知地图（初始全 UNKNOWN，逐步探索）
    global_map = grid_map
    # 将 global_map 中非障碍物格子标记为 FREE（场景生成器只标了障碍物）
    global_map.grid[(global_map.grid != GridMap.OBSTACLE)] = GridMap.FREE
    # 预计算全局非障碍物格子总数（用于覆盖率计算）
    total_non_obstacle = int(np.sum(global_map.grid != GridMap.OBSTACLE))

    robot_map = GridMap(grid_map.width * grid_map.resolution,
                         grid_map.height * grid_map.resolution,
                         grid_map.resolution)
    sensor = SensorModel(range_max=3.0, angle_resolution=2.0, fov=360.0)

    from hcpp.hcpp_planner import HCPPPlanner
    planner = HCPPPlanner(robot_map, global_map, sensor)
    planner.initialize(start_gx, start_gy)

    print(f"Start position: ({start_gx}, {start_gy})")
    print(f"Map size: {robot_map.width}x{robot_map.height}")
    print(f"Total non-obstacle cells: {total_non_obstacle}")

    max_steps = 1500
    for step in range(max_steps):
        result = planner.step()
        if result is None:
            print(f"Coverage complete after {step} steps!")
            break
        if step % 500 == 0:
            # 覆盖率 = 机器人已覆盖格子 / 全局非障碍物格子
            covered = int(np.sum(robot_map.grid == GridMap.COVERED))
            cov = covered / total_non_obstacle if total_non_obstacle > 0 else 0.0
            print(f"  Step {step}: coverage={cov:.2%} ({covered}/{total_non_obstacle}), "
                  f"CTS tasks={len([t for t in planner.cts if not t.completed])}")

    # 最终统计
    covered = int(np.sum(robot_map.grid == GridMap.COVERED))
    cov = covered / total_non_obstacle if total_non_obstacle > 0 else 0.0
    print(f"\nFinal coverage: {cov:.2%} ({covered}/{total_non_obstacle})")
    print(f"Path length: {len(planner.path)}")
    print(f"Local extremums: {planner.extremum_count}")
    print(f"Computation time: {planner.comp_time:.3f}s")
    print(f"CTS cells: {len(planner.cts)}")

    import copy
    visual_map = copy.deepcopy(robot_map)

    plot_map(visual_map, planner.path, title="HCPP - Single Test (Scenario1_Random)",
             save_path="results/single_test.png")
    print("Plot saved to results/single_test.png")
    
    plot_map_with_direction(visual_map, planner.path,
                           title="HCPP - Path with Direction (Scenario1_Random)",
                           save_path="results/single_test_direction.png",
                           arrow_interval=5)
    print("Direction plot saved to results/single_test_direction.png")

    from hcpp.visualization import plot_hcpp_cells
    plot_hcpp_cells(visual_map, planner.cts, planner.path, 1,
                    title="HCPP Cell Decomposition - Single Test",
                    save_path="results/single_test_cells.png")
    print("Cells plot saved to results/single_test_cells.png")

    global_visual = copy.deepcopy(global_map)
    global_visual.grid[robot_map.grid == GridMap.COVERED] = GridMap.COVERED
    plot_map(global_visual, planner.path,
             title="Global Map with Robot Path (Scenario1_Random)",
             save_path="results/single_test_global_map.png")
    print("Global map plot saved to results/single_test_global_map.png")

    plot_map(global_map, [],
             title="True Global Map (Scenario1_Random)",
             save_path="results/single_test_true_map.png")
    print("True map plot saved to results/single_test_true_map.png")

    return planner


if __name__ == "__main__":
    if "--quick" in sys.argv:
        quick_test()
    elif "--single" in sys.argv:
        single_test()
    else:
        run_all()
