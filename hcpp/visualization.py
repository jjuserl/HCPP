"""
可视化模块 (Visualization Module)

为 HCPP 实验生成高质量可视化图表:
- 网格地图 + 覆盖路径叠加
- 覆盖率随步数进展曲线
- 算法对比柱状图
- 场景对比图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from .map_grid import GridMap


# 全局样式设置
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    # 中文字体支持（优先顺序）
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})


# ---------- 辅助函数 ----------

def _draw_grid_lines(ax, h, w):
    """
    在地图上叠加网格线，间距根据地图尺寸自动选择，避免过密。
    """
    max_dim = max(h, w)
    if max_dim <= 40:
        step = 1
    elif max_dim <= 80:
        step = 2
    elif max_dim <= 160:
        step = 5
    else:
        step = 10

    ax.set_xticks(np.arange(-0.5, w, step), minor=False)
    ax.set_yticks(np.arange(-0.5, h, step), minor=False)
    ax.grid(True, which='major', color='gray', linewidth=0.3, alpha=0.5)
    ax.tick_params(length=2, labelsize=6)


# ---------- 公共 API ----------

def plot_map(grid_map, path=None, title="Map", save_path=None, show_start=True):
    """
    绘制网格地图，叠加覆盖路径

    颜色方案:
    - 灰色: 未知区域
    - 白色: 自由空间(未覆盖)
    - 黑色: 障碍物
    - 绿色: 已覆盖区域
    - 蓝色线: 覆盖路径
    - 绿色圆: 起点
    - 红色圆: 终点
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    h, w = grid_map.grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    img[grid_map.grid == GridMap.UNKNOWN] = [0.65, 0.65, 0.65]   # 灰色
    img[grid_map.grid == GridMap.FREE] = [1.0, 1.0, 1.0]          # 白色
    img[grid_map.grid == GridMap.OBSTACLE] = [0.1, 0.1, 0.1]      # 深灰
    img[grid_map.grid == GridMap.COVERED] = [0.5, 0.9, 0.5]       # 绿色

    ax.imshow(img, origin='lower', interpolation='nearest')

    # 绘制网格线
    _draw_grid_lines(ax, h, w)

    # 绘制覆盖路径 (蓝色实线)
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color='blue', linewidth=1.0, alpha=0.8)

    # 标记起点和终点
    if show_start and path and len(path) > 0:
        ax.plot(path[0][0], path[0][1], 'o', color='lime',
                markersize=10, markeredgecolor='darkgreen',
                markeredgewidth=1.5, label='Start', zorder=5)
        if len(path) > 1:
            ax.plot(path[-1][0], path[-1][1], 's', color='red',
                    markersize=10, markeredgecolor='darkred',
                    markeredgewidth=1.5, label='End', zorder=5)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    ax.legend(loc='upper right', framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()

    return fig


def plot_map_with_direction(grid_map, path=None, title="Map with Direction", 
                            save_path=None, show_start=True, arrow_interval=5):
    """
    绘制网格地图，叠加带方向箭头的覆盖路径

    颜色方案:
    - 灰色: 未知区域
    - 白色: 自由空间(未覆盖)
    - 黑色: 障碍物
    - 绿色: 已覆盖区域
    - 蓝色线: 覆盖路径
    - 青色箭头: 前进方向
    - 绿色圆: 起点
    - 红色圆: 终点

    参数:
        grid_map: GridMap 对象
        path: 机器人路径 [(x,y), ...]
        title: 图标题
        save_path: 保存路径
        show_start: 是否显示起点和终点
        arrow_interval: 箭头间隔（每隔多少步画一个箭头）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    h, w = grid_map.grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    img[grid_map.grid == GridMap.UNKNOWN] = [0.65, 0.65, 0.65]   # 灰色
    img[grid_map.grid == GridMap.FREE] = [1.0, 1.0, 1.0]          # 白色
    img[grid_map.grid == GridMap.OBSTACLE] = [0.1, 0.1, 0.1]      # 深灰
    img[grid_map.grid == GridMap.COVERED] = [0.5, 0.9, 0.5]       # 绿色

    ax.imshow(img, origin='lower', interpolation='nearest')

    # 绘制网格线
    _draw_grid_lines(ax, h, w)

    # 绘制覆盖路径和方向箭头
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        
        # 绘制路径线（渐变颜色）
        for i in range(len(px) - 1):
            ratio = i / (len(px) - 1)
            color = (0.2 + ratio * 0.6, 0.4 + ratio * 0.4, 0.8)  # 从深蓝到浅蓝
            ax.plot([px[i], px[i+1]], [py[i], py[i+1]], 
                    color=color, linewidth=1.5, alpha=0.9)
        
        # 绘制方向箭头
        for i in range(0, len(path) - 1, arrow_interval):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 计算箭头方向
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                # 箭头位置（稍微偏移以避免重叠）
                offset = 0.2
                ax.annotate(
                    '', 
                    xy=(x2 - dx * offset, y2 - dy * offset),
                    xytext=(x1 + dx * offset, y1 + dy * offset),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#00bcd4',  # 青色
                        linewidth=2,
                        alpha=0.9,
                        shrinkA=3,
                        shrinkB=3
                    ),
                    zorder=4
                )
            
            # 添加步数标签
            if i % (arrow_interval * 2) == 0:
                ax.text(x1, y1 - 0.3, str(i), 
                        fontsize=7, ha='center', va='top', 
                        color='darkblue', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # 标记起点和终点
    if show_start and path and len(path) > 0:
        ax.plot(path[0][0], path[0][1], 'o', color='lime',
                markersize=12, markeredgecolor='darkgreen',
                markeredgewidth=1.5, label='Start', zorder=5)
        ax.text(path[0][0] + 0.4, path[0][1], 'Start', 
                fontsize=8, ha='left', va='center', color='darkgreen')
        
        if len(path) > 1:
            ax.plot(path[-1][0], path[-1][1], 's', color='red',
                    markersize=12, markeredgecolor='darkred',
                    markeredgewidth=1.5, label='End', zorder=5)
            ax.text(path[-1][0] + 0.4, path[-1][1], f'End (step {len(path)-1})', 
                    fontsize=8, ha='left', va='center', color='darkred')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    ax.legend(loc='upper right', framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()

    return fig


def _cover_radius(grid, gx, gy, r):
    """用传感器半径 r 标记覆盖范围: dx^2 + dy^2 <= r^2"""
    r2 = r * r
    h, w = grid.grid.shape
    x_min = max(0, gx - r)
    x_max = min(w - 1, gx + r)
    y_min = max(0, gy - r)
    y_max = min(h - 1, gy + r)
    for ny in range(y_min, y_max + 1):
        row = grid.grid[ny]
        for nx in range(x_min, x_max + 1):
            if (nx - gx) ** 2 + (ny - gy) ** 2 <= r2 and row[nx] == GridMap.FREE:
                row[nx] = GridMap.COVERED


def plot_coverage_progress(paths_dict, grid_map, title="Coverage Progress",
                           save_path=None, sensor_range=3):
    """
    绘制多算法覆盖率随步数变化的进展曲线

    sensor_range: 传感器覆盖半径 (格数), 默认3 (对应 range_max=3.0, resolution=1.0)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    colors = {
        'HCPP': '#2196F3',
        'BSA': '#FF9800',
        'FS-STC': '#4CAF50',
        'SP2E': '#9C27B0',
        'Epsilon*': '#F44336',
    }

    for name, path in paths_dict.items():
        if path is None or len(path) == 0:
            continue

        # 采样
        n_samples = min(200, len(path))
        step_indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)
        coverage_vals = []

        for idx in step_indices:
            temp_grid = GridMap(grid_map.width * grid_map.resolution,
                               grid_map.height * grid_map.resolution,
                               grid_map.resolution)
            temp_grid.grid = grid_map.grid.copy()
            # Reset all covered cells first
            temp_grid.grid[temp_grid.grid == GridMap.COVERED] = GridMap.FREE
            # Apply sensor-radius coverage up to step idx
            for j in range(min(idx + 1, len(path))):
                gx, gy = path[j]
                if temp_grid.is_valid(gx, gy) and not temp_grid.is_obstacle(gx, gy):
                    _cover_radius(temp_grid, gx, gy, sensor_range)
            coverage_vals.append(temp_grid.get_coverage_ratio())

        color = colors.get(name, None)
        ax.plot(step_indices, coverage_vals, label=name, linewidth=2,
                color=color, alpha=0.85)

    ax.set_xlabel('Steps', fontweight='bold')
    ax.set_ylabel('Coverage Ratio', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    return fig


def plot_comparison_table(results, save_path=None):
    """
    绘制算法对比柱状图 (4个子图: 覆盖率/路径长度/极值数/计算时间)
    """
    scenarios = list(results.keys())
    algorithms = list(results[scenarios[0]].keys())
    metrics = ['coverage_ratio', 'path_length', 'local_extremums',
               'computation_time']
    metric_labels = ['Coverage Ratio', 'Path Length', 'Local Extremums',
                     'Computation Time (s)']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    bar_width = 0.15
    x = np.arange(len(scenarios))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        for j, algo in enumerate(algorithms):
            values = []
            for s in scenarios:
                v = results[s][algo].get(metric, 0)
                # 覆盖率转为百分比
                if metric == 'coverage_ratio':
                    v = v * 100
                values.append(v)
            offset = (j - len(algorithms) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width,
                         label=algo, color=colors[j], alpha=0.85,
                         edgecolor='white', linewidth=0.5)
            # 在柱子上标注数值
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.1f}' if metric != 'path_length' else f'{int(val)}',
                           ha='center', va='bottom', fontsize=6, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('Scenario', 'S').replace('_', '\n')
                           for s in scenarios], fontsize=7)
        ax.set_ylabel(label, fontweight='bold')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.suptitle('Algorithm Comparison Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    return fig


def plot_hcpp_cells(grid_map, cts, path, r1, title="HCPP Cell Decomposition",
                    save_path=None):
    """
    绘制 HCPP Boustrophedon 单元分解图。
    
    只显示单元划分，不显示覆盖状态。

    参数:
        grid_map: GridMap 对象
        cts: CoverageTask 列表
        path: 机器人路径 [(x,y), ...] (未使用)
        r1: 单元宽度（格数）
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    h, w = grid_map.grid.shape
    
    # 只显示白色背景和网格线
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.grid(True, which='major', color='gray', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal')

    half_w = r1 / 2.0

    # # 建立 task id -> task 索引
    # task_by_id = {t.id: t for t in cts}

    # # 从路径推断实际访问顺序
    # visit_order = []
    # seen_ids = set()
    # for (px, py) in (path or []):
    #     for t in cts:
    #         if t.id in seen_ids:
    #             continue
    #         y_min, y_max = t.y_range()
    #         if abs(px - t.px) <= half_w + 0.01 and y_min <= py <= y_max:
    #             visit_order.append(t.id)
    #             seen_ids.add(t.id)
    #             break

    # 定义高对比度颜色列表（相邻单元颜色明显区分）
    cell_colors = [
        [1.0, 0.7, 0.7],    # 粉红
        [0.7, 1.0, 0.7],    # 浅绿
        [0.7, 0.7, 1.0],    # 浅蓝
        [1.0, 1.0, 0.7],    # 浅黄
        [1.0, 0.7, 1.0],    # 浅紫
        [0.7, 1.0, 1.0],    # 浅青
        [1.0, 0.85, 0.6],   # 橙黄
        [0.85, 0.7, 1.0],   # 淡紫蓝
        [0.9, 1.0, 0.7],    # 黄绿
        [1.0, 0.7, 0.85],   # 玫红
    ]

    # 按 px 排序单元（确保顺序一致）
    sorted_cts = sorted(cts, key=lambda t: t.px)
    
    # 绘制单元矩形（带填充颜色）
    for idx, t in enumerate(sorted_cts):
        y_min, y_max = t.y_range()
        cell_h = y_max - y_min + 1
        x0 = t.px - half_w
        # 使用交替颜色，确保相邻单元颜色不同
        fill_color = cell_colors[idx % len(cell_colors)]
        rect = Rectangle(
            (x0, y_min - 0.5), r1, cell_h,
            linewidth=2.0, edgecolor='#333333', facecolor=fill_color,
            alpha=0.85
        )
        ax.add_patch(rect)

        # 单元 ID 标签
        cy = (y_min + y_max) / 2.0
        fontsize = 8 if len(cts) <= 30 else 6
        ax.text(t.px, cy, str(t.id),
                fontsize=fontsize, color='#000000',
                fontweight='bold', ha='center', va='center')

    # # 按访问顺序连接单元中心（橙色箭头）
    # for i in range(len(visit_order) - 1):
    #     t_cur = task_by_id.get(visit_order[i])
    #     t_nxt = task_by_id.get(visit_order[i + 1])
    #     if t_cur is None or t_nxt is None:
    #         continue
    #     y1_min, y1_max = t_cur.y_range()
    #     y2_min, y2_max = t_nxt.y_range()
    #     cx1, cy1 = t_cur.px, (y1_min + y1_max) / 2.0
    #     cx2, cy2 = t_nxt.px, (y2_min + y2_max) / 2.0
    #     ax.annotate(
    #         '', xy=(cx2, cy2), xytext=(cx1, cy1),
    #         arrowprops=dict(
    #             arrowstyle='->', color='#E65100',
    #             lw=1.0, alpha=0.65,
    #             connectionstyle='arc3,rad=0.05'
    #         )
    #     )

    # # 机器人路径（蓝色线）
    # if path and len(path) > 1:
    #     px = [p[0] for p in path]
    #     py = [p[1] for p in path]
    #     ax.plot(px, py, color='blue', linewidth=0.6, alpha=0.55)

    # # 起点和终点
    # if path and len(path) > 0:
    #     ax.plot(path[0][0], path[0][1], 'o', color='lime',
    #             markersize=10, markeredgecolor='darkgreen',
    #             markeredgewidth=1.5, label='Start', zorder=5)
    #     if len(path) > 1:
    #         ax.plot(path[-1][0], path[-1][1], 's', color='red',
    #                 markersize=10, markeredgecolor='darkred',
    #                 markeredgewidth=1.5, label='End', zorder=5)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()

    return fig


def plot_scenario_comparison(scenario_results, save_dir="results"):
    """
    绘制每个场景的算法对比图 (并排展示各算法的覆盖路径)

    scenario_results: list of (scenario_name, algo_results_dict)
      其中 algo_results_dict = {algo_name: {"grid": grid, "path": path}, ...}
      每个 algo 使用自己运行后的 grid，直接显示算法标记的 COVERED 区域。
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for scenario_name, algo_results in scenario_results:
        n_algos = len(algo_results)
        cols = min(3, n_algos)
        rows = (n_algos + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flatten()

        for idx, (algo_name, res) in enumerate(algo_results.items()):
            ax = axes[idx]

            algo_grid = res["grid"]
            path = res["path"]

            h, w = algo_grid.grid.shape
            img = np.zeros((h, w, 3), dtype=np.float32)

            # 直接使用算法运行后的网格状态 (包含传感器半径覆盖)
            img[algo_grid.grid == GridMap.UNKNOWN] = [0.65, 0.65, 0.65]
            img[algo_grid.grid == GridMap.FREE]     = [1.0, 1.0, 1.0]
            img[algo_grid.grid == GridMap.OBSTACLE] = [0.1, 0.1, 0.1]
            img[algo_grid.grid == GridMap.COVERED]  = [0.5, 0.9, 0.5]

            ax.imshow(img, origin='lower', interpolation='nearest')

            # 绘制网格线
            _draw_grid_lines(ax, h, w)

            # 绘制路径 (蓝色实线)
            if path and len(path) > 1:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                ax.plot(px, py, color='blue', linewidth=0.8, alpha=0.8)

                # 起点和终点
                ax.plot(path[0][0], path[0][1], 'o', color='lime',
                       markersize=8, zorder=5)
                ax.plot(path[-1][0], path[-1][1], 's', color='red',
                       markersize=8, zorder=5)

            # 直接用算法运行后的 grid 统计覆盖率
            total_free = np.sum(algo_grid.grid != GridMap.OBSTACLE)
            if total_free > 0:
                cov = np.sum(algo_grid.grid == GridMap.COVERED) / total_free
            else:
                cov = 0.0

            ax.set_title(f"{algo_name}\nCoverage: {cov:.1%}", fontsize=10,
                        fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        # 隐藏多余的子图
        for idx in range(len(algo_results), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(scenario_name, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{scenario_name}.png"),
                   dpi=200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
