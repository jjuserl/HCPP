# HCPP - 分层覆盖路径规划 (Hierarchy Coverage Path Planning)

## 项目简介

本项目复现了论文 **"Hierarchy Coverage Path Planning With Proactive Extremum Prevention in Unknown Environments"** 的核心算法，并实现了四种基线算法进行对比。

### 核心思想

在未知环境中进行覆盖路径规划时，机器人容易陷入**局部极值**（死胡同），导致无法完成全局覆盖。HCPP 算法通过**分层覆盖路径规划**，将覆盖任务分解为任务管理（Task Management）、全局巡游规划（Global Tour Planning, GTP）和局部路径规划（Local Path Planning, LPP）三个层次，实现**主动极值预防**，优先处理边缘区域从而避免机器人陷入困境。

## 算法列表

| 算法 | 全称 | 核心策略 |
|------|------|----------|
| **HCPP** | Hierarchy Coverage Path Planning | Boustrophedon 单元分解 + GTP + LPP |
| **BSA** | Backtracking Spiral Algorithm | 回溯法，陷入极值时回溯到最近未完全覆盖节点 |
| **FS-STC** | Full Spiral Spanning Tree Coverage | 生成树覆盖法，沿虚拟生成树边界移动 |
| **SP2E** | Spiral Coverage with 8-neighborhood | 8邻域螺旋覆盖，动态改变螺旋方向 |
| **Epsilon\*** | Epsilon-star Multi-scale Potential Field | 多尺度势场法，陷入局部极值时通过势场梯度逃离 |

## 项目结构

```
Extremum_prevention/
├── run_experiment.py          # 实验入口脚本
├── README.md                  # 项目文档
├── hcpp/                      # 核心代码包
│   ├── __init__.py            # 包初始化
│   ├── map_grid.py            # 网格地图表示（2D 占用栅格）
│   ├── sensor.py              # 模拟传感器模型（360° LiDAR）
│   ├── hcpp_planner.py        # HCPP 算法实现
│   ├── baselines.py           # 基线算法（BSA, FS-STC, SP2E, Epsilon*）
│   ├── scenarios.py           # 实验场景生成（10 个场景）
│   ├── experiment.py          # 实验运行与评估模块
│   └── visualization.py       # 可视化图表生成
└── results/                   # 实验结果输出目录
    ├── results.json           # 详细数值结果
    ├── comparison_charts.png   # 算法对比柱状图
    ├── comparison_table.png    # 汇总表格
    ├── Scenario*.png          # 各场景覆盖路径图
    └── Scenario*_progress.png # 覆盖率进展曲线
```

## 环境要求

- Python 3.7+
- 依赖包：
  - numpy
  - matplotlib

安装依赖：

```bash
pip install numpy matplotlib
```

## 快速开始

### 1. 运行单场景测试（验证代码正确性）

```bash
python run_experiment.py --single
```

仅运行 `Scenario1_Random` + HCPP 算法，输出覆盖率和路径图。

### 2. 快速测试（2 个场景）

```bash
python run_experiment.py --quick
```

运行前 2 个场景，5 种算法对比，结果保存在 `results/quick_test/`。

### 3. 运行完整实验

```bash
python run_experiment.py
```

运行全部 10 个场景（8 标准 + 2 复杂），5 种算法对比。

## 实验场景

### 标准场景（8 个，300m x 200m）

| 场景 | 描述 |
|------|------|
| Scenario1_Random | 随机矩形障碍物 |
| Scenario2_Sparse | 稀疏圆形障碍物 |
| Scenario3_Dense | 密集小障碍物 |
| Scenario4_Columns | 网格状排列的圆柱体 |
| Scenario5_Walls | 墙壁形成走廊（带缺口） |
| Scenario6_Maze | 简易迷宫结构 |
| Scenario7_Corner | 障碍物集中在角落 |
| Scenario8_Rooms | 多房间结构（带门洞） |

### 复杂场景（2 个）

| 场景 | 尺寸 | 描述 |
|------|------|------|
| Scenario_Island | 100m x 100m | 中心不规则岛屿障碍物 |
| Scenario_Indoor | 100m x 80m | 室内布局（客厅/厨房/卧室/走廊） |

## 评估指标

| 指标 | 说明 |
|------|------|
| **覆盖率 (Coverage Ratio)** | 已覆盖自由格子 / 总自由格子 |
| **路径长度 (Path Length)** | 机器人移动步数 |
| **局部极值数 (Local Extremums)** | 陷入死胡同的次数 |
| **计算时间 (Computation Time)** | 纯算法计算耗时（秒） |
| **转弯次数 (Num Turns)** | 路径方向变化次数 |

## 实验结果 (修正后)

### 覆盖率与性能汇总 (100x70, Scenario1_Random)

| 指标 | HCPP | BSA | SP2E | Epsilon* |
|------|------|-----|------|----------|
| **覆盖率** | **100.0%** | 83.5% | 85.1% | 87.8% |
| **路径长度** | 981 | 322 | 328 | 357 |
| **局部极值数** | **0** | 301 | 5673 | 126 |
| **计算时间** | **2.4s** | 2.8s | 47.9s | 2.7s |

### 关键发现

1. **HCPP** 是唯一实现 **100% 覆盖率** 的算法，且局部极值数为 **0**，验证了论文的主动极值预防机制
2. **BSA/SP2E/Epsilon\*** 在达到约 83-88% 覆盖率后陷入局部极值无法继续
3. **SP2E** 极值数为 5673（几乎步步卡住），计算时间 47.9s 远超其他算法
4. **HCPP** 的计算时间（2.4s）与其他快速算法相当，但覆盖率和极值预防远超它们
5. 论文核心贡献得到验证：全局巡游规划（brim-first GTP）+ 局部边界保持路径（LPP）有效预防了局部极值

## 核心算法说明

### HCPP 规划器 (`hcpp_planner.py`)

HCPP 将覆盖路径规划分解为三个层次：

**1. 任务管理 (Task Management, Algorithm 1)**
- Boustrophedon 矩形单元分解：以宽度 `r1`（任务执行器宽度）扫描已知自由空间，生成覆盖任务结构（CTS）
- CTS 五元组：`{ID, px, pu, pd, adj}` 存储单元信息（ID、中线 x 坐标、上下端点 y 坐标、相邻单元列表）
- 动态任务更新：
  - 障碍物边界探索：检测到新障碍物时，BFS 探索障碍物完整边界，删除重叠任务，分割为新的子任务
  - 单元覆盖更新：机器人覆盖单元部分区域时，若中间被覆盖则分割为多个新单元，若仅端点变化则更新端点
  - 新区域扩展：新探索的自由空间自动创建对应的覆盖单元
  - 计算复杂度优化：仅检查与障碍物边界相交的单元

**2. 全局巡游规划 (Global Tour Planning, Algorithm 2)**
- 邻接图构建：单元映射为无向图顶点，公共边界映射为边
- 顶点分类：边缘顶点（Brim vertex，度数 ≤ 1）和内部顶点（Internal vertex，度数 > 1）
- 巡游生成：优先选择边缘顶点，若候选为内部顶点则用 BFS 搜索第一个边缘顶点
- 动态更新：CTS 新增任务时重新构建邻接图并生成全局巡游

**3. 局部路径规划 (Local Path Planning, Algorithm 3)**
- 核心原则：沿未覆盖区域边界移动，保持未覆盖区域连通性
- 路径生成规则：
  - 相邻/不连通单元：直接规划到较近端点，沿单元中线覆盖
  - 连通但不相邻单元：使用右手法则（顺时针）和左手法则（逆时针）分别生成两条边界路径，选择较短者
- 边界追踪：沿未覆盖区域（FREE 格子）与障碍物/未知区域的边界移动

**主循环 (Algorithm 4)**
```
循环执行：
1. 传感器感知 → 标记 FREE 和 COVERED
2. 更新 CTS（障碍物边界探索、覆盖端点更新、新区域扩展）
3. 若 CTS 有变更，重新生成全局巡游
4. 沿全局巡游生成局部路径（当前任务完成后再前进到下一任务）
5. 执行局部路径运动
6. 终止条件：所有覆盖单元均标记为已完成，且无剩余 FREE/UNKNOWN 格子
```

### 基线算法 (`baselines.py`)

**Epsilon\* (ε\*)**
- 多尺度势场法：构建势场函数 U(x,y) = exp(-d/ε)，从所有前沿格子（FREE 邻接 UNKNOWN）向外扩散
- ε 参数控制势场衰减尺度：ε 增大时远处前沿也能产生足够吸引力
- 陷入局部极值时，沿势场梯度方向逃离
- 使用多源 BFS 传播势场，搜索势场梯度最大的方向

**BSA (回溯螺旋算法)**
- 回溯法：记录已访问节点和回溯栈
- 优先沿当前方向覆盖未覆盖区域
- 陷入局部极值时，回溯到最近"未完全覆盖"节点（周围 3x3 区域仍有 UNKNOWN 格子）
- 沿回溯栈反向查找，找到第一个未完全覆盖的节点继续覆盖

**FS-STC (生成树覆盖法)**
- 构建虚拟生成树：每个节点代表 2x2 粗网格单元，使用 DFS 构建
- 沿生成树边界顺时针移动，覆盖当前单元后沿边移动到子节点
- 节点无未覆盖区域时，沿生成树回溯到父节点
- 每 10 步重建生成树以适应新探索区域
- 主动预防极值：生成树保证连通性，避免死胡同

**SP2E (8邻域螺旋覆盖)**
- 使用 8 邻域连接，以螺旋方式向外覆盖
- 动态改变螺旋方向预防极值：仅连续卡住 3 步以上才切换方向
- 螺旋中心动态更新：当机器人远离当前中心时自动更新
- 螺旋半径随卡住次数递增，扩大搜索范围

## 许可

本代码仅用于学术研究和论文复现目的。

## 参考文献

> "Hierarchy Coverage Path Planning With Proactive Extremum Prevention in Unknown Environments"