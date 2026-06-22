"""
HCPP - 分层覆盖路径规划 (Hierarchy Coverage Path Planning)
     带主动极值预防 (Proactive Extremum Prevention)

论文复现：
"Hierarchy Coverage Path Planning With Proactive Extremum
 Prevention in Unknown Environments"

包含算法：
- HCPP: 分层覆盖路径规划（Boustrophedon 单元分解 + GTP + LPP）
- BSA: 回溯螺旋算法 (Backtracking Spiral Algorithm)
- FS-STC: 生成树覆盖法 (Full Spiral Spanning Tree Coverage)
- SP2E: 8邻域螺旋覆盖法 (Spiral Coverage with Proactive Prevention of Extremum)
- Epsilon*: 多尺度势场法 (Epsilon-star Multi-scale Potential Field)
"""

from .map_grid import GridMap
from .sensor import SensorModel
from .hcpp_planner import HCPPPlanner
from .baselines import BSAPlanner, FSSTCPlanner, SP2EPlanner, EpsilonStarPlanner
from .scenarios import ScenarioGenerator, custom_scenario1, custom_scenario2, show_scenario
from .experiment import run_experiments, main