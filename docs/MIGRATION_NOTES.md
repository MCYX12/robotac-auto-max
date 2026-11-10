# ROBOTAC 本机 Docker 自主建图

当前部署保留原迁移包名称，但默认采集预算为 **30 分钟**，速度上限 **0.45 m/s**。日常操作见 [`docs/DOCKER_USAGE.md`](DOCKER_USAGE.md)。

## 启动与停止

在项目根目录执行 `bash docker.sh start`，自动启动仿真、校验传感器、建图、保存并核验质量。另一个终端执行 `bash docker.sh stop` 可保留部分结果后退出。`bash docker.sh status` 查看状态，GPU 异常时用 `bash docker.sh repair` 重建本项目容器。

程序依次进行启动环视、自主探索、稳定停车采墙，以及基于实时点云的地面/墙面/顶部补扫。卡住检测按有效运动时间累计；传感器中断、坐标系跳变、关键组件退出或磁盘不足时不能报成功。

## 环境与构建

- 宿主机 Ubuntu 24.04，已有镜像 `robotac-humble:local` 内为 Ubuntu 22.04 / ROS 2 Humble / Python 3.10。
- 专用容器 `robotac-auto-max`，项目挂载到 `/workspace`，ROS_DOMAIN_ID=24，SDK 仅用容器内回环地址；只在本机仿真中控制。
- 厂商仿真运行文件位于 `runtime/`，SDK 及许可证保留。**仓库未包含 `runtime/src/` 下的 7.5 GB 仿真程序**，运行前需按 [`runtime/README.md`](../runtime/README.md) 放回并用 `deployment/SHA256SUMS.runtime.txt` 校验。不要移除运行文件、源码、配置或许可证。
- 修改源码后先执行 `bash docker.sh build`；不要向宿主机混装 Humble。运行期间修改源码、配置或列明程序文件会导致版本核对失败。
- 深度编码和相机标定必须匹配；启动检查会拒绝不合法的传感器输入。

## 地图与验证

每轮结果保存在 `dev_ws/midterm/runs/编号/`，包含原生 PCD、预览、运行日志、质量报告、SHA256 和源码/标定/程序版本记录。编号不可复用，历史结果不覆盖。

保存完成不等于验收完成。`COMPLETED` 仅表示本轮采集和全部设定的质量检查通过；`NEEDS_REVIEW`、`FAILED`、`CANCELLED` 分别表示质量未通过、故障、主动停止。异常退出时独立的 fault checkpoint 也会在结果记录中列明。

建图使用本机仿真的里程计锚定体素融合，包含实际深度地面采样和近地面归一化；不是带回环优化的完整 SLAM。简化场景几何只在采集完成后离线核验，不参与导航、不生成或填充点云。覆盖率按明确的距离容差计算，不能证明任意环境绝对零漂移、零缺口。

开发复测可在本机无其他建图任务时执行 `python3 deployment/validation/repeat_runs.py --count 2`，进行两轮 30 分钟采集及原生坐标一致性检查；每次尝试都保留，不隐藏失败重跑。

## 证据与清单

- `deployment/SHA256SUMS.runtime.txt`：厂商程序清单（代码启动时校验）。
- `docs/provenance/SHA256SUMS.original.txt`：最初迁移清单。
- `docs/provenance/SHA256SUMS-2026-11-09.txt`：整理前源码清单快照。
- `history/`（迁移前、质量修复前的源码备份）与本轮 `runs/`、`deployment/logs/`、验证原始数据体积较大，未纳入 git 仓库，仅在原工作目录保留。
