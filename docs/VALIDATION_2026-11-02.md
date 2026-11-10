# ROBOTAC Docker 部署与清理记录 — 2026-11-02

## 保留的部署

- 项目目录：`<项目根目录>`。
- Docker 镜像：`robotac-humble:local`，ID `sha256:c845efc11572da2d1e39fe870232b61e933a57343062770287cebd3f826005cb`。
- 容器：`robotac-auto-max`，使用现成镜像，测试结束后已主动停止，后续 `bash docker.sh sim` 会自动启动。
- 从回收站中的旧第二版移入 `runtime/` 一套匹配的 Matrix/UE、MuJoCo、控制器及场景配置；原算法包不含仿真资源，故这部分是必要保留内容。
- 原迁移包 52 项 SHA256 全部通过；未修改算法入口与源码。新增 `docker.sh`、`deployment/` 和本机说明。

## 实测结果

- 两个 ROS 包从源码编译成功；SDK导入、动态库、依赖导入通过。
- 12 秒内雷达 120 帧、里程计 2401 条、深度 120 帧。
- SDK在本机仿真中连接、站立并进入 ACTIVE；深度融合与在线前沿探索实际运行。
- RViz 创建地图纹理成功，OpenGL 4.6。
- 运行编号：`auto_max_deploy_20261102_01`。采集预算 840 秒，任务状态记录完成于 842.007 秒；启动、保存与预览另有开销。
- 最终状态：`COMPLETED`，原因 `native_pcd_saved_and_validated`，整条运行脚本退出码 0。
- 原生 PCD：1,985,482 个有限点，23,825,960 字节；无效点比例 0，包内 artifact validator 为 PASS，SHA256校验通过。
- 最终结果路径：`<项目根目录>/dev_ws/midterm/runs/auto_max_deploy_20261102_01`。
- 已查看俯视和侧视预览，见同目录 PNG。包内校验用于点云格式、有限点、尺寸和地面等基础有效性，不是官方评分或覆盖率测量。

## 清理结果

- 109 个已确认旧文件/目录目标全部不存在；包括旧工作树、回收站副本、旧教程/视频/补丁、任务三构建资料、旧会话备份、微信重复附件和指定网站存储。
- 删除旧容器：`task3-offline-rehearsal-20261023`、`task3-dev-ubuntu22`、`matrix-second-sim`、`robotac-finals-sim`。
- 删除镜像：`task3:ubuntu22.04`、`robotac-finals:local`；删除旧别名 `matrix-dev:latest`；保留当前部署所需 `robotac-humble:local`。
- 根分区从约 203 GiB 已用/30 GiB 可用/88%，降至约 128 GiB 已用/105 GiB 可用/55%，释放约 75 GiB。
- 文件与 Docker 操作明细保存在同名 JSON 记录中。

## 范围边界

未扫描未挂载的 NTFS 分区、U盘、远端设备；权限受限目录没有宣称完整扫描。未删除无关项目、通用 ROS/CUDA/MuJoCo/Isaac 环境、无法精确归属的 Docker 缓存/卷，以及未辨认归属的回收站 image.png 和 20260812-023552.mp4。Codex 当前会话和技能记忆不在清理范围。

## 使用

详见 [`docs/DOCKER_USAGE.md`](DOCKER_USAGE.md)。终端一运行 `bash docker.sh sim`；终端二先 `bash docker.sh check`，再 `bash docker.sh run 新编号`。
