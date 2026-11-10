# runtime/ — 厂商仿真运行文件

本目录存放本机仿真所需的厂商运行配置与许可证。**运行文件本体（约 7.5 GB）未纳入 git 仓库**：它们超过 GitHub 的单文件与仓库体积限制，且属于厂商分发包，应由使用者从原始发布包获取。

## 仓库内包含

| 路径 | 说明 |
| --- | --- |
| `LICENSE` | 厂商 BSD-3-Clause 许可 |
| `config/config*.json` | 仿真传感器配置，`deployment/simulation.py` 启动时会把 `config/config.json` 复制到 UeSim 场景配置 |
| `../deployment/SHA256SUMS.runtime.txt` | 三个厂商可执行文件的 SHA256 清单 |

## 需要自行放回的运行文件

按下列目录结构放置（相对项目根目录）：

```text
runtime/src/
├── UeSim/Linux/UeSim/Binaries/Linux/UeSim            # UE5 仿真场景（约 5.9 GB）
├── robot_mc/build/export/mc/bin/mc_ctrl              # 机器狗控制器（约 1.1 GB）
├── robot_mc/build/export/config/sdk_config.yaml      # SDK 配置
├── robot_mujoco/simulate/build/robot_mujoco          # MuJoCo 物理仿真（约 610 MB）
└── robot_mujoco/zsibot_robots/xgb/unreal.xml         # 机器人/场景描述
```

## 校验

放回后先校验，哈希不一致程序会直接拒绝启动（`vendor_executable_checksum_mismatch`）：

```bash
cd <项目根目录>
sha256sum -c deployment/SHA256SUMS.runtime.txt
```

清单内容：

```text
dc10d13b...dc80  runtime/src/UeSim/Linux/UeSim/Binaries/Linux/UeSim
4fa0c4a3...dea0  runtime/src/robot_mc/build/export/mc/bin/mc_ctrl
8f96a848...6e1c  runtime/src/robot_mujoco/simulate/build/robot_mujoco
```

此外，`deployment/provenance.py` 每轮启动时会对源码、配置、SDK 动态库与上述厂商程序做指纹，运行期间被修改会导致本轮判定失败。
