# 未入 Git 的阶段原型（历史归档）

这里保存外层阶段目录中 **3 份独有源码的原始快照**，仅用于队友审查是否保留，不作为自动书写或评分达标证据。没有替换 `panda_calligraphy/` 或 `robofab_sim/`。

## 文件

- `stage2_80pts/calligraphy_panda_viser_sim.py`：旧 Panda viewer 的手工轨迹变体，主要差异在 `CALLIGRAPHY_TRAJECTORIES`。
- `stage3_100pts/calligraphy_panda_viser_sim.py`：另一组手工轨迹；目录名来自历史实验，**并未自动分解字符**。
- `stage3_100pts/seq_to_trajectory.py`：独立 NPZ 转换实验，viewer 未调用它；窗口/游标/缩放/压力语义未验证，不是现有实验桥接的修复版。

## 不重复归档

外层 stage1 viewer 与 `panda_calligraphy/calligraphy_panda_viser_sim.py` 逐字节相同。外层各份 `deformable_brush_physics.py` 也与仓库已有模型相同，故没有复制；XML 数据复用 `robofab_sim/data/download_kanjivg.sh`。

## 运行边界

这不是独立安装包。旧 viewer 依赖 `numpy`、`Pillow`、`viser`、`yourdfpy`、`robot_descriptions`；首次加载官方 Panda URDF/mesh 可能联网。

若要启动归档 viewer，需把仓库 `panda_calligraphy/` 加入 `PYTHONPATH`，从而复用原笔刷模块。否则脚本捕获 ImportError 后可能静默使用内置 fallback；不能把这种运行当作原模型验证。按仓库根目录解析模块路径，不依赖本机绝对路径。

NPZ 脚本仅做 Python 语法检查，未进行端到端视觉验证。旧 viewer 的轨迹 pressure 字段与渲染浓淡处理存在限制，参见 `docs/LOCAL_WORK_REVIEW.md`。保留这些文件是为了避免遗漏用户成果，不代表建议合并它们。
