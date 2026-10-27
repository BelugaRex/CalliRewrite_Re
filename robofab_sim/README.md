# RoboFab：评分验收入口

## 当前交付与边界

主线是**输入单个汉字 → KanjiVG 有序笔画 → 自动采样与笔压 → Panda IK/FK → 笔刷接触 → 宣纸墨迹**。
没有逐字手写轨迹，也不依赖 TensorFlow、强化学习或 GPU。输入必须存在于 KanjiVG；不声称支持全部汉字或任意图像。

“未见字”在本交付中指**本项目改动前内置/演示样例以外的整体字符**：春、江、月、夜；并非机器学习训练集未见。
四字本来就在 KanjiVG 中，正是使用评分条款明确允许的数据集路线。最终分数与“未见”的课堂口径仍由教师认定，不能保证分数。

这是**运动学 + 参数化笔刷接触模拟**：向下工具轴、关节限位、实际 FK 误差均有检查；未实现全臂碰撞检测、动力学、真实力控制或硬件安全控制。
压力是按笔画类型自动生成的启发式量；日志中的 N 为模型估计，不是传感器测量。墨量固定，不模拟消耗。
`seq_extract_to_trajectory.py` / `infer.sh` 为未验收的实验性图像桥接，窗口/游标/贝塞尔/宽度语义尚未修复，**不作为本次 100 分路线证据**。

## 对照反馈

| 条款 | 实现与证据 |
|---|---|
| 60：顺序执行、落笔/抬笔、完整可辨汉字 | 每个接触采样实际求 IK，再由 FK 和笔刷接触落墨；`evidence/01_U6C38/` 等包含整字、关节与接触日志 |
| 80：深度/压力改变宽度与浓淡，起行收变化 | 连续压力—深度映射；force/radius 驱动画布 RGB 混合；`pressure_calibration.png`、`execution.npz` 和视频可核验 |
| 80：三个汉字各两次 | 永、永、一、一、人、人，`01`—`06`；每次独立执行并留存结果，配对结果可重复 |
| 100：自动决定分解、顺序、路径、waypoint，无逐笔指定 | XML 笔画顺序 + SVG 曲线采样 + 类型压力轮廓 + 自动安全转场；`trajectory.json` 留存生成结果 |
| 100：四个新字自动书写 | 春 9 笔、江 6 笔、月 4 笔、夜 8 笔，`07`—`10`；视频对应第 7—10 次 |
| 原提案：原始字符与书写结果对照 | `reference_comparison.png`：左为输入字符对应的 KanjiVG 中心线，右为实际 IK/FK + 笔刷墨迹，不冒充图像重建 |

## 直接查看

- `evidence/browser/demo.mp4`：**真实连接浏览器的 Viser 渲染**，左侧 Panda/纸面，右侧同 tick 墨迹放大；含六次基础与四次新字完整演示，超过 30 秒。不是离线假造的 3D 视频。
- `evidence/browser/snapshots/run_01.jpg`—`run_10.jpg`：各次完整书写后的真实浏览器帧。
- `evidence/contact_sheet.png`：十次实际运动学执行结果总览。
- `evidence/reference_comparison.png`：七个不同输入字的中心线/实际墨迹对照。
- `evidence/pressure_calibration.png` / `.json`：三个压力档的模型标定，较大压力应更宽、更深；它本身不是机器人演示。
- 每次目录：`canvas.png`、`trajectory.json`、`execution.npz`、`summary.json`。
- `manifest.json`：Unicode、XML ID、数据/源码/结果 SHA-256、误差、笔画/接触数和单位。
- `browser/capture.json` / `timing.json`：捕获时源码哈希、视频哈希、逐帧实际时刻和完成采样标签。中间 JPEG 帧不入库，可重录。

## 环境与复现

从仓库根目录运行，Python 3.11；依赖固定在 `robofab_sim/requirements.txt`，视频另需 `ffmpeg` 和 `ffprobe`。
首次加载 Panda 会由 `robot_descriptions` 下载官方 URDF/mesh，需要网络；之后可用缓存。没有 GPU 需求。

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r robofab_sim/requirements.txt
bash robofab_sim/data/download_kanjivg.sh
.venv/bin/python robofab_sim/calligraphy_sim.py --char 春 --port 8086
```

打开该端口的本地/转发浏览器。执行按钮支持暂停/继续/完成后从头重播；换字重置状态；Step Stroke 从当前进度运行到本笔结束，初始时从第一笔开始。重置机械臂不会自动清纸。

```bash
# 无浏览器、真实 IK/FK + 接触执行，不是 SVG preview
.venv/bin/python robofab_sim/acceptance.py
# 完整回归 + 每个保存关节样本重新 FK + 哈希 + 视频/十次完成状态
.venv/bin/python robofab_sim/validate.py
# 只跑单元/回归测试（无 ffmpeg/browser 视频依赖）
.venv/bin/python -m unittest discover -s robofab_sim/tests -v
# 真实 Viser 录制：启动后连接浏览器，十次完成后自动编码并退出
.venv/bin/python robofab_sim/calligraphy_sim.py --port 8086 \
  --demo 永永一一人人春江月夜 --record-dir robofab_sim/evidence/browser
```

录制以固定空间采样执行，播放速度不改变路径。捕获是有时间戳的稀疏真实浏览器帧，编码为 20fps 持帧视频，不声称 20fps 独立物理采样。
GUI 连续运行的机械臂从上次实际姿态继续；离线验收每次从 rest 开始，因此两者总样本数可不同。
源码改变后哈希校验会拒绝旧证据：重新导出并重录，不可只改哈希冒充重测。

## 关键实现

- `kanjivg_to_trajectory.py`：XML 查字；SVG M/L/C/Q/Z 解析；0.15m 字框；起行收连续笔压。不支持的 SVG 命令显式报错。
- `execution.py`：先竖直抬笔，再在安全高度水平转场，落笔书写，最后竖直抬笔；0.8mm 最大 Cartesian 步长；GUI/离线共用。
- `calligraphy_sim.py`：补齐 Panda hand -45° 固定旋转；位置与向下轴 DLS IK；RGB 光学覆盖，不再丢弃 alpha；暂停和重复 stamp 不加深。
- `validate.py`：单元测试、全部保存姿态 FK、关节范围、全笔覆盖、转场无接触、重复一致、压力单调性、完整十次视频与源码溯源。

## 授权

KanjiVG 发布版 `r20260714`，XML SHA-256 为 `293dc9741cf6b6cb47aabb6235dca05da0ae4210590e139b5b1aebe37ddf72a4`。
字形路径及其变换/渲染衍生数据按 CC BY-SA 3.0 分发；详见 `evidence/NOTICE.md`。原项目与其他资产保留各自许可证，不将其重新授权。