# Inverse Optimization of Strange Attractors

把任意二维目标图像反求成一个混沌离散映射的物理不变测度——也就是
**逆 Frobenius–Perron 问题 (IFPP)** 在二维图像生成上的工程化求解。本仓库
按论文 *Inverse Optimization of Chaotic Attractors* 中描述的架构完整实现
了该算法，包含主优化器 (`main.py`) 和目标预处理脚本
(`preprocess_target.py`)，自包含、仅依赖 NumPy 与 Pillow。

```bash
uv run python main.py target.png --gens 80 --pop 24 --size 96 --out result.png
uv run python main.py target.png --model poly_sin3d --loss-mode multiscale --size 128
```

输出是一张并排的 `target | attractor` 灰度图。

如果输入是普通照片或线稿，建议先把它转换成更像“物理不变测度”的目标密度：

```bash
uv run python preprocess_target.py portrait.jpg \
    --out portrait_target.png \
    --preview portrait_target_preview.png \
    --crop

uv run python main.py portrait_target.png --gens 80 --pop 24 --size 96
```

实验性的 3D latent `poly_sin3d` 路径推荐这样跑：

```bash
uv run python main.py experiment_outputs/target_density_512.png \
    --model poly_sin3d \
    --loss-mode multiscale \
    --size 128 \
    --n-iter 180000 \
    --gens 200 \
    --out face_poly_sin3d.png
```

---

## 1. 问题与困难

给定一张二维目标图像 `I`，寻找一个 12 维参数向量 `p`，使得二维一般二次多项式映射

```
x_{n+1} = p0 + p1·x + p2·x² + p3·xy + p4·y + p5·y²
y_{n+1} = p6 + p7·x + p8·x² + p9·xy + p10·y + p11·y²
```

迭代百万次后落点的空间频率（经验测度）在视觉上逼近 `I`。该问题在两层面
都极其病态：

1. **结构不稳定**：参数空间几乎处处布满分岔边界，无穷小扰动可让吸引子
   突然崩溃为定点 / 极限环或飞向无穷。损失景观非凸、非可微、几乎处处
   零梯度——梯度下降彻底失效。
2. **几何盲点**：MSE / SSIM / 直方图相交等逐像素度量在两个分布不重叠时
   一律给出最大惩罚，无法为优化器提供"往哪边走"的信号。

所以必须使用：**几何感知的最优传输代价 + 无导数的进化策略 + 廉价的预拒绝
机制**。

---

## 2. 算法架构

整个管线由 6 个组件构成，全部在 `main.py` 中：

### (1) 12 参数二次映射 — `quadratic_map`, `jacobian`

向量化实现：单次 Python 调用同时推进整个 CMA-ES 种群，是性能的关键。

### (2) 四层快速拒绝惩罚系统 — `transient_test`

在 `n_test ≈ 600` 步的瞬态窗口内并行地评估全种群，任何一层触发即对该
个体打上 `MAX_PENALTY` 并跳过昂贵的光栅化阶段。

| Tier | 目标 | 检测 |
|------|------|------|
| 1 | 数值发散 | `‖x‖, ‖y‖ > 1e6` 或非有限 |
| 2 | 定点坍缩 | `‖x_{n+1} - x_n‖ < 1e-6` |
| 3 | 低周期极限环 | 64 步环形缓冲扫描，匹配阈值 1e-5 |
| 4 | 负 Lyapunov 指数 | 同步推进切向量并每步重归一化，最终 λ < 0 即拒绝 |

经验上对随机参数能预拒绝 ~90% 的样本，把整体计算量降一个数量级。

### (3) 经验测度光栅化 — `simulate_orbit`, `normalise_points`, `histogram_from_points`

只对幸存个体执行长轨道迭代 (`n_iter`)，存为 `(n_iter, lam, 2)`，再线性
缩放到 `[0, size]²` 像素框中并累加进直方图。`render_attractor` 在最后用
对数拉伸生成高对比度的可视化。

### (4) 切片 Wasserstein 距离 — `sliced_wasserstein`

精确二维 Wasserstein 是 O(n³ log n) 的线性规划，单次评估都吃不消。本实现
按论文用 SWD 近似：

1. 在单位圆上抽样 `K = 48` 个方向；
2. 把生成点云与目标点云分别投影到每个方向；
3. 各自排序，逐元素相减取绝对值平均（一维最优传输的闭式解）；
4. 对所有方向取均值。

复杂度退化为 O(K · n log n)，且在两个分布完全不重叠时仍给出连续、单调的
"位移代价"——这是优化能 work 的本质原因。目标点云只在初始化时由
`sample_target_points` 按图像强度采一次。

### (5) 自实现 CMA-ES — `CMAES`

完整的 Hansen 公式：rank-μ + rank-1 协方差更新、共轭进化路径 `pc`、
σ-进化路径 `ps`、CSA 步长适应、周期性特征分解。协方差矩阵会自动旋转并
拉伸成与混沌脊对齐的椭球，沿狭长非分离脊行进时远胜 GA / PSO。

### (6) 边界感知突变策略

CMA-ES 在分岔悬崖附近会因为整代全军覆没而失去差异梯度。两项干预：

- **动态步长阻尼**：单代拒绝率 > 70% 时强行 `σ ← σ/2`，覆盖 CSA。
- **混沌噪声注入** (`_logistic_block`)：以小概率把高斯样本块替换成
  Logistic 映射 (r=4) 输出，注入重尾相关扰动以"隧穿"非混沌死区，
  在分裂的稳定岛之间迁移。

---

## 3. 主循环 — `optimise`

```
for gen in 1..max_gens:
    arx = es.ask(inject_chaos = bernoulli(p))     # 采样
    fitness, rej_rate = evaluate_population(...)  # 拒绝 + 光栅化 + SWD
    if rej_rate > 0.70:                           # 边界保护
        es.sigma *= 0.5
    es.tell(arx, fitness)                         # 协方差/步长更新
```

`evaluate_population` 自动跳过被拒绝个体，只对幸存的子种群跑 `simulate_orbit`
和 SWD，把廉价拒绝省下的算力直接转化为更大的 `n_iter` / `pop`。

---

## 4. 文件结构

```
main.py          # 完整算法（~ 450 行）
pyproject.toml   # 依赖 numpy + pillow
```

`main.py` 关键符号一览：

| 符号 | 作用 |
|------|------|
| `Config` | 所有超参数（pop, n_iter, n_proj, σ₀, 阈值…） |
| `quadratic_map` / `jacobian` | 向量化的映射与雅可比 |
| `transient_test` | 四层快速拒绝 |
| `simulate_orbit` | 仅幸存子种群的轨道生成 |
| `sliced_wasserstein` | 切片 Wasserstein 距离 |
| `CMAES` | 自包含 CMA-ES |
| `evaluate_population` | 拒绝 + 光栅化 + SWD 流水线 |
| `optimise` | 主循环（含阻尼 / 注入） |
| `render_attractor` | 高分辨率最终渲染 |
| `initial_mean` | 以经典 Hénon 系数为先验种子 |

---

## 5. 安装与用法

依赖通过 [uv](https://docs.astral.sh/uv/) 管理，需要 Python ≥ 3.14。

```bash
# 同步依赖（首次会创建 .venv）
uv sync

# 使用合成默认目标快速体验
uv run python main.py --gens 30 --pop 16 --n-iter 20000 --size 64

# 用一张自己的灰度图作为目标（原始 2D quadratic baseline）
uv run python main.py path/to/target.png \
    --gens 80 --pop 24 --size 96 --n-iter 60000 \
    --out result.png --seed 42

# 用新的 3D latent poly+sin 模型 + multiscale SWD
uv run python main.py path/to/target.png \
    --model poly_sin3d --loss-mode multiscale \
    --size 128 --n-iter 180000 --gens 200 \
    --out result_poly_sin3d.png --seed 42
```

命令行参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `target` (位置) | 合成图样 | 灰度目标图像路径，深色 = 高密度 |
| `--out` | `best_attractor.png` | 输出路径（target \| attractor 并排） |
| `--gens` | 80 | CMA-ES 代数 |
| `--pop` | 24 / 32 | 种群大小 λ（`poly_sin3d` 默认更大） |
| `--size` | 96 | 输出方形分辨率 |
| `--n-iter` | 60000 | 单次光栅化的轨道长度 |
| `--model` | `quadratic2d` | 动力系统族：原始 2D quadratic 或实验性 `poly_sin3d` |
| `--loss-mode` | `single` | 单尺度 SWD 或 coarse-to-fine 的 `multiscale` |
| `--seed` | 42 | NumPy 随机种子 |

---

## 6. 性能与调参建议

- 大部分时间花在 `simulate_orbit` 的 Python 步循环上。把 `pop` 调大几乎
  免费——所有个体在同一次循环里向量化推进。
- `n_iter` 决定经验测度的收敛精度。低分辨率快速实验用 1–2 万；要出图
  前升到 10 万以上，并用 `render_attractor` 的 `n_iter * 4` 自动倍率。
- `n_proj`（SWD 投影数）50–100 之间方差已经足够低，再加几乎没收益。
- 若初期长时间 100% 拒绝：调小 `sigma0`（默认 0.15）或换 `initial_mean`。
- 若困在局部最优：把 `chaotic_injection_prob` 提到 0.2–0.3 强迫探索。

---

## 7. 算法 ↔ 代码索引

| 论文中的概念 | `main.py` 中的实现 |
|--------------|-------------------|
| 二维一般二次多项式映射 | `quadratic_map`, `jacobian` |
| Birkhoff 遍历定理 / 经验测度 | `simulate_orbit` + `histogram_from_points` |
| 数值发散检测 (Tier 1) | `transient_test` 中 `bad` 掩码 |
| 定点坍缩 (Tier 2) | `transient_test` 中 `fp` |
| 低周期循环 (Tier 3) | `transient_test` 中环形缓冲 `hist_x/hist_y` |
| Lyapunov 指数 (Tier 4) | `transient_test` 中 `lyap_sum` 切向量传播 |
| Sliced Wasserstein | `sliced_wasserstein` |
| CMA-ES rank-μ/rank-1 | `CMAES.tell` |
| 动态步长阻尼 | `optimise` 中 `if rej_rate > threshold` |
| 混沌噪声注入 | `CMAES._logistic_block` + `ask(inject_chaos=…)` |
