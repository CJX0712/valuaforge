# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
所有条目作者：**晨星**。

---

## [Unreleased] — 算法层落地，旗舰性能 DoD **未达标**（C 级）

### ⚠️ 头条结论：旗舰 ValuaFuse 不占优，本项按 C 级阻断，不构成可用交付

| 口径 | 最强基线 | ValuaFuse | 比值 | 门槛 0.75 |
|---|---|---|---|---|
| A 等评估预算 | **β-Shapley(1,4)** 0.7470 | 0.8638 | 1.1563 | ❌ |
| B 等排列数 | β-Shapley(1,4) 0.7683 | 0.8699 | 1.1323 | ❌ |
| n=8 / 10 / 12 | — | — | 1.2561 / 1.0599 / 1.0359 | ❌ |

唯一例外 n=200（0.53/0.50）**无精确真值**（参考噪声 0.46），仅有比值可信。

### 🔺 本次最重要的更正：**最强基线不是规格里写死的那个**

> 规格 §4.3 初稿把 **tuned TMC-Shapley 0.9046 / 1.4101** 写作"最强基线"，
> 门槛 λ=0.75 当初就是对着这个数定的。
> **基线强度审计（`repro/audit_baseline_strength.py`）实测表明 β-Shapley(1,4) 才是最强基线**
> （0.7470 / 0.7683，两种口径下均优于 tuned TMC）。
>
> **"基线选错"会把一个方向性的失败看起来像接近达标。** 本次它就把
> 1.26× 的失败压到了"看起来只差一点点"。规格 §4.3 已全文更正，
> 并加粗警告：**门槛数值不得回头迁就结果**。

### 🔺 第二条根因：预算货币测不出"便宜"（`docs/algorithm_spec.md` §4.7）

预算是 `utility.evaluate` 的**调用次数**，代理与目标**各记 1 单位**（I21 记账一致）。
⟹ 旗舰赖以成立的「把预算挪给便宜代理」在预算口径下**恒为 1.0×**，无从体现。

实测（`gaussian_mixture`, n_train=10, seed=7, 200 次调用）：

| utility | 墙钟 µs/call | 预算单位/call |
|---|---|---|
| `knn`（demo 的 target） | 67.21 | 1 |
| `sgd`（硬编码 surrogate） | 185.94 | 1 |
| **墙钟比** | **2.77×** | **1.00×** |

`examples/run_demo.py` 曾报 **PASS 0.4351**，即记账假象 —— 现已改为打印
**`VOID` 并 exit 1**。新增硬门禁 **I27**：`c ≥ 1` 时该轨道的性能声明**作废**。

### 新增

- `valuation/`（7 模块，~2600 行）：utility oracle（KNN / SGD / Logreg）、
  暴力枚举锚点 A、KNN-Shapley 闭式锚点 B、TMC（72 组调参网格全落盘）、
  β-Shapley、**ValuaFuse（F1/F2/F3 三轴可独立开关，消融为交付物）**、LOO、随机基线
- `data/` 2 个 DGP、`eval/` 指标与报告、`pipeline/benchmark.py` 真实基准网格
- `repro/` 8 个标定/审计脚本 + `_logs/` 输出存档
- `docs/model_card.md` —— 含 7 条实测限制与伦理边界

### 修复（本轮）

- **交叉拟合切错轴**：先聚合成长度 n 的向量再用排列下标切片 ⟹ 第二折是**空切片**
  ⟹ `Var(dY)=0` ⟹ β̂ 被系统性减半（−0.139 而非 −0.278）。改为堆 M×N 逐排列矩阵、
  在排列维度切折。修正后 β̂ 全部 9 组配置一致为 **+0.43 ~ +0.49**
- **in-sample β 污染 n=200 结论**：样本内拟合使 β→0.9996，φ̂ = X̄ − Ȳ ≈ 噪声之差，
  L2Rel=1.0001。`calibrate_n200.py` 的 0.5315/0.5007 **作废**，
  隔离实验（in-sample β=0.9996 / cross-fit β=0.8411，比率 1.1883）已写入 README
- **β 权重漏 `−lgamma(x+y)`**：`sum(p)=24.0` 而非 1.0 —— 不崩不报错，只静默削弱基线
- **`scipy.stats.spearmanr` 常量输入返 `nan`**（`kendalltau` 更静默）→ 主动短路 + 拦截
- **tiny-jitter 放行区**：抖动 1e-15~1e-3 恒 `ρ=0.5774` 零警告 ⟹ std 幅度阈值被证伪，
  改用 `n_distinct(φ) ≥ max(3,⌈n/2⌉)`
- **`np.nanmean` 丢弃 2/5 样本仍给正常数** → 禁用，作废值剔除不填 0
- **DoD 取 `max` 挑最差基线** → 改 `min` + 回归测试
- **错误码重名静默遮蔽**（E103 被 E503 遮蔽，导入期零信号）→ 删重复 + `_discover_registry` 同名检测
- **`.gitattributes` 一行多 pattern**：只有最后一个 pattern 能带属性 ⟹ 一行一 pattern
- **`Config(methods=())` 误抛 E101** → 只对 `seeds`/`datasets` 要求非空
- **守卫反噬**：`NaN > 1e-14` 为 False ⟹ NaN 被翻译成「β=0 → CV 已关闭」⟹ 改 `isfinite` 前置

### 🔁 补充（n=200 cross-fit 重跑 · 2026-10-05）

`repro/recalibrate_n200.py` 用**折外 cross-fit** 重跑 n=200（修正 `calibrate_n200.py` 的
in-sample β 污染），预算 42210 等价单位、100 排列、参考 SGDx30@400：

| DGP | 估计器 | 比值 vs plain | 判定 |
|---|---|---|---|
| DGP-1 (n=200,d=5) | ValuaFuse form A（2-batch） | 0.7567 | ❌ FAIL |
| DGP-1 | ValuaFuse **form B（per-perm cross-fit）** | **0.5451** | ✅ PASS |
| DGP-4 (n=200,d=22) | ValuaFuse form A（2-batch） | 0.7704 | ❌ FAIL |
| DGP-4 | ValuaFuse **form B（per-perm cross-fit）** | **0.5815** | ✅ PASS |

**结论（非推翻 C 级阻断）**：form B 在 n=200 确实通过 DoD，证实旧 n=200 数字作废仅因
样本内 β；但 form A 仍败、小 N 精确区仍败、n=200 无精确真值 ⟹ **C 级阻断维持**。
L10（n=200 口径不干净）**部分解封**：form B 路径可引用，form A 与小 N 仍禁引。
详见 `BLOCKED.md` §2.4.1、`docs/model_card.md` L10。`recalibrate_n200.py` 与
`repro/_logs/recalibrate_n200.txt` 已进仓库。

---

## [0.1.0] — Phase 2：骨架与确定性底座

本阶段**不含任何估值算法**。落地的是承载算法的底座：类型契约、错误码体系、
配置、随机性入口、注册表，以及工程收口。

### 新增 · core/（基础设施，算法不进此处）

- **`core/seed.py`** — `SeedBank`，全仓**唯一**随机性入口。
  子流派生用 `zlib.crc32(name.encode("utf-8"))`，**禁用 `hash(str)`**
  （受 `PYTHONHASHSEED` 随机化，跨进程不稳定）。提供 `set_all` / `stream` /
  `entropy_for` / `fingerprint`，以及 `data_stream` / `val_stream` / `eval_stream`
  三个命名空间化的流构造器。
- **`core/errors.py`** — E100–E502 错误码体系，**每码唯一语义**，
  每个类带触发条件 docstring。含 `ensure_no_leakage`（E204 泄漏守卫）、
  `ensure_gold_size`（E400 金标准规模上限）、`ensure_deterministic`（E500 逐位比较）。
  导入期自检：重名类或重码直接 `RuntimeError`，不留到运行期。
- **`core/types.py`** — `Dataset` / `Budget` / `UtilitySpec` / `ValuationResult` /
  `BenchmarkRow` / `EvalReport`。构造即校验：dtype 归一（float64/int64）、
  形状一致、train/val 不相交。**字段顺序是契约**，新增字段只能追加带默认值。
- **`core/config.py`** — 默认值 ← `VALUA_*` 环境变量 ← 显式覆盖。
  声明式 schema 校验，越界抛 **E101**，**不做静默强转**。
- **`core/interfaces.py`** — `UtilityFn` / `ValuationMethod` / `GoldStandard` /
  `DGP` / `Evaluator` 的 Protocol 契约。预算货币 = `evaluate` 的真实调用次数。
- **`core/registry.py`** — 方法注册表（名字 → 构造器）+ tier 探测。
  重名注册抛 **E103**（注册期硬契约），不可用抛 **E300**。

### 新增 · 工程基座

- `pyproject.toml` —— ruff（line-length 96，钉 0.16.10）+ pytest（`pythonpath = ["."]`）。
- `.github/workflows/ci.yml` —— ubuntu/windows × py3.12/3.13 矩阵；
  lint → format-check → pytest → CLI 冒烟 → 密钥扫描。**无 `|| true`**。
- `cli.py` —— `info` / `seed` / `methods` / `errors` / `benchmark` 五个子命令；
  打印前强制 UTF-8 控制台（Windows GBK 下 `✅` 会崩）。
- `requirements.txt`（手写 `name==version`，**禁 `file:///` 本机路径**）、
  `Makefile`、`LICENSE`（MIT）、`.gitignore`。
- `repro/` —— V1–V11 实测脚本与输出存档，**进仓库**（引用的数字必须能重跑）。
- `docs/` —— 架构规格书、算法规格书、Phase 1 环境报告。

### 新增 · 测试

- `tests/test_core_invariants.py` —— SeedBank 三重确定性（含**跨进程**、
  三个不同 `PYTHONHASHSEED`）、registry 往返、config 越界抛 E101、
  错误码目录全覆盖。
- `tests/test_contracts.py` —— **A1 架构护栏**：AST 静态扫描，
  `core/` 不得 import `valuation`/`eval`/`pipeline`/`data`/`cli`，
  不得用相对导入，`core/` 下不得出现算法形状的模块名。
  已用**反向对照**验证：注入违规文件后该测试确实变红。
- `tests/test_axioms.py` —— Shapley 公理不变量 I1–I12；
  新增 **I10b** 独立钉住 Beta 权重的**归一化机制**（对数空间比 `scipy.special.betaln`），
  而不只查 `p.sum() == 1` 这个后果。

### 修复

- **Beta 权重漏 `−lgamma(x+y)`** —— `B(x,y) = Γ(x)Γ(y)/Γ(x+y)` 只减了前两项，
  导致权重和 `24.0` 而非 `1.0`。已修，并由 I10（查后果）+ I10b（查机制）双重守护。
- **错误码重名静默覆盖** —— 同名类定义会重新绑定名字，
  使先定义的错误码**不可达且无导入期信号**。`_discover_registry` 增加同名检测。
- **`Config(methods=())` 误抛 E101** —— 空元组本应表示"用全部可用方法"，
  不该按"空数据集列表"处理。

### 已知限制

- `valuation` / `data` / `eval` / `pipeline` **尚未实现**，CLI 的 `benchmark`
  与 `methods` 目前只打印解析后的配置。这是 Phase 2 后续阶段的范围。
- `repro/` 的 N=200 标定（`calibrate_n200.py`）为慢脚本，不进 CI 门禁。
