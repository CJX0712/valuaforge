# ValuaForge

**Data valuation for training sets — Data Shapley / 数据点估值。**

给训练集中的**每一个数据点**定价：它对最终模型性能贡献了多少。
本仓做的是**数据点归因（sample-level）**，不是特征归因（feature attribution）——
对象是**行**，不是列。

> 作者：**晨星** · 许可证：MIT · 阶段：算法层已实现，**性能 DoD 未达标（C 级）**

---

## 🚨 状态：旗舰 ValuaFuse 未达标，本仓按 C 级阻断交付

**诚实结论先行：核心算法（双锚点金标准、26 条不变量、CI 硬门禁）已落地并可复现，
但旗舰估值器 ValuaFuse 在所有可精确验证的规模上都不占优。**

| 口径 | 最强基线 | ValuaFuse | 比值 | 门槛 0.75 |
|---|---|---|---|---|
| A 等评估预算 | β-Shapley(1,4) 0.7470 | 0.8638 | **1.1563** | ❌ |
| B 等排列数 | β-Shapley(1,4) 0.7683 | 0.8699 | **1.1323** | ❌ |
| n=8 / 10 / 12（各 3 seeds） | — | — | **1.2561 / 1.0599 / 1.0359** | ❌ 全部 |

> **唯一例外是 n=200（比值 0.53/0.50），而 n=200 没有精确真值**（参考噪声 0.46），
> 该结果**仅有比值可信**，不构成达标证明。

**已定位的真实失败机制**（非猜测，脚本 `repro/audit_scale_and_budget.py`）：

> **方差缩减能降低"已有全保真样本"的方差，但造不出新的全保真信息。**
> 在全保真评估本身廉价时（L-BFGS，n≤12），把一半预算分给廉价代理
> 直接损失一半全保真信息量，CV 做得再好也追不回来。

**另有第二条、层级更深的根因**（`docs/algorithm_spec.md` §4.7）：
预算是 `utility.evaluate` 的**调用次数**，而代理与目标**各记 1 单位**——
预算货币 measure 不出"便宜"。于是在 KNN 轨道上代理根本不省钱（`c = 1.00`），
`examples/run_demo.py` 曾报出的 PASS 0.4351 是**记账假象，已作废**。
该 demo 现在显式打印 `VOID` 并 **exit 1**。

**因此本仓不宣称可用作数据估值系统。** 它宣称的是：
一套可复现的**估值算法实验基座**，外加一个**被诚实记录为失败**的旗舰。

| 状态 | 内容 |
|------|------|
| ✅ 可用 | `python -m cli info` / `seed` / `methods` / `errors` / `benchmark`、`examples/run_demo.py`（exit 1 是**正确**行为）、确定性 RNG 派生、22 个错误码、112 条测试、`repro/` 内 V1–V15 标定脚本与日志存档 |
| ⚠️ 有条件 | `valuation/` 全套算法可运行可验证（双锚点金标准下逐位一致），**但 ValuaFuse 不优于最强基线** |
| ❌ 不可用 | 把 ValuaFuse 当作"更好的数据估值器"使用——它比 β-Shapley(1,4) 差 4%~26% |

> 骨架阶段最忌讳的就是"看起来像个完整项目"，算法阶段同理：
> **一个能跑通、测试全绿、demo exit 0，但性能不达标的系统，比崩掉的系统更危险。**
> 这也是 `run_demo.py` 宁可 exit 1 也不给绿灯的原因。

---

## 这个仓解决什么问题

给定训练集 `D`、学习算法 `A`、验证集 `V`，效用函数

```
v(S) = Perf( A(S), V )        S ⊆ {1..n}
```

为每个训练点 `i` 赋一个实数值 `φ_i`。**预算是 utility 的评估次数**——这是唯一的货币，
由 oracle 真实计数，**禁止按公式估算**（估算与实际不符属作弊）。

支持的语义谱系（全部**手写纯 numpy**，见下方"为什么不用现成库"）：

| 语义 | 出处 |
|------|------|
| Data Shapley | Ghorbani & Zou, ICML 2019 |
| Beta Shapley | Kwon & Zou, ICML 2022 |
| Data Banzhaf | Wang & Jia, ICML 2023 |
| KNN-Shapley（闭式解） | Jia et al., ICML 2019 |
| Influence Function | Koh & Liang, ICML 2017 |
| Data-OOB | Kwon & Zou, ICML 2023 |
| **ValuaFuse**（旗舰：控制变量 + 逆方差加权融合） | 本仓 |

---

## 快速开始

```bash
python -m pip install -r requirements.txt

python -m cli info                       # 看配置与 tier 可用性
python -m cli methods                    # 看注册的方法
python -m cli errors                     # 看全部错误码
python -m cli seed "val/tmc/digits/7"    # 派生一条确定性随机流
python -m cli benchmark                  # 跑真实基准（退出码即 DoD 判定）

python examples/run_demo.py              # 端到端演示 —— **故意 exit 1**（见上文 I27）

make check                               # lint + format-check + test（与 CI 同序）
```

> **`run_demo.py` 退出码 1 是正确行为，不是坏了。** 它在第 5 步打印
> `VOID`：该演示轨道上 surrogate/target 成本比 `c = 2.77×`，
> 旗舰赖以成立的「代理更便宜」前提不成立，故其比值不作数。

> **Windows 用户**：若 pip 报 `Missing dependencies for SOCKS support` 或 502，
> 加 `--proxy "" --index-url https://pypi.org/simple`。pip 26 会从 Windows 注册表
> 取 SOCKS 源，清 shell 环境变量不够（踩坑库 §D）。

---

## 架构：单向无环

```
cli.py ──► pipeline ──► { data , valuation , eval } ──► core
                             ▲                            ▲
                             └──── 只允许 import core 类型 ───┘
```

- `core/` **只放基础设施**：`types` / `errors` / `config` / `interfaces` / `seed` / `registry`。
  **算法不进 `core/`**，旗舰 `valuafuse.py` 也在 `valuation/` 下，不开例外。
- `core` **不得** import 任何上层包。这条由 `tests/test_contracts.py::test_a1_*`
  用 **AST 静态扫描**强制：往 `core/` 里"顺手挪个算法"会被 CI 直接拦下。
- `data` / `valuation` / `eval` **彼此不横向 import**，需要对方类型时 import `core.types`。
- **平级顶层包 + 绝对导入**：`core.xxx` / `data.xxx` / `valuation.xxx`。
  `from ..core import x` 会 ImportError（`core` 是**兄弟**不是父包，踩坑库 §G）。

---

## 确定性是本仓的命门

所有随机性只有一个入口：

```python
from core.seed import SeedBank

bank = SeedBank(seed)
rng = bank.stream("val/tmc/digits/7")  # 同名两次派生逐位相等
```

子流派生**必须**用 `zlib.crc32(stream_name.encode("utf-8"))`：

```python
np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(name.encode("utf-8"))]))
```

**禁用 `hash(str)`。** CPython 按进程随机化字符串哈希（`PYTHONHASHSEED`），
用 `hash()` 派生会让测试进程与 benchmark 进程拿到不同的数，
且故障现象像数值 bug 而非播种 bug。`test_core_invariants.py` 用
**三个不同 `PYTHONHASHSEED` 起子进程**把这条钉死。

其余确定性约定：float64 / int64 强类型、集合迭代前排序、`n_jobs=1`
（Windows joblib IPC 崩溃）、钉 `numpy==2.5.3`（`Generator` 的流跨 numpy 版本
不保证一致，NEP 19 只保证 legacy `RandomState`——本仓以**钉版本**换可复现）。

---

## 双金标准锚点

| 锚点 | 原理 | 适用 | 成本 |
|------|------|------|------|
| **A 暴力穷举** | 枚举全部 `2^n` 子集 | **任意** utility，n ≤ 12 | ≤ 4096 次 |
| **B KNN 闭式解** | Jia et al. 证明 KNN utility 下闭式解 **== 精确 Shapley** | utility 必须是 KNN，**任意 n** | O(n log n) |

锚点 B 让 n=256/512 规模下的**精确误差**（不只是排序相关）成为可能。
n 超限**抛 E400，不静默近似**——近似出来的"金标准"只是又一个估计。

---

## 错误码 E100–E502

**每个码只有唯一语义**，两处共用一个码是排查误区的源头。

| 段 | 语义 | 码 |
|----|------|-----|
| **E1xx** 配置 | E100 未知配置键 · E101 schema 越界 · E102 `VALUA_*` 解析失败 · **E103 注册重名** |
| **E2xx** 数据 | E200 空数据集 · E201 形状不一致 · E202 未知数据集名 · E203 不支持 task · **E204 泄漏守卫** |
| **E3xx** 估值 | **E300 方法不可用（仅此一义）** · E301 预算耗尽 · E302 NaN/Inf · E303 utility 未定义 · E304 非 KNN utility |
| **E4xx** 评测 | **E400 金标准规模超限** · E401 金标准退化 · E402 指标未定义 · E403 曲线预算不足 |
| **E5xx** 管线 | **E500 确定性违规** · E501 写入失败 · E502 键序不稳定 |

- **E300 只有一义**："此法现在跑不了"（缺 tier / 未注册），调用方捕获它即可降级。
  注册表自相矛盾是 **E103**，在 `register()` 装饰时就抛，不等到 benchmark 中途。
- **E204** 是硬守卫：train/val 相交即抛，不只是 warning。
- **E500** 用**逐位比较**（`tobytes()`），不是 `allclose`——1e-12 的漂移也必须红。

---

## 为什么不用现成的估值库

**显式否决 `opendataval`**（实测，Phase 1）：其 METADATA 要求 `numpy<1.26`，
与本域强制的 `numpy==2.5.3` **直接冲突**，且拖入 `torch` / `pykeops`（需编译，
本机无 MSVC/cmake）。

因此：**估值算法 100% 手写纯 numpy**，sklearn 仅作 **Tier-1 交叉校验后端**，
缺失时相关行标 `skipped`，**不编数字**。

> **sklearn 装了也用不上的地方**：`random_state` 只能让**单次调用**确定，
> 无法保证"同一子集 S 的两次重训练逐位一致"。而 CRN（共同随机数）要的正是后者
> ——实测方差降幅 **77.0%** vs 朴素共享 seed 的 **2.2%**（`repro/README.md` V10/V10b）。
> 这是**接口能力差距，与装没装无关**。

---

## 证据可重跑

`repro/` 收录 V1–V11 的实测脚本与输出存档。README / 架构文档里引用的每个数字
都能在这里重跑出来——**引用了却不能重跑，等于不可证伪**。

`repro/` **不进 CI 门禁**（暴力穷举太慢），但**进仓库**，
在 `main` 分支与手动触发时跑（见 `.github/workflows/ci.yml` 的 `repro-smoke`）。

---

## 依赖

全部 **win_amd64 + cp313 有预编译 wheel，零源码编译**（Phase 1 实测）：

```
numpy==2.5.3   scipy==1.18.1   scikit-learn==1.9.1
pytest==9.1.1  pytest-cov==7.1.0  coverage==7.16.2  ruff==0.16.10
```

`requirements.txt` 是**手写的 `name==version`**，不是 `pip freeze` 的输出——
`pip freeze` 会把本机 venv 记成 `pkg @ file:///C:/...`，**装不上的 lock 不是 lock**。

---

## CI

矩阵：`ubuntu-latest` × `windows-latest` × `py3.12/3.13`，每格跑
**lint → format-check → pytest → CLI 冒烟 → 密钥扫描**。

**任何一步都没有 `|| true` / `continue-on-error`。** 一个不可能失败的检查
等于没有检查——这正是"宣称 ruff 绿、CI 实为 `|| true`"的同款病根。

---

## 文档

- `docs/architecture_spec.md` —— 架构契约（模块职责、依赖方向、错误码、门槛）
- `docs/algorithm_spec.md` —— 算法规格（公式、陷阱清单、标定）
- `docs/phase1_report.md` —— 环境就绪的实测证据
- `repro/README.md` —— V1–V11 复现口径 + 已知 bug 修复记录

---

> 作者：**晨星**
