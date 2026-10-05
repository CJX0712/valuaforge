# ValuaForge 架构规格书（architecture_spec.md）

> 域：**数据估值 Data Valuation（Data Shapley / 数据点归因）**
> 作者：**晨星**
> 阶段：Phase 1（架构 + 选型 + 轮子先验 + 环境就绪）——**本阶段不含业务算法实现**
> 状态：环境已就绪（实测见 §2），架构已冻结，等 Phase 2 算法规格书（苏推衍）后落实现代码

---

## 0. 结论摘要

| 项 | 结论 |
|----|------|
| 域是否已被覆盖 | **未覆盖**。已交付 36 系统中无数据估值/Data Shapley 系统；AttribForge 是**特征归因**（feature attribution），本系统是**数据点归因**（sample-level valuation），对象不同（列 vs 行），不构成近邻子域 |
| Tier-0 依赖 wheel | **7/7 全部有 win_amd64 + cp313 预编译 wheel**（实测，§2.1） |
| venv | `C:/Users/Administrator/.workbuddy/binaries/python/envs/valuaforge`（新建成功，`python -m venv` 本次未被沙箱拦） |
| 外部研究依赖 | **显式拒绝 `opendataval`**（实测：要求 `numpy<1.26`，与本域强制的 `numpy==2.5.3` 直接冲突，且拖入 `torch~=2.1.0`/`torchvision`/`transformers~=4.35`/`pykeops`（需编译，本机无 MSVC））。算法**全部手写纯 numpy** |
| 旗舰 | **ValuaFuse**——预算约束下的**控制变量 + 逆方差加权多源融合**（§11.4） |
| 评测口径 | **双金标准锚点**（§11）：锚点 A 暴力穷举（任意 utility，n≤12）；锚点 B KNN utility 闭式解（任意 n，KNN-Shapley 即精确 Shapley） |
| 门禁主轴 | **等 utility 评估预算下**，估值向量与精确 Shapley 的排序相关 / 相对 L2 误差 |

---

## 1. 域定义与问题形式化

### 1.1 问题

给定训练集 `D = {(x_i, y_i)}_{i=1..n}`、学习算法 `A`、验证集 `V`、效用函数

```
v(S) = Perf( A(S), V )        S ⊆ {1..n}
```

为**每个训练点 i** 赋一个实数值 `φ_i`，度量它对最终模型性能的贡献。

### 1.2 估值目标（本系统实现的四类语义）

| 语义 | 定义 | 实现模块 | 备注 |
|------|------|----------|------|
| **Data Shapley** | `φ_i = Σ_{S⊆D\{i}} [\|S\|!(n−\|S\|−1)!/n!]·(v(S∪i)−v(S))` | `valuation/shapley_exact.py`（金标准）、`valuation/tmc.py`（近似） | Ghorbani & Zou 2019 |
| **Beta Shapley** | 用 `Beta(α,β)` 权重替代均匀 `1/n` 的联盟大小权重 | `valuation/beta.py` | Kwon & Zou 2022；α=β=1 退化为 Data Shapley |
| **Data Banzhaf** | `β_i = E_{S⊆D\{i}}[v(S∪i)−v(S)]`（均匀随机子集，无 1/n 权重） | `valuation/banzhaf.py` | Wang & Jia 2023；对噪声效用更鲁棒 |
| **KNN-Shapley** | KNN utility 下 Shapley 的**闭式解** | `valuation/knn_shapley.py` | Jia et al. 2019；O(n log n)/测试点 |
| **Influence Function** | 一阶近似 `v(D)−v(D\{i})`（iHVP） | `valuation/influence.py` | Koh & Liang 2017 |
| **Data-OOB** | 袋装集成的 out-of-bag 估计，零额外重训 | `valuation/oob.py` | Kwon & Zou 2023 |

### 1.3 与相邻域的边界（避免混淆）

- vs **特征归因（AttribForge）**：本系统归因对象是**样本（行）**，不是特征（列）。
- vs **主动学习（ActiveForge）**：主动学习选**未标注**点，本系统给**已标注**训练点定价。
- vs **核心集（coreset）**：核心集是子集选择，本系统输出全量逐点标量值（可再用于子集选择）。

---

## 2. 技术选型与轮子先验实测

### 2.1 轮子先验（实测输出，非推断）

命令（每个包单独跑，`pip index versions` 一次只收一个参数）：

```
C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe -m pip index versions <pkg> --proxy "" --index-url https://pypi.org/simple
```

真实平台约束（`pip index` 只证明"有版本"，不证明"有 win_amd64/cp313 wheel"，故补一轮平台级探测）：

```
python.exe -m pip download --no-deps --only-binary=:all: \
  --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 -d . "<pkg>==<ver>"
```

| 包 | 锁定版本 | index 最新 | win_amd64+cp313 wheel 实测 | 状态 |
|----|---------|-----------|---------------------------|------|
| numpy | **2.5.3** | 2.5.3 | `numpy-2.5.3-cp313-cp313-win_amd64.whl` (12.6 MB) | ✅ |
| scikit-learn | **1.9.1** | 1.9.1 | `scikit_learn-1.9.1-cp313-cp313-win_amd64.whl` (8.2 MB) | ✅ |
| scipy | **1.18.1** | 1.18.1 | `scipy-1.18.1-cp313-cp313-win_amd64.whl` (36.6 MB) | ✅ |
| pytest | **9.1.1** | 9.1.1 | `pytest-9.1.1-py3-none-any.whl` (386 kB) | ✅ |
| pytest-cov | **7.1.0** | 7.1.0 | `pytest_cov-7.1.0-py3-none-any.whl` (22 kB) | ✅ |
| coverage | **7.16.2** | 7.16.2 | `coverage-7.16.2-cp313-cp313-win_amd64.whl` (226 kB) | ✅ |
| ruff | **0.16.10** | 0.16.10 | `ruff-0.16.10-py3-none-win_amd64.whl` (10.6 MB) | ✅ |

**结论：7/7 全绿，零源码编译。** 本机无 GPU / 无 MSVC / 无 cmake，故**禁止引入任何需要编译或需要 torch 的依赖**。

### 2.2 显式拒绝的依赖（附实测证据）

| 候选 | 结论 | 证据 |
|------|------|------|
| `opendataval` 1.3.0 | **拒绝** | wheel 是 `py3-none-any` 可下载，但 METADATA 实测 23 条 `Requires-Dist`，其中 `numpy<1.26,>=1.22.4`（与本域强制 `numpy==2.5.3` **直接冲突**）、`torch~=2.1.0`、`torchvision~=0.16.0`、`transformers~=4.35`、`pykeops~=2.1.2`（需编译）、`geomloss~=0.2.6`。引入即破坏 numpy 锁版本并触发 2GB+ 下载与编译失败 |
| `xgboost` / `lightgbm` | **不需要** | 本域 utility 模型用 LR / KNN / Ridge 即可，树模型不是估值质量的关键；如需再单独先验 wheel |
| `shap` | **不需要** | `shap` 是**特征**归因库，不是数据点估值；本域不复用 |

### 2.3 分层策略

- **Tier-0（必需，纯 numpy/scipy）**：全部估值算法、全部 DGP、全部指标、金标准、管线。无 sklearn 也能端到端跑通。
- **Tier-1（可选，sklearn 1.9.1）**：utility 模型（`LogisticRegression` / `KNeighborsClassifier` / `BaggingClassifier`）、真实数据集（`load_digits` / `load_breast_cancer` / `load_wine`，均**本地内置无需联网**）、指标交叉校验。**缺失时降级到 Tier-0 手写实现，功能不减。**

> 踩坑库 §G：包装基类 `available()` 必须走**类级惰性工厂** `_factory = staticmethod(_import_fn)`，不能读 `self._estimator_cls` 实例属性，否则所有算法"不可用"。

### 2.4 计划依赖的**具体 API** 实测（venv 内真跑，非推断）

import 通过 ≠ 要用的 API 还在。以下为实际调用结果：

| 检查项 | 实测结果 | 状态 |
|--------|---------|------|
| `np.trapezoid` | 存在 | ✅ |
| `np.trapz` | 已移除 | ✅（按预期，代码须用 `trapezoid`） |
| `np.float_` | 已移除 | ✅（须用 `np.float64`） |
| 确定性原语 `default_rng(SeedSequence([seed, crc32(stream)]))` | 同 stream 两次派生**逐位相等**；不同 stream **不相等** | ✅ |
| `sklearn.datasets.load_digits` | 离线可用 `(1797, 64)`，无需联网 | ✅ |
| `LogisticRegression(solver="lbfgs")` 多分类 | 可用，10 类 | ✅（sklearn 1.9 已移除 `multi_class=`，`liblinear` 不支持多分类 ⇒ **必须 `lbfgs`**） |
| `KNeighborsClassifier(5)` | 可用，val acc 0.895 | ✅ |
| `BaggingClassifier(oob_score=True)` | `oob_score_=0.91`，`oob_decision_function_` 存在 | ✅（Data-OOB 可行） |
| `scipy.stats.spearmanr` **常量输入** | **返回 `nan`** | ⚠️ 必须自守 |
| `scipy.stats.kendalltau` **常量输入** | **返回 `nan`** | ⚠️ 必须自守 |

> ⚠️ **实测修正**：scipy 对常量输入返回的是 **`nan`**，既不是踩坑库记载的 `±1`，也不是我们期望的 `0.0`。
> 因此 `eval/metrics.py` **不可直接透传 scipy 结果**：常量输入必须走短路返回 `0.0`，且任何 `nan` 都必须在指标层被显式拦截（否则 `nan` 会污染 mean±std 聚合，使整列指标静默失效）。
> 平局（tie）必须取**平均秩**——自实现路径不可依赖 `scipy` 默认行为。

**环境实测版本**：venv 与 base 的 `sys.version` 均为 **3.13.14**（托管目录名 `3.13.12` 仅为目录标签，实际解释器为 3.13.14，仍属 cp313 ABI，与 §2.1 的 `cp313` wheel 探测一致，无 ABI 风险）。

---

## 3. 目录骨架与依赖方向

```
valuaforge/                      # 交付仓库根
├── valuaforge/                  # 包根
│   ├── core/                    # 无业务依赖的最底层
│   │   ├── types.py             # dataclass: Dataset / ValuationResult / Budget / EvalReport / BenchmarkRow
│   │   ├── errors.py            # ValuaError + E100~E500
│   │   ├── config.py            # Config dataclass + ENV_VALUA_* 覆盖 + schema 校验
│   │   ├── interfaces.py        # Protocol: UtilityFn / Valuator / DatasetSource / GoldStandard / Evaluator
│   │   ├── seed.py              # SeedBank：唯一随机性入口（确定性核心）
│   │   └── registry.py          # available_*() 探测与降级
│   ├── data/
│   │   ├── dgp.py               # 合成 DGP（固定 seed 可复现）+ 质量结构注入
│   │   ├── loader.py            # 真实数据载入（sklearn 内置集，离线）
│   │   └── corrupt.py           # 标签噪声 / 冗余 / 稀有类注入（产出 corruption mask 真值）
│   ├── valuation/
│   │   ├── base.py              # ValuatorBase（类级惰性工厂 + available()）
│   │   ├── utility.py           # UtilityFn 实现：LR / KNN / Ridge（Tier-0 手写 + Tier-1 sklearn）
│   │   ├── shapley_exact.py     # 2^n 暴力穷举金标准（锚点 A）
│   │   ├── knn_shapley.py       # KNN 闭式解（锚点 B，亦作方法）
│   │   ├── tmc.py               # TMC-Shapley 截断置换蒙特卡洛
│   │   ├── beta.py              # Beta Shapley
│   │   ├── banzhaf.py           # Data Banzhaf
│   │   ├── influence.py         # 影响函数（iHVP：闭式 / 共轭梯度）
│   │   ├── oob.py               # Data-OOB（袋装）
│   │   ├── valuafuse.py         # 【旗舰】控制变量 + 逆方差加权多源融合
│   │   └── registry.py          # available_valuators() / get_valuator()
│   ├── eval/
│   │   ├── gold.py              # 双锚点金标准计算 + 磁盘缓存 + n 上限守卫
│   │   ├── metrics.py           # Spearman / Kendall / 相对 L2 / top-k Jaccard（tie 与常量安全）
│   │   ├── downstream.py        # 移除曲线 AUC / 添加曲线 / 噪声检测 AUROC
│   │   └── report.py            # 跨 seed×dataset 聚合 mean±std + 显著性
│   ├── pipeline/
│   │   ├── pipeline.py          # ValuaPipeline.run()
│   │   └── benchmark.py         # benchmark() → benchmark.json
│   └── cli.py                   # argparse
├── examples/run_demo.py
├── tests/
├── docs/architecture_spec.md    # 本文档
├── .github/workflows/ci.yml
├── pyproject.toml               # 【先定 pyproject 再 ruff format】
├── requirements.txt
├── requirements.lock.txt
├── Dockerfile
├── Makefile
├── README.md / CHANGELOG.md / LICENSE / .gitignore
```

**依赖方向（单向无环，强制）**

```
cli.py ──► pipeline ──► { data , valuation , eval } ──► core
                              ▲                          ▲
                              └──── valuation 可 import core/data 类型 ────┘
```

- `core` **不得** import 任何上层包。
- `data` / `valuation` / `eval` **彼此不得横向 import**（需要对方类型时 import `core.types`）。
- `pipeline` 是唯一允许同时 import 三者的层。
- **统一绝对导入** `core.xxx` / `data.xxx` / `valuation.xxx` / `eval.xxx`（踩坑库 §G：`..core` 相对导入在平级顶层包会 ImportError）。

---

## 4. 模块职责表

| 模块 | 职责 | 输入 | 输出 | 不得做 |
|------|------|------|------|--------|
| `core.types` | 定义跨层数据契约（dataclass，float64 约定） | — | 类型 | 不含任何计算逻辑 |
| `core.errors` | 统一错误码，携带 `code` + `context` | — | 异常 | 不吞异常 |
| `core.config` | 默认值 + `ENV_VALUA_*` 覆盖 + schema 校验 | dict/env | `Config` | 不做 IO |
| `core.interfaces` | 声明 Protocol（鸭子契约） | — | Protocol | 无实现 |
| `core.seed` | **唯一**随机性来源，派生具名子流 | seed, stream名 | `np.random.Generator` | 不允许别处调 `np.random.*` |
| `core.registry` | Tier 探测 + 降级决策 | — | bool / 工厂 | 不做重量级 import 副作用 |
| `data.dgp` | 合成数据 + **质量结构真值** | seed, 难度旋钮 | `Dataset` | 不做归一化 fit 在测试集上 |
| `data.corrupt` | 注入噪声/冗余/稀有类，返回 mask | Dataset, frac | Dataset + mask | 不改 X（只改 y 或复制行） |
| `data.loader` | 载入 sklearn 内置集（离线） | name, n, seed | `Dataset` | 不联网下载 |
| `valuation.utility` | utility oracle，**真实计数**每次评估 | 子集索引 | float + 计数 | 不缓存跨 run（防污染） |
| `valuation.*` | 各估值算法 | Dataset, UtilityFn, Budget, rng | `ValuationResult` | 不自行造 rng |
| `valuation.valuafuse` | 旗舰融合 + 预算分配 | 同上 | `ValuationResult` | 不超预算 |
| `eval.gold` | 双锚点精确值 + 缓存 | Dataset, utility | np.ndarray | n 超限必须抛 E400，不静默近似 |
| `eval.metrics` | 排序/误差指标 | 估计值, 金标准 | float | 常量输入必须返回 0.0（不返回 ±1） |
| `eval.downstream` | 移除/添加曲线、噪声检测 | Dataset, 估计值 | float | 曲线值必须 clip 到 [0,1] |
| `eval.report` | 聚合 + 显著性 | 行列表 | `EvalReport` | 不混聚不同 task 的指标 |
| `pipeline.pipeline` | 编排 run | Config | `EvalReport` | 不含算法逻辑 |
| `pipeline.benchmark` | 基准跑 + 落盘 JSON | Config | benchmark.json | 不重排键 |
| `cli` | 参数解析 + 表打印 | argv | — | 不做业务判断 |

---

## 5. 关键接口 Protocol（`core/interfaces.py`）

```python
from typing import Protocol, Sequence, runtime_checkable
import numpy as np
from core.types import Dataset, ValuationResult, Budget, EvalReport


@runtime_checkable
class UtilityFn(Protocol):
    """效用 oracle。budget 的货币单位 = 本对象被 evaluate 的次数。"""

    @property
    def n_train(self) -> int: ...
    @property
    def n_evals(self) -> int:
        """真实累计评估次数（硬计数，禁止估算）。"""

    def evaluate(self, subset: np.ndarray) -> float:
        """subset: 升序 int64 训练索引数组 -> 效用标量（越大越好）。"""

    def remaining(self, budget: "Budget") -> int: ...


@runtime_checkable
class Valuator(Protocol):
    name: str  # 稳定标识符，进 benchmark.json

    @classmethod
    def available(cls) -> bool: ...  # 类级惰性工厂探测
    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: Budget,
        rng: np.random.Generator,  # 由 SeedBank 派生后注入，算法内禁止自建
    ) -> ValuationResult: ...


@runtime_checkable
class GoldStandard(Protocol):
    name: str
    is_exact: bool  # True=数学精确；False 仅作参照
    max_n: int | None  # None=无上限

    def values(self, data: Dataset, utility: UtilityFn) -> np.ndarray: ...


@runtime_checkable
class DatasetSource(Protocol):
    name: str

    def load(self, seed: int) -> Dataset: ...


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def run(
        self,
        data: Dataset,
        values: dict[str, np.ndarray],
        gold: np.ndarray | None,
        budget: Budget,
    ) -> EvalReport: ...
```

### `core/types.py` 关键字段

```python
@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # (n, d) float64
    y: np.ndarray  # (n,) int64(分类) / float64(回归)
    name: str
    task: str  # "classification" | "regression"
    val_idx: np.ndarray  # 验证集索引（与训练集严格不相交）
    train_idx: np.ndarray
    meta: dict  # corruption_mask / quality_group 等真值结构


@dataclass(frozen=True)
class Budget:
    max_utility_evals: int  # 硬上限
    max_wall_seconds: float | None = None


@dataclass(frozen=True)
class ValuationResult:
    values: np.ndarray  # (n_train,) float64，与 train_idx 顺序一致
    method: str
    n_utility_evals: int  # 真实计数（取自 UtilityFn.n_evals）
    wall_seconds: float
    diagnostics: dict  # tier / converged / per_source_var / beta_hat / ...
    tier: str  # "tier0" | "tier1"
```

---

## 6. 数据流图（文字版）

```
[Config]  (defaults ← ENV_VALUA_* ← CLI)
   │
   ▼
[core.seed.SeedBank(seed)]
   │  derive("data/<ds>/<seed_s>")  derive("val/<method>/<ds>/<seed_s>")  derive("eval/...")
   ▼
[data.dgp | data.loader] ──► Dataset(X, y, train_idx, val_idx, meta[corruption_mask])
   │
   ├──► [valuation.utility.make_utility(model, data)] ──► UtilityFn（带真实计数器）
   │
   ├──► [eval.gold]
   │        ├─ 锚点 A: shapley_exact  (n ≤ GOLD_MAX_N=12)  → 2^n 次 utility
   │        └─ 锚点 B: knn_shapley    (utility 必须是 KNN) → 闭式，O(n log n)
   │
   ▼
[valuation.<method>.value(data, utility, budget, rng)]  ── 每个方法分到**等额** budget
   │   tmc / beta / banzhaf / influence / oob / knn_shapley / valuafuse
   ▼
ValuationResult(values, n_utility_evals=[真实], tier, diagnostics)
   │
   ├──► [eval.metrics]    vs gold: spearman / kendall / rel-L2(最优缩放) / top-k Jaccard
   ├──► [eval.downstream] 移除曲线 AUC / 添加曲线 AUC / 噪声检测 AUROC(用 corruption_mask)
   ▼
[eval.report]  跨 dataset × method × seed 聚合 → mean±std，显著性判据见 §11.5
   ▼
[pipeline.benchmark] ──► benchmark.json（键序固定，供确定性 diff）
   ▼
[cli] 表格打印（固定宽度，中文宽度对齐）
```

**预算流向铁律**：`UtilityFn.evaluate` 每被调用一次计数 +1；`Valuator` 在开跑前记录 `n_evals` 基线，结束时 `n_utility_evals = now − base`。**不按公式估算评估次数**（踩坑库 §A：估算与实际不符属作弊）。

---

## 7. 错误码 `E100~E502`（`core/errors.py`）

**每个码只有唯一语义。** 两处共用一个码是排查误区的源头——调用方按码捕获，
语义一污染，fallback 路径就会把"注册表坏了"当成"缺 tier"静默降级。

| 段 | 语义 | 码 | 触发 |
|----|------|----|------|
| **E1xx** 配置 | E100 | 未知配置键 |
| | E101 | schema 校验失败（类型/范围） |
| | E102 | `VALUA_*` 解析失败（**值无法转成目标类型**，如 `VALUA_BUDGET=abc`） |
| | **E103** | **注册表重名**——两个**不同**对象注册到同名（`DuplicateRegistrationError(ConfigError)`）。在 `register()` 装饰时即抛，不等到 benchmark 中途。归 E1xx 而非 E5xx：它与 E100/E101/E102 同属**注册期契约**（配置与接线），不与 E5xx 的"运行期产物失败"混段 |
| **E2xx** 数据 | E200 | 空数据集 / n=0 |
| | E201 | X/y 形状不一致 |
| | E202 | 未知数据集名（**禁止静默兜底**——踩坑库 §G MiaForge 真实缺陷） |
| | E203 | 不支持的 task 类型 |
| | E204 | train_idx 与 val_idx 相交（泄漏守卫） |
| **E3xx** 估值 | E300 | 方法不可用（**仅此一义**：缺 tier / `available()` 为 False / 未注册） |
| | E301 | 预算耗尽（超出 `max_utility_evals`） |
| | E302 | 估值出现 NaN/Inf |
| | E303 | utility 未定义（子集为空 / 单类） |
| | E304 | KNN 闭式解要求 utility 为 KNN 但未满足 |
| **E4xx** 评测 | E400 | 金标准规模超限（n > GOLD_MAX_N 且走锚点 A） |
| | E401 | 金标准退化（全常量 → 相关未定义） |
| | E402 | 指标未定义（长度不一致） |
| | E403 | 下游曲线预算不足 |
| **E5xx** 管线 | E500 | 确定性违规（同 seed 二次运行非逐位一致） |
| | E501 | 输出写入失败 |
| | E502 | benchmark.json 键序不稳定 |

### 7.1 E300 与 E103 为什么必须分开

| | E300 | E103 |
|---|-----|------|
| 含义 | "此法**现在**跑不了"（缺 tier / 降级） | "注册表**本身**坏了"（同名两个对象） |
| 性质 | **可恢复**：`except E300` → 跳过该方法继续跑 | **不可恢复**：一定是编程错误 |
| 修法 | 装依赖 / 换 backend | 改接线代码 |
| 抛出时机 | 查询时（`get_constructor`） | **注册时**（`register()` 装饰） |

若合并，`except E300` 的降级路径会把"插件表损坏"当成"可选 tier 缺失"**静默吞掉**——
benchmark 照常出数，只是少了一个方法，且没人知道为什么。

> **本轮真实踩到**：曾同时存在两个 `DuplicateRegistrationError`（一个 E103、一个 E503）。
> 同名定义会**重新绑定名字**，先定义的那个类**不可达且导入期零信号**——
> `error_class("E103")` 直接 KeyError，而 `E103` 明明在 `ERROR_CATALOG` 里。
> 已在 `core/errors.py::_discover_registry` 加**同名检测**（原先只查重码，不查重名），
> 撞名/撞码一律 `RuntimeError` 立即炸。

---

## 8. 配置与 `ENV_VALUA_*` 覆盖（`core/config.py`）

优先级：**CLI > ENV_VALUA_* > 默认值**。

| 键 | 默认 | 环境变量 | 说明 |
|----|------|----------|------|
| `seed` | 7 | `VALUA_SEED` | **唯一**随机性入口 |
| `seeds` | (7, 17, 2027) | `VALUA_SEEDS`（逗号分隔） | 主门禁 ≥3 seeds |
| `datasets` | 见 §10 | `VALUA_DATASETS` | |
| `methods` | 全可用 | `VALUA_METHODS` | |
| `budget` | 4096 | `VALUA_BUDGET` | 每方法每数据集的 **utility 评估**上限 |
| `gold_max_n` | 12 | `VALUA_GOLD_MAX_N` | 锚点 A 上限，超则抛 E400 |
| `utility_model` | `"logreg"` | `VALUA_UTILITY_MODEL` | `logreg` / `knn` / `ridge` |
| `knn_k` | 5 | `VALUA_KNN_K` | |
| `beta_alpha` / `beta_beta` | 1.0 / 4.0 | `VALUA_BETA_ALPHA` / `VALUA_BETA_BETA` | α=β=1 退化为 Data Shapley |
| `influence_damping` | 1e-3 | `VALUA_INFLUENCE_DAMPING` | iHVP 阻尼 |
| `oob_n_estimators` | 64 | `VALUA_OOB_N_ESTIMATORS` | |
| `removal_steps` | 20 | `VALUA_REMOVAL_STEPS` | 移除曲线检查点数 |
| `noise_frac` | 0.15 | `VALUA_NOISE_FRAC` | 标签噪声比例 |
| `output_dir` | `results/` | `VALUA_OUTPUT_DIR` | |
| `n_jobs` | 1 | — | **Windows 强制 1**（joblib 并发 IPC 崩溃） |

schema 校验：类型不符 / 范围非法 → `E101`，**不做静默强转**。

---

## 9. 确定性方案（唯一 seed 入口）

1. **唯一入口**：`Config.seed`。所有随机性必须经 `core/seed.SeedBank`。
2. **子流派生**（关键，避免串扰）：
   ```python
   rng = SeedBank(seed).derive(stream_name)
   # 实现：np.random.default_rng(np.random.SeedSequence([seed, crc32(utf8(stream_name))]))
   ```
3. **禁止 `hash(str)`**：Python 字符串 hash 受 `PYTHONHASHSEED` 随机化，跨进程不稳定 → **必须**用 `zlib.crc32(name.encode("utf-8"))` 或 `hashlib.blake2b`。这是隐蔽的确定性杀手。
4. **流名规范**：`data/<dataset>/<seed>`、`val/<method>/<dataset>/<seed>`、`eval/<evaluator>/<dataset>/<seed>`。不同用途**不得共用**一条流。
5. **dtype 统一 float64**；索引统一 int64。
6. **`n_jobs=1`**（Windows joblib `concurrent send_bytes` 崩溃）。
7. **集合/字典迭代前必须排序**（避免 set/dict 顺序影响结果）。
8. **版本锁定**：`requirements.lock.txt` 钉 `numpy==2.5.3` / `scikit-learn==1.9.1` / `scipy==1.18.1`。踩坑库 §C：`np.random.Generator` 的流跨 numpy 版本不保证一致（NEP 19 只保证 legacy `RandomState`）——**本系统以"钉版本"而非"用 legacy API"换取可复现**，并在 CI 锁 `requirements.lock`。
9. **确定性门禁**：同 seed 连续跑两次 → `benchmark.json` **逐位一致**（`git diff --exit-code` 为空），否则抛 `E500`。
10. **不跨 run 缓存模型状态**（热启动残留破坏确定性）。

---

## 10. 合成 DGP 与难度旋钮（`data/dgp.py`）

踩坑库 §B：DGP 必须**有难度梯度**且**随机基线明显低于全量天花板**，否则全方法满分/全崩无区分度。

| DGP | 旋钮 | 甜点（待 Phase 2 扫参确认） | 产出真值结构 |
|-----|------|--------------------------|-------------|
| `gaussian_mixture` | `class_sep` | 1.0（默认 1.0，扫 0.8/1.0/1.2） | — |
| `moons` | `noise` | 0.40（踩坑库：0.18 太易） | — |
| `sparse_linear` | `n_informative`, `snr` | d=50, n_inf=10, snr=1.0 | 信息/噪声特征（→ 高/低价值点） |
| `xor_like` | `n`, `noise` | n=300 | — |
| `label_noise` | `noise_frac` | 0.15 | **corruption_mask**（噪声检测金标准） |
| `redundant` | `dup_frac` | 0.20 | **dup_mask**（复制点应≈低值） |
| `rare_class` | `rare_frac` | 0.05 | **rare_mask**（稀有点应≈高值） |
| `heterogeneous` | 以上组合 | — | 四类质量组齐全 → 值应分层 |

**泄漏守卫**（踩坑库 §B）：
- scaler / 特征选择 **只允许在 train 折 fit**，验证集只 transform。
- HPO（若有）使用**与 benchmark 数据集 seed 不相交**的独立验证实例。
- `train_idx ∩ val_idx = ∅` 断言，违反抛 `E204`。
- 合成"困难版"必须靠**共享结构 + 加噪/翻转**，不是两次独立采样（否则等于换问题）。

---

## 11. 金标准、评测口径与旗舰设计

### 11.1 锚点 A —— 暴力穷举精确 Shapley（任意 utility）

```
φ_i = Σ_{S ⊆ D\{i}} (|S|! (n−|S|−1)! / n!) · [v(S∪i) − v(S)]
```
共享 `v(S)` → 总计 `2^n` 次 utility 评估。`n ≤ 12` ⇒ ≤4096 次，每次在 ≤12 样本上训练 LR ⇒ 秒级。**n>12 抛 E400，不许静默近似。**

### 11.2 锚点 B —— KNN utility 闭式解（任意 n）

Jia et al. (2019) 证明：**当 utility 为 KNN 时，KNN-Shapley 闭式解 = 精确 Data Shapley**。故锚点 B 提供**任意 n** 的精确金标准（类似 SeqForge 用"KF log-lik vs 独立多元高斯闭式解差 1.1e-13"作黄金锚点）。

用途：在 n=256/512 规模上，仍能对 TMC / Beta / Banzhaf / ValuaFuse 做**精确误差**评估，而不只是排序相关。

### 11.3 指标（tie 与常量安全）

| 指标 | 定义 | 陷阱与修法 |
|------|------|-----------|
| Spearman ρ | 对估计值与金标准分别取秩后 Pearson | **必须先过 §11.3.1 的退化门禁**；实测 scipy 常量输入返回 `nan`（§2.4），**禁止透传** |
| Kendall τ | 一致对 − 不一致对 | **平局取平均秩**；全相同 → τ=0（不是 −1）；实测 scipy 常量输入返回 `nan` 且**完全静默无警告**，同样须短路 |
| 相对 L2（最优缩放） | `min_c \|\|φ̂ − c·φ*\|\|₂ / \|\|φ*\|\|₂` | 解决 Banzhaf/Shapley 量纲不同不可直接比 |
| Top-k Jaccard | 最高/最低 k 个点的集合交并比 | k 固定（默认 0.1n） |
| 移除曲线 AUC | 按估值升序移除，记录 acc 曲线 | **曲线值必须 `clip(·,0,1)`**（踩坑库：累计式会 >1，实测达 1.875） |
| 噪声检测 AUROC | 以 `−φ` 为分数，`corruption_mask` 为标签 | 常量估值 → AUROC=0.5，须显式标注 |

### 11.3.1 退化门禁（**tiny-jitter 陷阱，最高优先级**）

> **实测（本仓 `tests/test_contracts.py::test_i23_tiny_jitter_defeats_every_magnitude_threshold` 已锁）**：
> `x = [1]*n` 中令 `x[-1] += eps`，则

| eps | `spearmanr` | 警告数 | `std` 相对门禁 | `n_distinct` 门禁 |
|-----|-------------|-------|----------------|------------------|
| 0 | `nan` | 1 | False ✅ | False ✅ |
| 1e-15 … 1e-8 | **0.5774** | **0** | False ✅ | False ✅ |
| **1e-7 … 1e-3** | **0.5774** | **0** | **True ❌ 放行** | False ✅ |

**结论（三条，都要写进 `eval/metrics.py`）**：

1. **秩相关只依赖排序，与扰动幅度完全无关** ⇒ ρ 在 12 个数量级上恒为 `0.5774`。
   因此**任何基于 std/幅度的阈值都无法区分 1e-15 抖动与 1e-3 抖动**。
2. **绝对门禁 `std() > 0` 不够**：单点 `1e-7` 抖动即可骗过它，产出"看起来正常"的 ρ=0.58 而零警告。
   **相对门禁 `std > 1e-8·max(|φ|max, 1)` 也不够**——实测在 `eps ≥ 1e-7` 时**放行**（上表），而那些同样是退化的（只有 2 个不同取值）。
3. **唯一可靠判据是 `n_distinct`**：`n_distinct(φ) ≥ max(3, ⌈n/2⌉)`。
   实测对**所有**抖动幅度均拦截，且不误杀合法估值（连续向量 8/8、离散但合法 4/8 均通过）。

**因此 `eval/metrics.py` 的强制契约**：

```python
def assert_non_degenerate(values, *, name="valuation"):
    # ① 有限性
    # ② n_distinct >= max(3, ceil(n/2))   ← 主判据，唯一可靠
    # ③ std > 0                            ← 仅作恒定向量的廉价前置
    ...
```

- **必须在算任何秩相关之前调用**，失败即 `fail-fast`（抛 E4xx），**不得**降级为"记 0 分继续"。
- 报告里每个估值向量**必须附 `n_distinct(φ)`**；`n_distinct < n/2` 时该行的 Spearman/Kendall **一律作废**，不参与排名与均值。
- **禁用 `np.nanmean`**：实测 `[0.9, nan, nan, 0.8, 0.85]` → `nanmean` 返回 **0.85**（静默丢 2 个点仍给正常数），`mean` 返回 `nan`。指标列出现 NaN 必须**报错而非忽略**。
- 覆盖实现的门禁见 `tests/test_contracts.py::assert_non_degenerate`。

### 11.4 旗舰 ValuaFuse 设计（交 Phase 2 细化，此处给设计空间）

**目标轴（选它是因为可度量且不与成员结构性冲突）**：
> **在同一 utility 评估预算 B 下**，比任何单一蒙特卡洛估值器都更接近精确 Shapley（更低相对 L2 / 更高 ρ）。

即比较口径是**效率轴**（同预算比精度），而非"比最强成员再高 X%"——后者在融合成员含该最强方法时结构性不可达（踩坑库 §A，与 TscForge 同款）。

**三步机制**：

1. **Pilot（约 15% 预算）**：对每个 MC 源（TMC / Beta / Banzhaf）各跑少量置换采样，估计**逐源采样方差** `σ²_s`（用多次重复或批内方差）。
2. **Allocation（Neyman 分配）**：剩余预算按 `B_s ∝ σ_s · √(cost_s)` 的最优分配（最小化融合估计量总方差），预算分配**必须有总额兜底**（踩坑库 §G：预算上限契约被破的教训）。
3. **Control Variate + 逆方差聚合**：取**零成本确定性估计量**（KNN-Shapley 闭式解 或 Influence）作控制变量 `C`（`E[C]` 精确已知，因确定性）：
   ```
   φ̂_fused = Σ_s w_s · [ MC_s − β_s (C − E[C]) ],   w_s ∝ 1/σ²_s
   ```
   控制变量是**教科书级方差缩减**，其有效性可实测（β̂ 及其方差缩减比），不依赖运气。

**诚实要求**：
- 若实测 ValuaFuse 在某数据集上不敌最强单法，**必须如实报告**（不做选择性呈现）。
- 消融必须包含**被否决组件**的负结果（踩坑库 §A）。
- 若 ValuaFuse 在 KNN utility 下退化为 KNN-Shapley（结构性相等），须像 DensForge 那样**诚实声明"是推广非超越"**，并给出逐位相等证据或差异证据。

### 11.5 门禁（预设，Phase 2 与算法组共同定稿）

| 门禁 | 判据（草案） | 说明 |
|------|------------|------|
| G1 金标准保真 | 锚点 A/B 上，TMC/Beta/Banzhaf 相对 L2 随预算单调下降；ValuaFuse 在**等预算**下相对 L2 ≤ 最优单 MC 源 | 主轴 |
| G2 效率 | ValuaFuse 达到单源 X% 精度所需预算更少（≥2×） | 效率轴佐证 |
| G3 下游·移除曲线 | 按估值升序移除，曲线下面积 ≤ 随机移除基线 − Δ（Δ 预注册） | 高值点先移除掉得更快 |
| G4 下游·噪声检测 | 噪声检测 AUROC ≥ 0.8（噪声比例 0.15） | 实用价值 |
| G5 确定性 | 同 seed 二次运行 benchmark.json 逐位一致 | 硬门禁 |
| G6 不伪胜 | 在无噪声/全等值 DGP 上，ValuaFuse 相对基线增益 ≈ 0 | 防止结构泄漏 |

**统计严谨**（踩坑库 §A 最高频）：≥3 seeds 报 mean±std；"胜基线"判据 = 均值差 > ½(σ₁+σ₂)；5 seeds 重跑须不翻转。

---

## 11.4 前提门禁 I26 / I27（**由实测失败驱动，不是预防性 prudence**）

> 作者注：这两条曾被漏掉。补记于此，并作为 CI 门禁
> （`tests/test_flagship_premise.py::TestI27Documentation` 会检查本文件是否声明它们）。

### I26：F1 符号门禁必须自洽

**曾经的失败**：本项目一版报告把「β̂ 算出来是负的，所以控制变量在加噪」当作**方法的性质**写进了结论。
**那是实现 bug**：交叉拟合把 `(M, n)` 的逐排列矩阵先折叠成长度-n 向量、再用**排列索引**去切，
第二折因此为空 ⇒ `Var(dY) = 0` ⇒ β̂ 被污染。修正后 9 组 (n, seed) 的 β̂ **全为正 +0.43~+0.49**。

**门禁**：任何负 β̂ 在被叙述成"发现"之前，**必须先被报告为可能的 bug**。
一个异常结果如果"讲得通"，第一反应应是怀疑测量，而不是给它写机制。

### I27：代理必须**真的**更便宜（**当前 KNN 赛道上不成立**）

**问题**：预算口径（`utility.evaluate` 调用次数）下，目标与代理**每次都记 1 单位**。
所以旗舰的核心前提「把预算挪到廉价代理上」**在预算单位下根本不可度量** —— 它恒等于 1.0×。

**独立实测**（本机，n=60 / d=6 / val=40，300 次调用取均值）：

| 模型 | µs/call | 相对 knn |
|------|---------|---------|
| `knn`（目标） | 168.40 | 1.00× |
| `sgd`（代理） | 209.24 | **1.24× —— 更贵** |

**⇒ 在 KNN 赛道上，代理在 wall-clock 上比目标更贵。**
规格 §2.7 假设的成本比 `c ≈ 0.1~0.16` **在 KNN 目标下不成立**（该假设来自 SGD×30 vs LBFGS 的另一组配置）。

**后果（必须写进交付物）**：任何「同预算下更准 ⇒ 更省算力」的表述，
若建立在 KNN 赛道上，都是**在比较两种不同货币**。`examples/run_demo.py` 曾据这种口径打印 PASS。

**门禁**：**`c ≥ 1` 的赛道上不得产出性能声明**，只能报"等预算下的精度比较"，
并显式标注该赛道代理并不更便宜。

### 预算双字段（`BenchmarkRow`）

预算以**两种货币**报告，缺一不可：

| 字段 | 含义 | 实测（ValuaFuse, n=10, budget=1200） |
|------|------|----------------------------------|
| `n_calls_raw` | `utility.evaluate` 实际调用总次数（含代理调用） | 2376 |
| `n_calls_full_equiv` | 以"一次全保真调用"为单位的等价成本 | 1247.4 |
| `n_calls_full` / `n_calls_cheap` | 两者的拆分 | 1188 / 1188 |

单用 `raw` 把 ValuaFuse 成本**高估 1.90×**；单用 `equiv` 则**低估真实调用压力**。
主判用 `equiv`，但两值必须同时出现（`core/types.py::BenchmarkRow`）。

---


| 场景 | 预算 | 分配 |
|------|------|------|
| `examples/run_demo.py` | **≤ 60s**（硬上限） | 锚点 A（n=10, 1024 evals）≈3s；锚点 B（n=256, KNN 闭式 + 3 MC 源 @2048 evals）≈15s；移除曲线 20 检查点 ×4 方法 ≈25s；噪声检测 ≈10s；缓冲 ≈7s |
| `pytest` 全量 | ≤ 120s | 单元为主，集成用小 n |
| 完整 benchmark（5 seeds × 6 datasets × 7 methods） | ≤ 15 min | 不在 demo/CI 内 |
| 内存 | **≤ 2 GB** | 最大矩阵 n=1000 距离阵 = 8 MB；n>1000 分块 |

**护栏**：
- `UtilityFn` 超 `max_utility_evals` 抛 `E301`，不静默截断。
- 锚点 A 超 `gold_max_n=12` 抛 `E400`。
- demo 自检：墙钟 > 60s 时打印 ⚠️ 并写进 `benchmark.json` 的 `perf` 段（不隐藏）。

---

## 13. SOTA 对标

| 方法 | 出处 | 本系统 | 复用还是手写 |
|------|------|--------|-------------|
| Data Shapley | Ghorbani & Zou, **ICML 2019** | `shapley_exact.py`（金标准）+ `tmc.py`（其 TMC 近似） | **手写**（纯 numpy） |
| KNN-Shapley（闭式） | Jia et al., **ICML 2019** | `knn_shapley.py` | **手写** |
| Beta Shapley | Kwon & Zou, **ICML 2022** | `beta.py` | **手写** |
| Influence Function | Koh & Liang, **ICML 2017** | `influence.py` | **手写** |
| Data Banzhaf | Wang & Jia, **ICML 2023** | `banzhaf.py` | **手写** |
| Data-OOB | Kwon & Zou, **ICML 2023** | `oob.py` | **手写** |
| 截断置换 MC 估计器 | Castro et al. / Ghorbani TMC | `tmc.py` 估计核心 | **手写** |

**为什么全部手写**：
1. 无成熟生产级库——`opendataval` 实测要求 `numpy<1.26`（与本域 `numpy==2.5.3` 冲突）且拖入 torch/pykeops，本机无 GPU/无 MSVC 不可用（§2.2 实测证据）。
2. 已交付记录中评级 S/A 的系统（AttribForge / PTQForge / HawkesForge / SeqForge / BayesForge）**全部是"纯 numpy 手写内核 + sklearn 仅作 Tier-1 交叉校验"**这一模式，CI 稳定、确定性可控。
3. 估值算法内核都是 20~200 行线性代数，手写可逐位对齐论文公式并写不变量测试。

**Tier-1 复用的部分**：仅 utility 模型（sklearn `LogisticRegression` / `KNeighborsClassifier` / `BaggingClassifier`）与内置数据集、指标交叉校验。缺失即降级到 Tier-0 手写实现，**功能与门禁不减**。

---

## 14. 工程收口（踩坑库 §H）

- **先写 `pyproject.toml`（line-length=96）再跑 `ruff format`**，否则 CI `ruff format --check` 必红。
- `pyproject.toml` 必须含 `[tool.pytest.ini_options] pythonpath = ["."]`，否则干净环境 `ModuleNotFoundError`。
- ruff 钉 **0.16.10**，CI 与 Dockerfile **不得 `|| true`**（MiaForge v0.1.1 自纠：swallow-lint 反模式）。
- 交付前 `git grep -nE "(sk-|api[_-]?key|token|secret|password)"` 密钥自查。
- 所有 `open(..., encoding="utf-8")`；CLI 打印前 `sys.stdout.reconfigure(encoding="utf-8")`（Windows GBK 崩 UnicodeEncodeError）。
- 推送一律走 `scripts/gh_push.py`（git 智能协议 502）。
- `.gitignore` 必须排除 `results/*.json` / `.coverage` / `benchmark*.json`；且注意 `gh_push.py` 的 API 通道**不解析 .gitignore**，需按 blob-sha 跳过或事后清理（踩坑库 §E）。

---

## 15. 已知风险与开放问题（交 Phase 2 · 苏推衍）

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| R1 | 锚点 A 的 `2^n` 在 n=12 需 4096 次训练，若 utility 模型变贵（如 RF）会爆预算 | 金标准不可得 | 锚点 A 固定用 **LR/Ridge**（廉价确定性）；RF 类 utility 只走锚点 B + 下游指标 |
| R2 | 锚点 B 要求 utility 必须是 **KNN**，此时 KNN-Shapley 是"参赛者兼裁判" | 公平性争议 | 明确标注锚点 B 的 utility 口径；KNN-Shapley 在该口径下的成绩**单独列**，不参与"最优单法"排名（踩坑库 §A：线性专用方法的同款陷阱） |
| R3 | Influence Function 需 iHVP；非凸模型下一阶近似失效 | 数值不稳 | 先在 LR/Ridge（凸）上做；非凸时标 `skipped` 并说明，**不伪造数字** |
| R4 | Beta Shapley 的 (α,β) 需网格；不同 α,β 与不同数据集强弱不同 | 固定权重可能在半数数据集为负 | 网格预注册 + 如数据依赖选择须用**独立验证实例**（踩坑库 §A/B） |
| R5 | 旗舰可能结构性地退化为某个成员 | 门禁不可达 | 按 §11.4 走**效率轴**口径，并诚实声明同款（TscForge / DensForge 先例） |
| R6 | MC 置换采样在 numpy 版本间流不稳定 | 跨版本漂移 | 钉 `numpy==2.5.3` + CI 矩阵 3.12/3.13 + 门禁预算留结构性余量 |

---

## 16. 环境就绪实测（Phase 1 证据）

见 Phase 1 报告 §2 / §3（venv 路径、`pip freeze`、import 验证输出）。本文件与报告互为补充：本报告记录**实测命令与输出**，本文件记录**架构契约**。

> 作者：**晨星**
