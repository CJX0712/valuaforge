# ValuaForge Phase 1 报告：轮子先验 + 环境就绪 + 架构

> 作者：**晨星**
> 阶段：Phase 1（架构 + 选型 + 轮子先验 + 环境就绪）——**未写任何业务算法实现**
> 全部数据为**实测输出**，无推断、无占位。

---

## 交付结论

**环境已就绪，架构已冻结，零阻塞项。**
7/7 个 Tier-0 依赖在 win_amd64 + cp313 上均有预编译 wheel，venv 已建成并完成 import 与 API 级验证；架构规格落盘 `docs/architecture_spec.md`；已显式否决 `opendataval`（实测与 `numpy==2.5.3` 冲突）。

---

## 环境前提

| 项 | 实测值 |
|----|--------|
| OS | Windows（Git Bash），无 GPU、无 MSVC、无 cmake |
| 解释器 | **Python 3.13.14**（托管目录名 `3.13.12` 仅为标签；仍属 cp313 ABI） |
| venv | `C:/Users/Administrator/.workbuddy/binaries/python/envs/valuaforge` |
| venv python | `C:/Users/Administrator/.workbuddy/binaries/python/envs/valuaforge/Scripts/python.exe` |
| venv pip | 26.2.1（已从 26.1.2 升级） |
| BLAS | scipy-openblas |

---

## A. 轮子先验（实测）

### A.1 版本索引

```
C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe \
  -m pip index versions <pkg> --proxy "" --index-url https://pypi.org/simple
```

> `pip index versions` **一次只收一个参数**（实测报 `ERROR: You need to specify exactly one argument`），必须逐包执行。

### A.2 平台级 wheel 探测（关键）

`pip index` 只证明"有版本"，不证明"有 win_amd64/cp313 预编译 wheel"，故补一轮：

```
python.exe -m pip download --no-deps --only-binary=:all: \
  --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 \
  -d . "<pkg>==<ver>" --proxy "" --index-url https://pypi.org/simple
```

| 包 | 锁定版本 | index 最新 | 实测 wheel 文件名 | 大小 | 状态 |
|----|---------|-----------|------------------|------|------|
| numpy | **2.5.3** | 2.5.3 | `numpy-2.5.3-cp313-cp313-win_amd64.whl` | 12.6 MB | ✅ |
| scikit-learn | **1.9.1** | 1.9.1 | `scikit_learn-1.9.1-cp313-cp313-win_amd64.whl` | 8.2 MB | ✅ |
| scipy | **1.18.1** | 1.18.1 | `scipy-1.18.1-cp313-cp313-win_amd64.whl` | 36.6 MB | ✅ |
| pytest | **9.1.1** | 9.1.1 | `pytest-9.1.1-py3-none-any.whl` | 386 kB | ✅ |
| pytest-cov | **7.1.0** | 7.1.0 | `pytest_cov-7.1.0-py3-none-any.whl` | 22 kB | ✅ |
| coverage | **7.16.2** | 7.16.2 | `coverage-7.16.2-cp313-cp313-win_amd64.whl` | 226 kB | ✅ |
| ruff | **0.16.10** | 0.16.10 | `ruff-0.16.10-py3-none-win_amd64.whl` | 10.6 MB | ✅ |

**结论：7/7 全绿，全部预编译，零源码编译。**

### A.3 显式否决的外部依赖

| 候选 | 结论 | 实测证据 |
|------|------|---------|
| `opendataval` 1.3.0 | **否决** | wheel 为 `py3-none-any` 可下载，但 METADATA 实测 **23 条 `Requires-Dist`**：`numpy<1.26,>=1.22.4`（与本域强制 `numpy==2.5.3` **直接冲突**）、`torch~=2.1.0`、`torchvision~=0.16.0`、`transformers~=4.35`、`pykeops~=2.1.2`（需编译，本机无 MSVC）、`geomloss~=0.2.6`。引入即破坏 numpy 锁版本并触发 2GB+ 下载与编译失败 |
| `xgboost` / `lightgbm` | **不需要** | 本域 utility 用 LR/KNN/Ridge 足够 |
| `shap` | **不需要** | 是**特征**归因库，非数据点估值 |

> 因此：估值算法 **100% 手写纯 numpy**；sklearn 仅作 Tier-1 utility 模型与指标交叉校验。此模式与已交付评级 S/A 的系统（AttribForge / PTQForge / HawkesForge / SeqForge / BayesForge）一致。

---

## B. 环境就绪（实测）

### B.1 建 venv

```
C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe \
  -m venv C:/Users/Administrator/.workbuddy/binaries/python/envs/valuaforge
```

- **本次 `python -m venv` 未被沙箱拦截**（在 `dangerouslyDisableSandbox=true` 下执行成功）。
- 既有 36 个 venv（attribforge / densforge / labelforge / hawkesforge …），**无** `valuaforge` ⇒ 新建，未复用。
- `Scripts/` 下 `python.exe` / `pip.exe` / `ruff.exe` / `activate.bat` 齐备（Windows 是 `Scripts` 非 `bin`）。

### B.2 装依赖

```
<venv>/Scripts/python.exe -m pip install -U pip \
  --proxy "" --index-url https://pypi.org/simple        # pip 26.1.2 → 26.2.1

<venv>/Scripts/python.exe -m pip install "numpy==2.5.3" "scikit-learn==1.9.1" "scipy==1.18.1" \
  "pytest==9.1.1" "pytest-cov==7.1.0" "coverage==7.16.2" "ruff==0.16.10" \
  --proxy "" --index-url https://pypi.org/simple
```

> `--proxy ""` 一次根治 pip 26 从 Windows 系统代理/注册表另取 SOCKS 源导致的 `Missing dependencies for SOCKS support`（踩坑库 §D）。

### B.3 import 验证输出

```
numpy        2.5.3
scipy        1.18.1
sklearn      1.9.1
pytest       9.1.1
coverage     7.16.2
sklearn submodules OK: linear_model/neighbors/ensemble/datasets
scipy.stats OK: spearmanr/kendalltau/rankdata
BLAS: scipy-openblas
```

### B.4 `pip freeze`（valuaforge venv 全量）

```
cloudpickle==3.1.2
colorama==0.4.6
coverage==7.16.2
iniconfig==2.3.0
joblib==1.6.0
narwhals==2.26.0
numpy==2.5.3
packaging==26.3
pluggy==1.6.0
Pygments==2.21.0
pytest==9.1.1
pytest-cov==7.1.0
ruff==0.16.10
scikit-learn==1.9.1
scipy==1.18.1
threadpoolctl==3.7.0
```

（无 `pip` / `setuptools` 行，因 `pip freeze` 不含自身；venv pip 实测 26.2.1）

`ruff --version` → `ruff 0.16.10`；`pytest --version` → `pytest 9.1.1`。

### B.5 计划依赖的**具体 API** 实测（import 通过 ≠ API 还在）

| 检查项 | 实测结果 | 状态 |
|--------|---------|------|
| `np.trapezoid` | 存在 | ✅ |
| `np.trapz` | 已移除 | ✅ 预期内 |
| `np.float_` | 已移除 | ✅ 须用 `np.float64` |
| `default_rng(SeedSequence([seed, crc32(stream)]))` | 同 stream 两次派生**逐位相等**；不同 stream 相异 | ✅ 确定性原语可用 |
| `load_digits`（离线） | `(1797, 64)`，无需联网 | ✅ |
| `LogisticRegression(solver="lbfgs")` 多分类 | 10 类可用 | ✅ sklearn 1.9 已移除 `multi_class=`，`liblinear` 不支持多分类 ⇒ **必须 lbfgs** |
| `KNeighborsClassifier(5)` | val acc 0.895 | ✅ |
| `BaggingClassifier(oob_score=True)` | `oob_score_=0.91`，`oob_decision_function_` 存在 | ✅ Data-OOB 可行 |
| `scipy.stats.spearmanr` **常量输入** | **返回 `nan`** | ⚠️ 必须自守 |
| `scipy.stats.kendalltau` **常量输入** | **返回 `nan`** | ⚠️ 必须自守 |

> **重要实测修正**：scipy 对常量输入返回的是 **`nan`**——既不是踩坑库记载的 `±1`，也不是我们期望的 `0.0`。
> `eval/metrics.py` **禁止直接透传 scipy 结果**：常量输入短路返回 `0.0`，任何 `nan` 必须在指标层显式拦截，否则会污染 mean±std 聚合使整列指标静默失效。平局（tie）取**平均秩**，自实现路径不可依赖 scipy 默认行为。

---

## C. 架构骨架要点

落盘：`docs/architecture_spec.md`（完整版）。要点：

1. **依赖方向单向无环**：`cli → pipeline → {data, valuation, eval} → core`；统一**绝对导入**（`core.xxx` / `data.xxx` / `valuation.xxx` / `eval.xxx`）。
2. **模块**：`core/{types, errors, config, interfaces, seed, registry}`、`data/{dgp, loader, corrupt}`、`valuation/{base, utility, shapley_exact, knn_shapley, tmc, beta, banzhaf, influence, oob, valuafuse, registry}`、`eval/{gold, metrics, downstream, report}`、`pipeline/{pipeline, benchmark}`、`cli.py`。
3. **错误码** E100~E500（配置 / 数据 / 估值 / 评测 / 管线五段），含泄漏守卫 `E204`、金标准超限 `E400`、确定性违规 `E500`。
4. **确定性**：唯一 seed 入口 `core/seed.SeedBank`；子流派生用 `crc32(stream_name)`（**禁用 `hash(str)`**，受 `PYTHONHASHSEED` 随机化）；`n_jobs=1`；集合迭代前排序；钉 numpy 2.5.3。
5. **Tier-0/Tier-1**：Tier-0 纯 numpy/scipy 可端到端跑通；Tier-1 sklearn 仅作 utility 模型与交叉校验，缺失降级但功能不减。`available_*()` 走**类级惰性工厂**。
6. **性能预算**：demo ≤60s（锚点A≈3s / 锚点B≈15s / 移除曲线≈25s / 噪声检测≈10s / 缓冲≈7s）、内存 ≤2GB、pytest ≤120s。
7. **预算货币 = utility 评估次数**，由 `UtilityFn` **硬计数**（禁止按公式估算——估算与实际不符属作弊）。

### 双金标准锚点（本系统评测设计的核心）

| 锚点 | 原理 | 适用 | 成本 |
|------|------|------|------|
| **A 暴力穷举** | `φ_i = Σ_{S⊆D\{i}} \|S\|!(n−\|S\|−1)!/n! ·[v(S∪i)−v(S)]`，共享 `v(S)` ⇒ `2^n` 次 | **任意 utility**，n ≤ 12 | ≤4096 次小样本训练，秒级 |
| **B KNN 闭式解** | Jia et al. 2019 证明 KNN utility 下 **KNN-Shapley 闭式解 = 精确 Data Shapley** | utility 必须是 KNN，**任意 n** | O(n log n)，可在 n=256/512 上做精确误差评估 |

锚点 B 使大规模下的**精确误差**（非仅排序相关）成为可能，等价于 SeqForge 的"KF log-lik vs 闭式解差 1.1e-13"黄金锚点打法。

---

## D. 选型依据与 SOTA 对标

| 方法 | 出处 | 模块 | 复用/手写 |
|------|------|------|----------|
| Data Shapley | Ghorbani & Zou, ICML 2019 | `shapley_exact.py`（金标准） | 手写 |
| TMC-Shapley | 同上（截断置换 MC） | `tmc.py` | 手写 |
| KNN-Shapley 闭式 | Jia et al., ICML 2019 | `knn_shapley.py` | 手写 |
| Beta Shapley | Kwon & Zou, ICML 2022 | `beta.py` | 手写 |
| Influence Function | Koh & Liang, ICML 2017 | `influence.py` | 手写 |
| Data Banzhaf | Wang & Jia, ICML 2023 | `banzhaf.py` | 手写 |
| Data-OOB | Kwon & Zou, ICML 2023 | `oob.py` | 手写 |

**全部手写**，理由见 §A.3（无可用的生产级库；`opendataval` 实测与 numpy 锁版本冲突且需 torch/编译）。

**旗舰 ValuaFuse**（设计空间，终稿归 Phase 2 算法组）：
控制变量 + 逆方差加权多源融合 —— ① Pilot 估逐源方差 → ② Neyman 最优预算分配 → ③ 以**零成本确定性估计量**（KNN-Shapley / Influence，`E[C]` 精确已知）作控制变量做方差缩减，逆方差加权聚合。
**比较轴选"效率轴"**（等 utility 预算下比精度），而非"比最强成员再高 X%"——后者在融合成员含该最强方法时**结构性不可达**（踩坑库 §A，TscForge / DensForge 先例）。

---

## E. 域覆盖性判定

已交付 36 系统中**无数据估值 / Data Shapley 系统**。
- AttribForge 是**特征归因**（列），本系统是**数据点归因**（行）→ 对象不同，非近邻子域。
- ActiveForge 选**未标注**点，本系统给**已标注**训练点定价 → 不同任务。
⇒ 域未覆盖，可安全开工。

---

## F. 阻塞项

**无阻塞项。**

### 待 Phase 2 处理（非阻塞，已列入 architecture_spec.md §15）

| # | 风险 | 建议 |
|---|------|------|
| R1 | 锚点 A 的 `2^n` 若 utility 变贵（RF）会爆预算 | 锚点 A 固定用 LR/Ridge；RF 只走锚点 B + 下游指标 |
| R2 | 锚点 B 下 KNN-Shapley "参赛者兼裁判" | 该口径成绩**单独列**，不参与"最优单法"排名 |
| R3 | Influence 在非凸模型下一阶近似失效 | 先在 LR/Ridge 上做；非凸标 `skipped`，**不伪造数字** |
| R4 | Beta(α,β) 网格可能数据依赖 | 预注册网格；如需数据依赖选择须用**独立验证实例** |
| R5 | 旗舰可能结构性退化为某成员 | 走效率轴口径并诚实声明 |
| R6 | `Generator` 流跨 numpy 版本不稳定（NEP 19） | 钉 2.5.3 + CI 矩阵 3.12/3.13 + 预算留结构性余量 |

### 给 Phase 2 实现的前置约束（踩坑库）

- **先写 `pyproject.toml`（line-length=96）再跑 `ruff format`**，否则 CI `ruff format --check` 必红。
- `pyproject.toml` 必含 `[tool.pytest.ini_options] pythonpath = ["."]`。
- ruff 钉 0.16.10，CI/Dockerfile **不得 `|| true`**。
- 所有 `open(encoding="utf-8")`；CLI 打印前 `sys.stdout.reconfigure(encoding="utf-8")`。
- 推送一律走 `scripts/gh_push.py`。

---

## 交付清单

| 项 | 路径 | 说明 |
|----|------|------|
| 架构规格 | `C:/Users/Administrator/WorkBuddy/2026-10-05-03-14-04/valuaforge/docs/architecture_spec.md` | 模块职责表 / Protocol / 数据流 / E100~E500 / 配置 / 确定性 / 降级 / 双锚点 / 预算 / SOTA / 风险 |
| 本文件 | `.../valuaforge/docs/phase1_report.md` | 全部实测证据 |
| venv | `C:/Users/Administrator/.workbuddy/binaries/python/envs/valuaforge` | 7 包锁定，import + API 双验证 |

> 作者：**晨星**
