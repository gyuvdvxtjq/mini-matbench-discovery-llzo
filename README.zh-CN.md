# mini-matbench-discovery-llzo

**基于 CHGNet 的 LLZO 固态电解质热力学稳定性基准测试**

[English](README.md) | [简体中文](README.zh-CN.md)

---

## 项目概述

本仓库提供一套聚焦且可复现的基准测试，用于评估预训练通用机器学习原子间势
**CHGNet** 在 Li–La–Zr–O 化学空间（石榴石型固态电解质 LLZO 所在相空间）中
预测能量与热力学稳定性的可靠性。判断一个晶体结构是否稳定，传统上依赖昂贵的
量子力学计算（DFT，单个结构耗时数小时）；CHGNet 这类原子间势把这一步加速到
秒级，正被广泛用于新材料筛选。本基准回答的问题是：**这种加速在该体系中是否
可信——以及它什么时候会失灵。**

评测分为两个层级：

- **相空间层**：Li、La、Zr、O 四种元素的全部单质、二元、三元、四元体系，
  共 150 个结构（14 个子化学体系）。用 CHGNet 单点能重建凸包，与 DFT 参考
  对比形成能、凸包距离与稳定相分类。
- **LLZO 层**：对 Materials Project 上仅有的 3 个精确四元 LLZO 结构做完整
  结构弛豫（最多 100 离子步，力收敛判据 0.1 eV/Å），弛豫能量在同一元素空间内
  与 DFT 直接对比。

失败样本不会被删除，而是写入 checkpoint 和最终 CSV 作为基准证据保留。

> Materials Project 当前只返回 3 个精确 LLZO 四元结构。稳定性不能只由这 3 个
> 结构决定，因此相空间评测额外纳入其余 147 个元素及竞争相，而不是用无关材料补数。

原始 DFT 总能与 CHGNet 能量含有不同的元素相关参考偏移，因此所有跨组分结论均使用
形成能和凸包距离，不使用跨组分原始总能 MAE。

---

## 主要结果

| 指标 | 数值 |
| --- | ---: |
| 凸包距离（E-hull）MAE | **0.0376 eV/atom** |
| 凸包距离 Spearman ρ | **0.8645** |
| 50 meV/atom 候选分类 F1（正例 57 个） | **0.813** |
| 严格稳定相分类 F1（正例 15 个） | 0.444 |
| 形成能 MAE | 0.0418 eV/atom |
| LLZO 弛豫收敛数 / 弛豫能量 MAE | 3 / 3 · 0.0450 eV/atom |

![Hull parity](data/phase_space/plots/hull_parity.png)

![Hull MAE by chemical system](data/phase_space/plots/hull_mae_by_chemsys.png)

完整指标、口径说明与误差归因分析——两个富 La 结构为何被系统性低估约
60 meV/atom、单质多形体排序翻转、O₂ 分子晶体双峰失效簇——见
[`RESULTS.md`](RESULTS.md)。

---

## 多模型原子间势审计框架（`mlip_audit`）

上文的单模型试点被一个通用审计框架扩展：让**多个通用原子间势在同一套协议**下
完成评测并横向对比：

- `mlip_audit/`——配置驱动的运行器：单点能、晶胞弛豫、NVT→NVE 分子动力学，
  每条记录写入 `data/runs/llzo/checkpoints.jsonl`，支持断点续跑，新增模型不会
  触发已有模型的重复计算。
- `configs/llzo.yaml`——LLZO 审计配置（Li/La/Zr/O 相空间 + mp-942733 在
  800–1400 K 的锂离子扩散率）；`configs/smoke.yaml` 为可在 CPU 上跑通的小型
  冒烟测试。
- `run_audit.py`——命令行入口（`python run_audit.py --config configs/llzo.yaml`）。
- `cloud/`——Bohrium 云端批量任务脚本（详见 `cloud/README.md`）。
- `tests/`——框架核心单元测试，无需下载任何模型。

本次 LLZO 运行覆盖 CHGNet 0.3.0 与 MACE-MP-0（medium）；DPA-4 因 deepmd-kit
无法与 mace-torch 共存（pip 依赖冲突）而跳过。所有后端均为可选——缺某个模型
只会记入 `report.json` 的 `skipped_models`，不会让整个任务失败。结果（含按原样
保留的异常：MACE 在 O₂ 弛豫上的灾难性失效、小超胞导致 1200 K 扩散率为负）位于
`data/runs/llzo/`，运行说明见 [`data/runs/llzo/NOTES.md`](data/runs/llzo/NOTES.md)。

---

## 仓库结构

```
.
├── phase_space_benchmark.py   # 相空间基准（150 个结构，单点能）
├── battery_mlip_pilot.py      # LLZO 结构弛豫（3 个四元结构）
├── mlip_audit/                # 多模型审计框架（配置 -> 协议 -> 指标）
├── run_audit.py               # 审计框架的命令行入口
├── configs/                   # 审计配置（llzo.yaml、smoke.yaml）
├── tests/                     # 审计核心单元测试
├── cloud/                     # Bohrium 云端任务脚本（setup/submit/run_job）
├── models/                    # 模型文件（DPA-4 训练输入；.pt 运行时拉取）
├── environment.yml            # Conda 环境配置
├── requirements.txt           # pip 依赖清单
├── RESULTS.md                 # 完整结果与误差分析
├── .env.example               # MP_API_KEY 模板（不入库）
└── data/                      # 运行后生成的输出
    ├── chgnet_vs_mp_llzo.csv  #   弛豫 vs DFT 对比表（单模型试点）
    ├── metrics.json           #   核心指标（单模型试点）
    ├── phase_space/           #   逐结构 CSV + 凸包指标 + 图表
    ├── runs/llzo/             #   多模型审计结果（report.json、指标、CSV）
    ├── raw/                   #   MP API 原始快照（不入库）
    └── checkpoints/           #   断点续跑 checkpoint（不入库）
```

---

## 使用方法

### 环境配置

前置要求：安装 [conda](https://docs.conda.io/)；注册一个免费的
[Materials Project](https://next-gen.materialsproject.org/) API key。

```bash
git clone https://github.com/gyuvdvxtjq/mini-matbench-discovery-llzo.git
cd mini-matbench-discovery-llzo
conda env create -f environment.yml --override-channels
conda activate mini-matbench-llzo
cp .env.example .env        # Windows PowerShell 用：Copy-Item .env.example .env
# 打开 .env 填入 MP_API_KEY 后继续执行下面两条
python phase_space_benchmark.py
python battery_mlip_pilot.py
```

两个脚本都支持断点续跑：中断后再次执行会自动跳过已成功的结构。

### 常用参数

```bash
# 只验证 API 连通性和缓存数据
python battery_mlip_pilot.py --fetch-only

# 从头运行固定晶格弛豫
python battery_mlip_pilot.py --fresh --no-relax-cell
```

### 计算参数

- **模型**：`chgnet==0.4.2` 包加载的 CHGNet `0.3.0` 预训练权重。
- **弛豫设置**：最大 100 离子步，最大原子力 ≤ 0.1 eV/Å 判收敛。
- **硬件**：默认 CPU 可运行；通过 `--device` 支持 CUDA/MPS 加速（非必需）。

所有输出为 CSV 表格、JSON 指标文件和 PNG 图片，无需重跑即可直接查看。

---

## 范围与局限

CHGNet 的训练数据来自 Materials Project，而本项目评测的 150 个结构也全部取自
MP。因此这是一项**域内评测**：它衡量的是工作流的正确性，以及模型在训练数据
所属化学空间中的预测可靠性；它不衡量对训练分布之外新结构的泛化能力——那需要
像 Matbench-Discovery 那样在 WBM 等独立发现集上按完整协议另行评测，本文结果
不应被解读为对该成绩的估计。

---

## 致谢

感谢 CHGNet 开发团队和 Materials Project 联盟提供的开放模型与参考数据。

## 联系方式

欢迎通过 [GitHub Issues](https://github.com/gyuvdvxtjq/mini-matbench-discovery-llzo/issues) 提交问题或建议。
