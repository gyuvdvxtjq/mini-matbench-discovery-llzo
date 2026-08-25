# Mini Matbench-Discovery: CHGNet on LLZO

这是一个可复现的两层评测：从 Materials Project 获取 `Li-La-Zr-O` 相空间中的结构，
用 CHGNet 评估能量与热力学稳定性，并对精确 LLZO 四元结构进行完整结构弛豫。

![CHGNet vs MP convex-hull distance parity](data/phase_space/plots/hull_parity.png)

- 相空间层凸包距离 MAE **0.038 eV/atom**、Spearman **0.86**；50 meV/atom 候选分类 F1 **0.81**
- 完整结果与误差归因分析见 [`RESULTS.md`](RESULTS.md)

## 评测口径

- 相空间层：Li、La、Zr、O 的元素、二元、三元和四元体系，共 150 个结构。
- LLZO 层：Materials Project `chemsys=Li-La-Zr-O` 的 3 个精确四元结构。
- 模型：CHGNet 预训练模型。
- 相空间指标：形成能 MAE、凸包距离 MAE、Spearman、稳定相分类。
- LLZO 弛豫：默认 CPU、最多 100 步、目标最大力 `0.1 eV/Å`。
- 失败不会被删除，会写入 checkpoint 和最终 CSV。

> Materials Project 当前只返回 3 个精确 LLZO 四元结构。稳定性不能只由这 3 个结构
> 决定，因此相空间评测额外纳入 147 个元素及竞争相，而不是用无关材料补数。

相空间脚本使用 CHGNet 单点能重建凸包；LLZO 脚本比较弛豫后能量。不同赝势/元素基准下
原始总能存在元素相关常数偏移，因此稳定性结论使用形成能和凸包距离，而不使用跨组分
原始总能 MAE；LLZO 弛豫层的能量误差是同一 `La-Li-O-Zr` 元素空间内的相对比较。
误差归因分析（富 La 结构低估、单质多形体排序翻转、O2 分子晶体失效）见
`RESULTS.md` 第 4 节。

## 环境

```powershell
conda env create -f "environment.yml" --override-channels
conda activate "mini-matbench-llzo"
Copy-Item ".env.example" ".env"
# 在 .env 中填写 MP_API_KEY；该文件已被 git 忽略
```

## 运行

```powershell
python "phase_space_benchmark.py"
python "battery_mlip_pilot.py"
```

常用参数：

```powershell
# 只验证 API 和缓存数据
python "battery_mlip_pilot.py" --fetch-only

# 从头运行固定晶格弛豫
python "battery_mlip_pilot.py" --fresh --no-relax-cell
```

中断后再次执行会跳过已经成功的材料。主要输出为：

- `data/chgnet_vs_mp_llzo.csv`
- `data/metrics.json`
- `data/largest_errors.csv`
- `data/plots/energy_parity.png`
- `data/plots/absolute_errors.png`
- `data/phase_space/chgnet_phase_space.csv`
- `data/phase_space/metrics.json`
- `data/phase_space/plots/hull_parity.png`
- `data/phase_space/plots/hull_mae_by_chemsys.png`

API 原始结构与运行 checkpoint 位于 `data/raw/` 和 `data/checkpoints/`，不会提交到仓库。

## 解释边界

CHGNet 的训练数据包含 Materials Project 数据，因此这里衡量的是已知 MP 化学空间上的
**工作流正确性与域内可靠性**，不是严格的域外泛化成绩，也不冒充完整
Matbench-Discovery 榜单结果。
