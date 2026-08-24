# Mini Matbench-Discovery: CHGNet on LLZO

这是一个可复现的小型评测：从 Materials Project 获取 `Li-La-Zr-O` 精确化学体系中的
LLZO 结构，以 CHGNet 进行结构弛豫，并将弛豫能量与 MP 的 DFT 参考能量进行对比。

## 评测口径

- 数据范围：Materials Project `chemsys=Li-La-Zr-O`，最多 120 个原子。
- 模型：CHGNet 预训练模型。
- 弛豫：默认 CPU、最多 100 步、目标最大力 `0.1 eV/Å`、晶格与原子同时弛豫。
- 指标：能量 MAE、RMSE、Spearman 秩相关、收敛数和失败数。
- 失败不会被删除，会写入 checkpoint 和最终 CSV。

> Materials Project 当前只返回 3 个满足上述精确化学体系的结构，虽然命令默认上限是
> 20，但程序不会用无关材料补足样本。因此本仓库是 LLZO 小样本 pilot，而不是完整的
> Matbench-Discovery 稳定性榜单复现。

这里比较的是 CHGNet 与 DFT 的**弛豫后能量**。`energy_above_hull` 被保留为参考字段，
但不会错误地拿原始总能与凸包距离直接计算相关性。

## 环境

```powershell
conda env create -f "environment.yml" --override-channels
conda activate "mini-matbench-llzo"
Copy-Item ".env.example" ".env"
# 在 .env 中填写 MP_API_KEY；该文件已被 git 忽略
```

## 运行

```powershell
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

API 原始结构与运行 checkpoint 位于 `data/raw/` 和 `data/checkpoints/`，不会提交到仓库。
