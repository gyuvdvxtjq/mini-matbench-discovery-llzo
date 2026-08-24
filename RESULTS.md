# Li-La-Zr-O 相空间与 LLZO 评测结果

## 1. 相空间稳定性评测

### 数据与方法

- 数据源：Materials Project API
- 范围：Li、La、Zr、O 的全部元素、二元、三元和四元化学体系
- 样本：150 个结构，覆盖 14 个有数据的子化学体系
- 模型：CHGNet `0.3.0`（软件包 `chgnet==0.4.2`）
- 推理：在 MP 的 DFT 弛豫结构上计算 CHGNet 单点能
- 稳定性：用全部 150 个 CHGNet 能量重新构建相图与凸包

### 汇总结果

| 指标 | 结果 |
| --- | ---: |
| 成功 / 失败 | 150 / 0 |
| 形成能 MAE | 0.0418 eV/atom |
| 凸包距离 MAE | 0.0376 eV/atom |
| 凸包距离 RMSE | 0.0558 eV/atom |
| 凸包距离 Spearman | 0.8645 |
| 严格稳定相 Precision / Recall / F1 | 0.500 / 0.400 / 0.444 |
| 50 meV/atom 候选分类 Precision / Recall / F1 | 0.758 / 0.877 / 0.813 |

原始 DFT 总能与 CHGNet 能量含有不同的元素参考偏移，尤其在 La、Zr 金属体系中明显，
所以跨组分稳定性结论采用形成能和凸包距离，而不采用原始总能 MAE。

![Hull parity](data/phase_space/plots/hull_parity.png)

![Hull MAE by chemical system](data/phase_space/plots/hull_mae_by_chemsys.png)

## 2. 精确 LLZO 四元结构

Materials Project 当前仅有 3 个精确 `Li-La-Zr-O` 结构。相空间模型对它们的凸包预测为：

| MP ID | 化学式 | MP E-hull | CHGNet E-hull | 误差 (eV/atom) |
| --- | --- | ---: | ---: | ---: |
| `mp-1120817` | Li17La12Zr8O48 | 0.1283 | 0.0652 | -0.0632 |
| `mp-1239180` | Li19La12Zr8O48 | 0.0932 | 0.0345 | -0.0587 |
| `mp-942733` | Li7La3Zr2O12 | 0.0068 | 0.0188 | +0.0120 |

这说明 CHGNet 正确保留了三者的非稳定相判断，但对两个富 La 结构的亚稳程度有约
59–63 meV/atom 的低估。

## 3. LLZO 完整弛豫

配置：CPU，最多 100 个离子步，`fmax=0.1 eV/Å`，允许晶格弛豫。

| 指标 | 结果 |
| --- | ---: |
| 成功 / 失败 | 3 / 0 |
| 达到目标力阈值 | 3 / 3 |
| 弛豫能量 MAE | 0.0450 eV/atom |
| 弛豫能量 RMSE | 0.0455 eV/atom |

完整数值见：

- `data/phase_space/chgnet_phase_space.csv`
- `data/phase_space/metrics.json`
- `data/chgnet_vs_mp_llzo.csv`
- `data/metrics.json`

## 4. 解释边界

CHGNet 的训练数据包含 Materials Project 数据，因此这是一项域内可靠性和工作流验证，
不是严格的域外泛化测试。150 个相空间结构让稳定性指标具有可分析性，但仍不能替代
WBM 等独立发现集上的完整 Matbench-Discovery 评测。
