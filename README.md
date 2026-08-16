# 虚拟细胞竞赛 — 蛋白质组扰动响应预测（exp17 基线）

预测酵母在化学扰动 × 菌株 × 时间条件下 4,232 个蛋白质的 log2 丰度变化。

本仓库是**未引入蛋白质网络信息**的版本，对应实验 `exp17_high_wfc_ensemble`。

## 结果

7-seed ensemble 在验证集上的表现：

| Split | n | fc_pcc | per_protein_r2 | global_r2 | resid_ctx | resid_chem |
|---|---|---|---|---|---|---|
| val_strain_only | 1357 | 0.3807 | 0.4941 | 0.9632 | 0.4101 | 0.4727 |
| val_chem_only | 1065 | 0.3986 | 0.7125 | 0.9773 | 0.3562 | 0.4076 |
| val_both | 269 | 0.2730 | 0.5921 | 0.9661 | 0.3309 | 0.3884 |
| val_time | 142 | 0.5990 | 0.7981 | 0.9838 | 0.5013 | 0.5589 |

`val_both`（同时未见菌株与化合物）是最难的场景，也是主要优化目标。
相对最初的单模型基线（0.2451）提升约 11%。

**关于种子数：** 实测 ensemble 收益在 7 个种子已饱和，继续增加种子数不再带来提升
（提升幅度小于单模型的种子间标准差，约 0.007）。

主要评测指标 `fc_pcc` 为 fold-change 的逐样本 Pearson 相关系数，
Δ 用 matched control 计算（同来源/仪器/板/菌株/培养基/温度/时间）。

## 方法

### 架构：残差分解（`model_residual.py`）

预测值分解为可辨识的几项，靠**输入隔离**保证各项不互相吸收信号：

```
ŷ = 共享应激响应(上下文 + 菌株，不含化合物身份)
  + 化合物特异残差(化合物 + 上下文，跨菌株共享的扰动效应)
  + 菌株调制残差(菌株 × 化合物)
  + 菌株×化合物低秩双线性交互(rank=32)
  + 观测校准(仪器/板/来源，零初始化，不参与生物泛化)
```

残差分支末层零初始化：训练初期等价于纯共享模型，收敛更稳。

### 关键技巧

按消融实验的贡献排序：

1. **Feature dropout**（最大单项提升，+8.5%）
   训练时按样本随机置零菌株 one-hot（p=0.3）与化合物 one-hot（p=0.2），
   模拟测试时的未见实体，迫使模型依赖可迁移信号（Uni-Mol 表征）而非记忆 one-hot。

2. **提高 fold-change 损失权重**（`w_fc=0.6`）
   `loss = masked_mse + w_fc * masked_fc_loss`。
   直接对齐评测指标；相对变化比绝对丰度更稳健（消除了仪器批次偏移）。

3. **多种子 ensemble**（10 seeds，预测平均）

### 特征编码（`features.py`）

生物条件与观测过程**分两路**，观测字段（仪器/板/来源）不进生物编码——
否则模型会把仪器偏好当生物规律，评测时仪器分布一变就崩。

- 菌株：one-hot（未见菌株全零）
- 化合物：one-hot + Uni-Mol 512 维表征经 PCA 降到 32 维
  （one-hot 对已见化合物更强，Uni-Mol 是未见化合物唯一的可迁移信号）
- 上下文：培养基 + 温度 one-hot + 时间周期编码（sin/cos）
- 观测：data_source / instrument / plate one-hot

### 数据预处理（`common.py`）

- 蛋白质过滤：仅用训练行计算缺失率，取缺失率最低的 4,232 个
- log2 变换 + 跨样本中位数归一化（校正仪器/上样量偏移）
- 缺失值用 mask 处理，不做填充

## 使用

```bash
pip install -r requirements.txt

# 数据放到 dataset/input/（见下方"数据"一节）

# 可选：生成 Uni-Mol 分子表征（缺失时自动回落到 hash 编码）
python code/fetch_smiles.py
python code/build_unimol.py

# 统计基线（验证数据流水线，所有模型必须超越 matched control）
python code/stage0_baseline.py

# 训练 exp17（7-seed ensemble，推荐）
python code/stage4_exp.py --exp exp17_high_wfc_ensemble

# 生成提交文件
python code/generate_submission.py --model exp17_high_wfc_ensemble
```

路径默认以仓库根目录为基准，可用 `VC_ROOT` 环境变量覆盖。

## 数据

原始数据未包含在仓库中（约 300MB），需自行放置到 `dataset/input/`：

```
WAYB_WAYC_metadata_train_val(1).csv
WAYB_WAYC_metadata_test(1).csv
WAYB_WAYC_proteome_raw_train_val.csv
WAYB_WAYC_proteome_raw_test.csv
```

首次加载会在 `cache/` 生成 npy 缓存（原始 CSV 读取约 1-2 分钟，之后秒级）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `common.py` | 数据加载、蛋白质过滤、log2 归一化、matched control |
| `features.py` | 条件编码器（生物/观测分离，Uni-Mol + one-hot） |
| `metrics.py` | mask-aware 评测指标（fc_pcc / residual_pcc / per_protein_r2） |
| `model_residual.py` | 残差分解模型 + 损失函数 |
| `model_shared.py` | 共享 encoder + 多头输出（备选架构，消融证明不如残差分解） |
| `stage0_baseline.py` | 统计基线与分场景诊断 |
| `stage3_residual.py` | 残差分解单模型训练 |
| `stage4_exp.py` | 系统实验链（exp02–exp22 配置与训练/评估循环） |
| `preprocess.py` | 独立的预处理脚本 |
| `generate_submission.py` | 生成提交文件（支持 ensemble） |

## 实验记录

`stage4_exp.py` 的 `EXPERIMENTS` 字典包含全部实验配置。要点：

| 实验 | 变化 | val_both fc_pcc |
|---|---|---|
| exp01 | 残差分解基线 | 0.2451 |
| exp02 | + feature dropout | 0.2660 |
| exp03 | 更大模型（hidden=1024） | 0.2633 ↓ 过拟合 |
| exp05 | + RDKit 指纹 | 0.2593 ↓ 不如 Uni-Mol |
| exp06 | SharedMultiHead 架构 | 0.2549 ↓ |
| exp11 | w_fc=0.6（最佳单模型） | 0.2675 |
| exp12 | 更长训练（800 ep） | 0.2612 ↓ 早停更好 |
| **exp17** | **7-seed ensemble** | **0.2730** ← 本仓库默认 |

两点值得注意：

- **增大模型容量不如加强正则化。** exp03（hidden=1024）与 exp08（叠加全部技巧）
  都低于简单配置，瓶颈在泛化而非拟合能力。
- **ensemble 收益在 7 个种子已饱和**，继续加种子的提升小于随机波动。
