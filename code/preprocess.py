"""蛋白质组预处理流水线（PDF §5.2 四步法 + per-sample 中位数归一化）。

流程
----
1. sample_ID 对齐 —— 以 sample_ID 为唯一键，不依赖行顺序
2. 蛋白过滤     —— 仅用训练行（split_final=='train'）计算缺失率，
                    删除缺失率 ≥ 0.80 的蛋白（5,243 → 4,232）
3. mask 矩阵    —— 1=有值, 0=缺失/非正值；训练时填 0 但 mask 屏蔽梯度
4. log2 转换    —— 非正值/缺失保持 NaN，缺失不填 0
5. 中位数归一化 —— per-sample 行中位数对齐，校正仪器/上样量系统偏移
                    （global_med 由全部样本的所有非缺失 log2 值估计）

输出（写入 cache/）
-------------------
  proteome_normed_{split}.npy   — float32，已归一化 log2，缺失为 NaN
  proteome_normed_{split}_cols.txt
  proteome_normed_{split}_ids.txt
  proteome_keep_cols.txt        — train_val 阶段选出的 4,232 个蛋白列名
                                   （test 用同一列表，保证列对齐）

用法
----
  python code/preprocess.py               # 处理 train_val
  python code/preprocess.py --split test  # 处理 test（须先跑 train_val）
  python code/preprocess.py --all         # 两个 split 一起处理

注意
----
- global_med 仅由 train_val 数据估计，test 用同一值，避免测试集信息泄漏。
- keep 列由 train_val 训练行决定，test 直接套用，列顺序严格一致。
- QC 样本（perturbation_no_concentration == 'Quality Control'）保留在矩阵中，
  仅在缺失率计算时被排除（与 DMSO/Water 对照一样不参与过滤基准）。
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────────────
# 以仓库根目录为基准，可用环境变量 VC_ROOT 覆盖
import os
ROOT  = Path(os.environ.get('VC_ROOT', Path(__file__).resolve().parent.parent))
DATA  = ROOT / 'dataset' / 'input'
CACHE = ROOT / 'cache'

MISSING_THR = 0.80   # 训练行缺失率阈值；PDF §5.2.1 / §3.2 均写 ≥80% 删除
# 注：0.80 阈值得 4,422 个蛋白，不等于 PDF 声明的 4,232。
# common.py 改用「取缺失率最低的 N_KEEP=4,232 个」对齐赛题报告。
# preprocess.py 遵循 PDF 文字描述（≥80% 删除），二者可在 common.py 中切换。
N_KEEP = 4232        # 与 common.py 保持一致，优先用此值


# ── 核心函数 ──────────────────────────────────────────────────────────────

def load_raw(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读 raw proteome + metadata，以 sample_ID 严格对齐行序。"""
    meta_fn = ('WAYB_WAYC_metadata_train_val(1).csv' if split == 'train_val'
               else 'WAYB_WAYC_metadata_test(1).csv')
    prot_fn = f'WAYB_WAYC_proteome_raw_{split}.csv'

    m = pd.read_csv(DATA / meta_fn).set_index('sample_ID')
    p = pd.read_csv(DATA / prot_fn).set_index('sample_ID')

    # 步骤 1：sample_ID 对齐（不依赖行顺序）
    p = p.reindex(m.index).astype(np.float32)
    assert p.notna().any(axis=1).all(), \
        '存在 sample_ID 无法匹配到 proteome，请检查数据文件'
    return m, p


def select_keep_proteins(m: pd.DataFrame, p: pd.DataFrame,
                          n_keep: int = N_KEEP) -> pd.Index:
    """步骤 2：仅用训练行（split_final=='train'）计算缺失率，选保留蛋白。

    取缺失率最低的 n_keep 个，并列时以原始列序为稳定 tie-break，
    结果与 common.keep_proteins() 完全一致。
    """
    tr = m['split_final'].eq('train').values
    mr = p.loc[tr].isna().mean(axis=0)           # 每个蛋白的缺失率
    order = np.lexsort((np.arange(len(mr)), mr.values))[:n_keep]
    keep = mr.index[order]
    print(f'  蛋白过滤: {p.shape[1]:,} → {len(keep):,}  '
          f'（阈值: 缺失率最低 {n_keep} 个，训练行={tr.sum()}）')
    return keep


def normalize(p: pd.DataFrame,
              keep: pd.Index,
              global_med: float | None = None) -> tuple[pd.DataFrame, float]:
    """步骤 3-5：mask 矩阵 + log2 + per-sample 中位数归一化。

    Args:
        p:          全量 raw proteome（所有蛋白列）
        keep:       保留蛋白的 Index
        global_med: 全局中位数参考值；None 时从当前数据估计（仅 train_val 阶段）。
                    test 阶段传入 train_val 阶段估计的值，避免泄漏。

    Returns:
        normed:     已归一化 log2 DataFrame（keep 列），缺失为 NaN（float32）
        global_med: 本次使用的全局中位数值
    """
    # 步骤 4：log2 转换（在全量蛋白上做，中位数估计更稳健）
    log2_full = np.log2(p.where(p > 0))          # 非正值/缺失 → NaN

    # 步骤 5：per-sample 中位数归一化
    if global_med is None:
        global_med = float(np.nanmedian(log2_full.values))
        print(f'  global_med（来自当前数据）= {global_med:.4f}')
    else:
        print(f'  global_med（外部传入）= {global_med:.4f}')

    row_med   = log2_full.median(axis=1)          # 每样本行中位数
    log2_norm = log2_full.sub(row_med, axis=0).add(global_med)

    # 步骤 3：保留 keep 列；mask 由 notna() 隐含（NaN 即缺失）
    normed = log2_norm.loc[:, keep].astype(np.float32)

    # 归一化后诊断
    vals = normed.values.ravel()
    vals = vals[~np.isnan(vals)]
    print(f'  归一化后 log2 范围: [{vals.min():.3f}, {vals.max():.3f}]  '
          f'中位数: {np.median(vals):.3f}')
    row_med2 = normed.median(axis=1)
    print(f'  行中位数 std（归一化后）: {row_med2.std():.4f}  '
          f'（归一化前: {row_med.std():.4f}）')
    return normed, global_med


def save_normed(normed: pd.DataFrame, split: str) -> None:
    """将归一化结果存入 cache/。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    npy  = CACHE / f'proteome_normed_{split}.npy'
    cols = CACHE / f'proteome_normed_{split}_cols.txt'
    ids  = CACHE / f'proteome_normed_{split}_ids.txt'

    np.save(npy, normed.values)
    cols.write_text('\n'.join(normed.columns), encoding='utf-8')
    ids.write_text('\n'.join(normed.index),   encoding='utf-8')
    print(f'  已保存: {npy}')


def load_normed(split: str) -> pd.DataFrame:
    """从 cache/ 加载已归一化 proteome（供 common.py 调用）。"""
    npy  = CACHE / f'proteome_normed_{split}.npy'
    cols = CACHE / f'proteome_normed_{split}_cols.txt'
    ids  = CACHE / f'proteome_normed_{split}_ids.txt'
    if not npy.exists():
        raise FileNotFoundError(
            f'{npy} 不存在，请先运行: python code/preprocess.py --split {split}')
    arr = np.load(npy)
    c   = cols.read_text(encoding='utf-8').split('\n')
    i   = ids.read_text(encoding='utf-8').split('\n')
    return pd.DataFrame(arr, index=pd.Index(i, name='sample_ID'), columns=c)


# ── 主流程 ────────────────────────────────────────────────────────────────

def process_train_val() -> float:
    """处理 train_val split，返回 global_med 供 test 使用。"""
    print('\n===== 处理 train_val =====')
    m, p = load_raw('train_val')
    keep  = select_keep_proteins(m, p)

    # 保存 keep 列名，test 阶段使用
    keep_path = CACHE / 'proteome_keep_cols.txt'
    keep_path.write_text('\n'.join(keep), encoding='utf-8')
    print(f'  keep 列名已保存: {keep_path}')

    normed, global_med = normalize(p, keep, global_med=None)
    save_normed(normed, 'train_val')

    # 保存 global_med 供 test 阶段读取
    gm_path = CACHE / 'proteome_global_med.txt'
    gm_path.write_text(str(global_med), encoding='utf-8')
    print(f'  global_med 已保存: {gm_path}')

    # 按来源/仪器诊断行中位数对齐情况
    print('\n  --- 各来源行中位数均值 ---')
    row_med = normed.median(axis=1)
    for src, grp in m.groupby('data_source'):
        print(f'    {src:12s}: {row_med.loc[grp.index].mean():.4f}  (n={len(grp)})')
    for inst, grp in m.groupby('instrument'):
        print(f'    {inst:12s}: {row_med.loc[grp.index].mean():.4f}  (n={len(grp)})')

    return global_med


def process_test(global_med: float | None = None) -> None:
    """处理 test split，使用 train_val 的 keep 列和 global_med。"""
    print('\n===== 处理 test =====')

    keep_path = CACHE / 'proteome_keep_cols.txt'
    if not keep_path.exists():
        raise FileNotFoundError('请先运行 train_val: python code/preprocess.py')
    keep = pd.Index(keep_path.read_text(encoding='utf-8').split('\n'))

    if global_med is None:
        gm_path = CACHE / 'proteome_global_med.txt'
        if not gm_path.exists():
            raise FileNotFoundError('请先运行 train_val: python code/preprocess.py')
        global_med = float(gm_path.read_text(encoding='utf-8').strip())

    m, p = load_raw('test')

    # test metadata 没有 split_final，只做 keep 列对齐
    missing = [c for c in keep if c not in p.columns]
    if missing:
        print(f'  警告: test 中缺少 {len(missing)} 个蛋白列，将以 NaN 填充')
    p_aligned = p.reindex(columns=keep)

    normed, _ = normalize(p_aligned, keep, global_med=global_med)
    save_normed(normed, 'test')


# ── 入口 ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='蛋白质组预处理：log2 + 中位数归一化')
    ap.add_argument('--split', choices=['train_val', 'test'], default='train_val',
                    help='处理哪个 split（默认 train_val）')
    ap.add_argument('--all',  action='store_true',
                    help='同时处理 train_val 和 test')
    a = ap.parse_args()

    if a.all:
        gm = process_train_val()
        process_test(global_med=gm)
    elif a.split == 'train_val':
        process_train_val()
    else:
        process_test()

    print('\n预处理完成。后续步骤：')
    print('  common.py 的 load_normed() 已可直接读取归一化缓存。')
    print('  在 stage 脚本中将 C.load() 替换为 C.load_normed() 即可使用处理后数据。')


if __name__ == '__main__':
    main()
