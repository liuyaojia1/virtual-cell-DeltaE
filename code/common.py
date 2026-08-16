"""数据加载与预处理：sample_ID 对齐 → 缺失过滤 → log2 → mask。

字段名以真实数据为准（Strains / perturbation_no_concentration / pert_time ...），
与解题思路 PDF 的示例代码不同。
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 路径以仓库根目录为基准（code/ 的上一级），可用环境变量 VC_ROOT 覆盖
ROOT = Path(os.environ.get('VC_ROOT', Path(__file__).resolve().parent.parent))
DATA = ROOT / 'dataset' / 'input'
CACHE = ROOT / 'cache'
OUT = ROOT / 'output'

# 元数据字段分为两类，不可混用（PDF 3.1.2 / 5.3.3）
BIO_COLS = ['Strains', 'perturbation_no_concentration', 'Medium', 'Temperature', 'pert_time']
OBS_COLS = ['data_source', 'instrument', 'Yeast_cell_plate']
# matched control 的匹配键：同来源/仪器/板/菌株/培养基/温度/时间
CTRL_KEY = ['data_source', 'instrument', 'Yeast_cell_plate', 'Strains',
            'Medium', 'Temperature', 'pert_time']
CONTROLS = ['Water', 'DMSO']
# 蛋白过滤：仅用训练行计算缺失率。
# PDF 声明 5,243 → 4,232，但 0.80 阈值实际得 4,422、0.70 得 4,235，均对不上。
# 4,232 对应缺失率 <= 0.6965，边界无并列，故按「缺失率最低的 N 个」精确选取。
N_KEEP = 4232
MISSING_THR = None   # 设为浮点数则改用阈值规则，N_KEEP 失效

VAL_SPLITS = ['val_strain_only', 'val_chem_only', 'val_both', 'val_time']
TEST_SPLITS = ['test_strain_only', 'test_chem_only', 'test_both', 'test_time']


def _load_proteome(split):
    """读 raw proteome，带 npy 缓存（原始 CSV 276MB，首次约 1~2 分钟）。"""
    fn = f'WAYB_WAYC_proteome_raw_{split}.csv'
    npy, cols = CACHE / f'prot_{split}.npy', CACHE / f'prot_{split}_cols.txt'
    ids = CACHE / f'prot_{split}_ids.txt'
    if npy.exists():
        arr = np.load(npy)
        c = cols.read_text(encoding='utf-8').split('\n')
        i = ids.read_text(encoding='utf-8').split('\n')
        return pd.DataFrame(arr, index=pd.Index(i, name='sample_ID'), columns=c)
    df = pd.read_csv(DATA / fn).set_index('sample_ID')
    df = df.astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(npy, df.values)
    cols.write_text('\n'.join(df.columns), encoding='utf-8')
    ids.write_text('\n'.join(df.index), encoding='utf-8')
    return df


def load(split):
    """返回 (metadata, proteome)，行序严格按 sample_ID 对齐。"""
    fn = ('WAYB_WAYC_metadata_train_val(1).csv' if split == 'train_val'
          else 'WAYB_WAYC_metadata_test(1).csv')
    m = pd.read_csv(DATA / fn).set_index('sample_ID')
    p = _load_proteome(split).reindex(m.index)
    assert p.notna().any(axis=1).all(), 'proteome 存在无法匹配的 sample_ID'
    return m, p


def keep_proteins(m_tr, p_tr, thr=MISSING_THR, n_keep=N_KEEP):
    """仅用 split_final=='train' 的行计算缺失率，返回布尔 Series（保持原列序）。

    thr 给定时用阈值规则；否则取缺失率最低的 n_keep 个蛋白。
    并列时以原始列序为稳定 tie-break。
    """
    tr = m_tr['split_final'].eq('train').values
    mr = p_tr.loc[tr].isna().mean(axis=0)
    if thr is not None:
        return mr < thr
    order = np.lexsort((np.arange(len(mr)), mr.values))[:n_keep]
    keep = pd.Series(False, index=mr.index)
    keep.iloc[order] = True
    return keep


def to_log2(p, keep, median_norm=True):
    """log2 转换 + 跨样本中位数归一化，返回 (y, mask)。

    median_norm=True（默认）：在全部蛋白上计算每样本 log2 中位数，
    减去行中位数再加全局中位数，校正仪器/上样量引起的系统偏移。
    归一化在蛋白过滤前的全矩阵上计算，之后再取 keep 列，保证中位数估计稳健。
    """
    # 全蛋白 log2（用于归一化，非正值/缺失保持 NaN）
    log2_full = np.log2(p.where(lambda d: d > 0))
    if median_norm:
        global_med = float(np.nanmedian(log2_full.values))
        row_med    = log2_full.median(axis=1)          # 每样本中位数
        log2_full  = log2_full.sub(row_med, axis=0).add(global_med)
    y = log2_full.loc[:, keep]
    return y, y.notna()


def is_control(m):
    return m['perturbation_no_concentration'].isin(CONTROLS)


def control_table(m, y):
    """按 CTRL_KEY 分组的对照 log2 均值表。"""
    c = is_control(m).values
    return y[c].groupby([m.loc[c, k] for k in CTRL_KEY]).mean()


def matched_control(m, rows, ctl):
    """给定样本行，查其 matched control 向量；未命中的行返回全 NaN。"""
    key = pd.MultiIndex.from_frame(m.loc[rows, CTRL_KEY])
    hit = key.isin(ctl.index)
    out = np.full((len(rows), ctl.shape[1]), np.nan, dtype=np.float32)
    if hit.any():
        out[hit] = ctl.loc[key[hit]].values
    return out, hit


def save_history(rows, name):
    """训练轨迹落盘。rows 为 dict 列表，列以首行为准。"""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    return path


def time_cyc(pert_time):
    """时间周期编码。真实数据单位为分钟：15/30/60/90/120/240。"""
    t = pert_time.astype(float).values
    n = 2 * np.pi * t / t.max()
    return np.stack([np.sin(n), np.cos(n)], axis=1)
