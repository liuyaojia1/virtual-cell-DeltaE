"""mask-aware 评测指标。

对齐 PDF 5.1.2 的评分逻辑：Global R² 区分度低，真正拉开差距的是
逐蛋白 R²、fold change PCC 与两类残差指标（合计 65%）。
Δ 用真实 matched control 计算：Δ = y_treat - y_control。
"""
import numpy as np
import pandas as pd


def _valid(mk, pred):
    return mk & np.isfinite(pred)


def global_r2(y, pred, mk):
    v = _valid(mk, pred)
    e = np.where(v, y - np.where(v, pred, 0.0), 0.0)
    mu = y[v].mean()
    return 1 - (e ** 2).sum() / (np.where(v, y - mu, 0.0) ** 2).sum()


def per_protein_r2(y, pred, mk, min_n=10):
    """逐蛋白 R² 的中位数——均值基线下为负，control 基线可达 0.7+。"""
    v = _valid(mk, pred)
    out = []
    p2 = np.broadcast_to(pred, y.shape)
    for j in range(y.shape[1]):
        k = v[:, j]
        if k.sum() < min_n:
            continue
        t = y[k, j]
        den = ((t - t.mean()) ** 2).sum()
        if den > 0:
            out.append(1 - ((t - p2[k, j]) ** 2).sum() / den)
    return float(np.median(out)) if out else np.nan


def _pcc_rows(a, b, v, min_n=10):
    """逐样本 Pearson，返回均值。"""
    out = []
    for i in range(a.shape[0]):
        k = v[i]
        if k.sum() < min_n:
            continue
        x, y_ = a[i, k], b[i, k]
        if x.std() > 0 and y_.std() > 0:
            out.append(np.corrcoef(x, y_)[0, 1])
    return float(np.mean(out)) if out else np.nan


def fc_pcc(y, pred, mk, ctl):
    """fold change PCC（模块 1，25%）：Δ_pred vs Δ_true，逐样本相关。"""
    v = _valid(mk, pred) & np.isfinite(ctl)
    return _pcc_rows(pred - ctl, y - ctl, v)


def residual_pcc(y, pred, mk, ctl, groups):
    """残差 PCC：Δ 去掉分组均值后的特异响应。

    groups 为分组标签数组——按上下文分组对应模块 3（20%），
    按化合物分组对应模块 4（20%）。
    """
    v = _valid(mk, pred) & np.isfinite(ctl)
    dp = np.where(v, pred - ctl, np.nan)
    dt = np.where(v, y - ctl, np.nan)
    g = pd.Series(np.asarray(groups))
    rp, rt = dp.copy(), dt.copy()
    for _, idx in g.groupby(g).groups.items():
        i = g.index.get_indexer(idx)
        if len(i) < 2:
            rp[i], rt[i] = np.nan, np.nan
            continue
        with np.errstate(invalid='ignore'):
            rp[i] -= np.nanmean(dp[i], axis=0)
            rt[i] -= np.nanmean(dt[i], axis=0)
    vv = np.isfinite(rp) & np.isfinite(rt)
    return _pcc_rows(np.nan_to_num(rp), np.nan_to_num(rt), vv)


def evaluate(y, pred, mk, ctl=None, ctx_groups=None, chem_groups=None):
    r = {'n': int(y.shape[0]),
         'global_r2': global_r2(y, pred, mk),
         'per_protein_r2': per_protein_r2(y, pred, mk)}
    if ctl is not None:
        r['fc_pcc'] = fc_pcc(y, pred, mk, ctl)
        if ctx_groups is not None:
            r['resid_ctx_pcc'] = residual_pcc(y, pred, mk, ctl, ctx_groups)
        if chem_groups is not None:
            r['resid_chem_pcc'] = residual_pcc(y, pred, mk, ctl, chem_groups)
    return r


def report(rows):
    """rows: list[(name, dict)] → 打印对照表。"""
    df = pd.DataFrame({n: v for n, v in rows}).T
    if 'n' in df:
        df['n'] = df['n'].astype(int)
    print(df.to_string(float_format=lambda x: f'{x:.4f}'))
    return df
