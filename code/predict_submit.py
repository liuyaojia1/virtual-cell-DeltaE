"""生成 prediction.csv 并在测试集上本地评测。

尺度纪律（PDF 5.2.2）：提交必须为 log2 intensity，提交信息中声明
prediction_scale=log2。测试集 proteome 含真值，故可本地打分。
"""
import argparse
import numpy as np
import pandas as pd
import torch

import common as C
import features as F
import metrics as M
from stage0_baseline import context_group
from stage1_mlp import ConditionMLP, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', action='store_true',
                    help='后处理校准：pred - control_mean + global_mean（PDF 5.6.3 技巧 1）')
    a = ap.parse_args()

    ck = torch.load(C.OUT / 'stage1_model.pt', map_location=DEV, weights_only=False)
    center, prot_cols = ck['center'], ck['keep']

    m_tr, p_tr = C.load('train_val')
    keep = pd.Series(False, index=p_tr.columns)
    keep[prot_cols] = True
    tr = m_tr['split_final'].eq('train').values
    # 编码器必须与 checkpoint 训练时一致，否则输入维度对不上
    enc = F.Encoder(use_unimol=not ck['args'].get('no_unimol', False)).fit(m_tr[tr])
    print(f'化合物编码: {"Uni-Mol" if enc.use_unimol else "hash"}')

    m, p = C.load('test')
    y, mask = C.to_log2(p, keep)
    assert list(y.columns) == list(prot_cols), '蛋白列顺序与训练不一致'
    Xb, Xo = enc.bio(m), enc.obs(m)

    model = ConditionMLP(Xb.shape[1], Xo.shape[1], len(prot_cols),
                         ck['args']['hidden']).to(DEV)
    model.load_state_dict(ck['state'])
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xb, device=DEV),
                     torch.tensor(Xo, device=DEV)).cpu().numpy() + center

    if a.calibrate:
        ctl_tbl = C.control_table(m, y)   # 测试集自带对照样本
        cm, hit = C.matched_control(m, m.index, ctl_tbl)
        gm = center
        adj = np.where(np.isfinite(cm), pred - cm + gm, pred)
        pred = np.where(hit[:, None], adj, pred)
        print(f'后处理校准: {int(hit.sum())}/{len(m)} 样本命中 matched control')

    isctl = C.is_control(m)
    ctl_tbl = C.control_table(m, y)
    rows = []
    for sc in C.TEST_SPLITS:
        sel = m['split_final'].eq(sc) & ~isctl
        idx = m.index[sel]
        pos = m.index.get_indexer(idx)
        yy, mm = y.loc[idx].values, mask.loc[idx].values
        ctl, _ = C.matched_control(m, idx, ctl_tbl)
        ctx = context_group(m, idx)
        chem = m.loc[idx, 'perturbation_no_concentration'].values
        rows.append((f'均值|{sc}', M.evaluate(
            yy, np.broadcast_to(center, yy.shape), mm, ctl, ctx, chem)))
        rows.append((f'MLP|{sc}', M.evaluate(yy, pred[pos], mm, ctl, ctx, chem)))
        rows.append((f'control|{sc}', M.evaluate(yy, ctl, mm)))

    print('\n=== 测试集本地评测 ===')
    df = M.report(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    tag = '_calib' if a.calibrate else ''
    df.to_csv(C.OUT / f'test_metrics{tag}.csv', encoding='utf-8-sig')

    sub = pd.DataFrame(pred, index=m.index, columns=prot_cols)
    sub.index.name = 'sample_ID'
    assert not sub.isna().any().any(), '存在 NA'
    assert np.isfinite(sub.values).all(), '存在 inf'
    out = C.OUT / f'prediction{tag}.csv'
    sub.to_csv(out, encoding='utf-8')
    print(f'\n提交文件: {out}')
    print(f'样本数 {len(sub)} | 蛋白列数 {sub.shape[1]} | 尺度 log2 '
          f'(声明 prediction_scale=log2)')
    print(f'预测值范围 [{pred.min():.2f}, {pred.max():.2f}]  校验通过：无 NA、无 inf')


if __name__ == '__main__':
    main()
