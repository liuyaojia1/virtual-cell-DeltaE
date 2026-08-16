"""阶段 0：统计基线（蛋白均值 / matched control）+ 分场景诊断。

用途：验证数据流水线、对齐 PDF 4.2.3 的诊断表。
所有建模方案必须超越 matched control，否则模型无意义。
"""
import numpy as np
import pandas as pd

import common as C
import metrics as M


def context_group(m, rows):
    """上下文分组：同菌株/培养基/温度/时间（不含化合物）。"""
    return m.loc[rows, ['Strains', 'Medium', 'Temperature', 'pert_time']] \
            .astype(str).agg('|'.join, axis=1).values


def main():
    m, p = C.load('train_val')
    keep = C.keep_proteins(m, p)
    y, mask = C.to_log2(p, keep)
    tr = m['split_final'].eq('train').values
    ctl_tbl = C.control_table(m, y)
    isctl = C.is_control(m)

    print(f'蛋白过滤: {p.shape[1]} → {int(keep.sum())} (缺失率 < {C.MISSING_THR}, 仅训练行)')
    print(f'有效观测: {100 * mask.values.mean():.1f}%   对照样本: {int(isctl.sum())}')
    print(f'matched control 分组数: {len(ctl_tbl)}\n')

    protein_mean = y.loc[tr].mean(axis=0).values

    rows = []
    for sc in C.VAL_SPLITS:
        idx = m.index[m['split_final'].eq(sc) & ~isctl]
        yy, mm = y.loc[idx].values, mask.loc[idx].values
        ctl, hit = C.matched_control(m, idx, ctl_tbl)
        ctx, chem = context_group(m, idx), m.loc[idx, 'perturbation_no_concentration'].values
        print(f'{sc}: {len(idx)} 处理样本, matched control 命中 {int(hit.sum())}')

        rows.append((f'均值|{sc}', M.evaluate(
            yy, np.broadcast_to(protein_mean, yy.shape), mm, ctl, ctx, chem)))
        rows.append((f'control|{sc}', M.evaluate(yy, ctl, mm, ctl, ctx, chem)))

    print('\n=== 分场景诊断（PDF 4.2.3 对照）===')
    df = M.report(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / 'stage0_diagnostics.csv', encoding='utf-8-sig')
    print(f'\n已保存: {C.OUT / "stage0_diagnostics.csv"}')
    print('注：control 基线的 fc/残差指标恒等于自身 Δ，仅作上界参考。')


if __name__ == '__main__':
    main()
