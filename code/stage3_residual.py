"""阶段 3：残差分解训练 + 消融实验。

loss = mask-aware MSE + w_fc * fold-change loss（PDF 5.6.1）。
Δ 用训练集 matched control 计算——control 由元数据给出，非模型输出，
所以这不构成泄漏；评测服务器用的是它自己持有的 y_control。
"""
import argparse
import numpy as np
import pandas as pd
import torch

import common as C
import features as F
import metrics as M
from model_residual import ResidualDecomp, masked_mse, masked_fc_loss
from model_shared import SharedMultiHead
from stage0_baseline import context_group

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ALL_PARTS = ('shared', 'chem', 'smod', 'inter', 'calib')
SHARED_PARTS = ('abs', 'fc', 'calib')


def build(use_unimol=True):
    """加载数据、编码、构造训练所需张量。"""
    m, p = C.load('train_val')
    keep = C.keep_proteins(m, p)
    y, mask = C.to_log2(p, keep)
    tr = m['split_final'].eq('train').values
    isctl = C.is_control(m)
    ctl_tbl = C.control_table(m, y)

    enc = F.Encoder(use_unimol=use_unimol).fit(m[tr])
    blk = enc.blocks(m)
    blk['isctl'] = enc.is_control_feat(m)
    center = y.loc[tr].mean(axis=0).values.astype(np.float32)

    # matched control（相对 center 的偏移），供 FC loss 与评测共用
    ctl_all, hit = C.matched_control(m, m.index, ctl_tbl)
    ctl_off = np.where(np.isfinite(ctl_all), ctl_all - center, 0.0).astype(np.float32)
    ctl_ok = (np.isfinite(ctl_all) & hit[:, None]).astype(np.float32)

    Y = (np.nan_to_num(y.values, nan=0.0).astype(np.float32) - center)
    MK = mask.values.astype(np.float32)
    return dict(m=m, y=y, mask=mask, keep=keep, tr=tr, isctl=isctl,
                ctl_tbl=ctl_tbl, enc=enc, blk=blk, center=center,
                Y=Y, MK=MK, ctl_off=ctl_off, ctl_ok=ctl_ok)


def tens(d, rows):
    return {k: torch.tensor(d['blk'][k][rows], device=DEV)
            for k in ['ctx', 'strain', 'chem', 'obs', 'isctl']}


def train(d, a, use=None, verbose=True):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dims = {k: d['blk'][k].shape[1] for k in ['ctx', 'strain', 'chem', 'obs']}
    n_prot = int(d['keep'].sum())
    shared_arch = a.arch == 'shared'
    if use is None:
        use = SHARED_PARTS if shared_arch else ALL_PARTS
    if shared_arch:
        model = SharedMultiHead(dims, n_prot, a.hidden, a.emb, a.latent,
                                use=use).to(DEV)
    else:
        model = ResidualDecomp(dims, n_prot, a.hidden, a.emb, a.rank,
                               use=use).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    ti = np.where(d['tr'])[0]
    T = tens(d, ti)
    ty = torch.tensor(d['Y'][ti], device=DEV)
    tm = torch.tensor(d['MK'][ti], device=DEV)
    tc = torch.tensor(d['ctl_off'][ti], device=DEV)
    tk = torch.tensor(d['ctl_ok'][ti], device=DEV)

    m = d['m']
    vb = m.index[m['split_final'].eq('val_both') & ~d['isctl']]
    vpos = m.index.get_indexer(vb)
    V = tens(d, vpos)
    vctl, _ = C.matched_control(m, vb, d['ctl_tbl'])
    # best_state 预置为初始权重：即使全程 NaN 也不会崩在 load_state_dict
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best, bad = -np.inf, 0
    hist, stop_ep = [], a.epochs

    n = len(ti)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEV)
        agg = np.zeros(2)
        for s in range(0, n, a.batch):
            j = perm[s:s + a.batch]
            opt.zero_grad()
            pr, pt = model(T['ctx'][j], T['strain'][j], T['chem'][j],
                           T['obs'][j], T['isctl'][j], return_parts=True)
            l_mse = masked_mse(pr, ty[j], tm[j])
            # shared 架构下 fc 头直接预测 Δ；残差架构沿用总输出减 control
            fc_pred = pt['fc'] if shared_arch and 'fc' in use else pr - tc[j]
            l_fc = masked_fc_loss(fc_pred, ty[j] - tc[j], tm[j], tk[j])
            (l_mse + a.w_fc * l_fc).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg += np.array([l_mse.item(), l_fc.item()]) * len(j)
        sched.step()

        if ep % 10 == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                pv = model(V['ctx'], V['strain'], V['chem'], V['obs'],
                           V['isctl']).cpu().numpy() + d['center']
            fc = M.fc_pcc(d['y'].loc[vb].values, pv, d['mask'].loc[vb].values, vctl)
            if verbose:
                print(f'  epoch {ep:4d} | mMSE {agg[0]/n:.4f} | fc_loss {agg[1]/n:.4f}'
                      f' | val_both fc_pcc {fc:.4f} | lr {sched.get_last_lr()[0]:.2e}')
            hist.append({'epoch': ep, 'train_mse': agg[0] / n, 'train_fc_loss': agg[1] / n,
                         'val_both_fc_pcc': fc, 'lr': sched.get_last_lr()[0]})
            if fc > best:
                best, bad = fc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    if verbose:
                        print(f'  早停于 epoch {ep}')
                    stop_ep = ep
                    break
    model.load_state_dict(best_state)
    model.eval()
    best_ep = max(hist, key=lambda r: r['val_both_fc_pcc'])['epoch'] if hist else 0
    return model, best, best_state, {'history': hist, 'best_ep': best_ep,
                                     'stop_ep': stop_ep}


def eval_splits(d, model, splits, tag, rows=None, parts_report=False):
    rows = [] if rows is None else rows
    m = d['m']
    for sc in splits:
        idx = m.index[m['split_final'].eq(sc) & ~d['isctl']]
        pos = m.index.get_indexer(idx)
        B = tens(d, pos)
        with torch.no_grad():
            pred = model(B['ctx'], B['strain'], B['chem'], B['obs'],
                         B['isctl']).cpu().numpy() + d['center']
        ctl, _ = C.matched_control(m, idx, d['ctl_tbl'])
        rows.append((f'{tag}|{sc}', M.evaluate(
            d['y'].loc[idx].values, pred, d['mask'].loc[idx].values, ctl,
            context_group(m, idx),
            m.loc[idx, 'perturbation_no_concentration'].values)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--emb', type=int, default=64)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--latent', type=int, default=256,
                    help='shared 架构 trunk 输出维度')
    ap.add_argument('--arch', choices=['residual', 'shared'],
                    default='residual',
                    help='residual=分支分解（原架构）；shared=共享 encoder+多头')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--w_fc', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ablate', action='store_true', help='跑消融实验')
    ap.add_argument('--no_unimol', action='store_true',
                    help='化合物回落到 hash 编码（阶段 2 前的行为），用于 A/B 对照')
    a = ap.parse_args()

    d = build(use_unimol=not a.no_unimol)
    print(f'化合物编码: {"Uni-Mol" if d["enc"].use_unimol else "hash"}')
    print(f'设备 {DEV} | 输出 {int(d["keep"].sum())} 蛋白 | '
          f'ctx {d["blk"]["ctx"].shape[1]} strain {d["blk"]["strain"].shape[1]} '
          f'chem {d["blk"]["chem"].shape[1]} obs {d["blk"]["obs"].shape[1]}')
    print(f'matched control 覆盖训练行: '
          f'{100 * d["ctl_ok"][d["tr"]].any(1).mean():.1f}%\n')

    tag = '共享多头' if a.arch == 'shared' else '残差分解'
    print(f'=== 完整模型（{tag}）===')
    model, best, state, tr_log = train(d, a)
    rows = eval_splits(d, model, C.VAL_SPLITS, tag)
    print(f'\n=== 阶段 3 验证结果（val_both fc_pcc={best:.4f} @ epoch '
          f'{tr_log["best_ep"]}，停止于 {tr_log["stop_ep"]}）===')
    df = M.report(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    suf = '' if a.arch == 'residual' else f'_{a.arch}'
    if a.no_unimol:
        suf += '_hash'
    df.to_csv(C.OUT / f'stage3_validation{suf}.csv', encoding='utf-8-sig')
    hp = C.save_history(tr_log['history'], f'stage3_history{suf}.csv')
    ckpt = C.OUT / f'stage3_model{suf}.pt'
    torch.save({'state': state, 'center': d['center'],
                'keep': d['keep'][d['keep']].index.tolist(), 'args': vars(a),
                'dims': {k: d['blk'][k].shape[1]
                         for k in ['ctx', 'strain', 'chem', 'obs']},
                'best': best, **tr_log},
               ckpt)
    print(f'已保存: {ckpt}, {hp}')

    if a.ablate:
        print('\n=== 消融实验（逐个移除组件，看 val_both）===')
        if a.arch == 'shared':
            ab = [('完整', SHARED_PARTS), ('-fc头', ('abs', 'calib')),
                  ('-calib', ('abs', 'fc')), ('仅abs', ('abs',))]
        else:
            ab = [('完整', ALL_PARTS)]
            for p in ['chem', 'smod', 'inter', 'calib']:
                ab.append((f'-{p}', tuple(x for x in ALL_PARTS if x != p)))
            ab.append(('仅shared', ('shared',)))
        arows, ahist = [], []
        for name, use in ab:
            _, bfc, _, tl = train(d, a, use=use, verbose=False)
            arows.append((name, {'val_both_fc_pcc': bfc,
                                 'best_ep': tl['best_ep'], 'stop_ep': tl['stop_ep']}))
            ahist += [dict(变体=name, **r) for r in tl['history']]
            print(f'  {name:10s} val_both fc_pcc = {bfc:.4f}'
                  f'  (best @ {tl["best_ep"]}, 停止 {tl["stop_ep"]})')
        adf = M.report(arows)
        adf.to_csv(C.OUT / f'stage3_ablation{suf}.csv', encoding='utf-8-sig')
        C.save_history(ahist, f'stage3_ablation_history{suf}.csv')


if __name__ == '__main__':
    main()
