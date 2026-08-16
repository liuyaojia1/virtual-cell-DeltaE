"""生成最终提交文件。

用法:
  python code/generate_submission.py --model exp13_ensemble_best_config

支持:
  - 单模型: 从 .pt 文件加载
  - ensemble: 自动检测 ensemble_seeds 配置并加载多个 checkpoint
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import common as C
import features as F
from model_residual import ResidualDecomp
from model_shared import SharedMultiHead

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, d):
    """从 checkpoint 加载模型"""
    ck = torch.load(model_path, map_location=DEV, weights_only=False)
    cfg = ck['cfg']

    blk = d['blk']
    dims = {k: blk[k].shape[1] for k in ['ctx', 'strain', 'chem', 'obs']}
    n_prot = int(d['keep'].sum())

    if cfg.get('arch', 'residual') == 'shared':
        model = SharedMultiHead(dims, n_prot, cfg['hidden'],
                               cfg['emb'], cfg.get('latent', 256)).to(DEV)
    else:
        model = ResidualDecomp(dims, n_prot, cfg['hidden'],
                              cfg['emb'], cfg['rank']).to(DEV)

    model.load_state_dict(ck['state'])
    model.eval()
    return model, ck['center']


def predict_test(model, d, center):
    """在测试集上预测"""
    m_test, p_test = C.load('test')
    y_test, mask_test = C.to_log2(p_test, d['keep'])

    pos = m_test.index
    blk = d['enc'].blocks(m_test)

    X = {k: torch.tensor(blk[k], device=DEV) for k in ['ctx', 'strain', 'chem', 'obs']}
    X['isctl'] = torch.tensor(d['enc'].is_control_feat(m_test), device=DEV)

    with torch.no_grad():
        pred = model(X['ctx'], X['strain'], X['chem'], X['obs'], X['isctl'])
        pred = pred.cpu().numpy() + center

    return pred, m_test, y_test, mask_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='exp13_ensemble_best_config',
                    help='模型名称（exp ID）')
    ap.add_argument('--out', type=str, default=None,
                    help='输出文件名（默认: prediction_{model}.csv）')
    a = ap.parse_args()

    model_name = a.model
    out_name = a.out or f'prediction_{model_name}.csv'

    print(f'加载模型: {model_name}')

    # 构建数据
    from stage4_exp import build_data

    # 检测配置
    model_path = C.OUT / f'{model_name}_model.pt'

    if not model_path.exists():
        print(f'错误: 模型文件不存在 {model_path}')
        return

    # 加载配置检测是否是 ensemble
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']

    # 根据配置重建数据
    n_mol = cfg.get('n_mol', 32)
    use_rdkit = cfg.get('use_rdkit', False)

    print(f'配置: n_mol={n_mol}, use_rdkit={use_rdkit}')
    d = build_data(n_mol=n_mol, use_rdkit=use_rdkit)

    # 检查是否是 ensemble
    if 'ensemble_seeds' in cfg:
        seeds = cfg['ensemble_seeds']
        print(f'检测到 ensemble 配置: {len(seeds)} seeds')

        # 加载所有 seed 的模型
        models = []
        for seed in seeds:
            seed_path = C.OUT / f'{model_name}_seed{seed}.pt'
            if seed_path.exists():
                print(f'  加载 seed {seed}')
                model, center = load_model(seed_path, d)
                models.append(model)
            else:
                print(f'  警告: seed {seed} 模型不存在，跳过')

        if not models:
            print('错误: 没有找到任何 ensemble seed 模型')
            return

        print(f'成功加载 {len(models)}/{len(seeds)} 个模型，开始预测...')

        # Ensemble 预测
        m_test, p_test = C.load('test')
        y_test, mask_test = C.to_log2(p_test, d['keep'])
        pos = m_test.index
        blk = d['enc'].blocks(m_test)
        X = {k: torch.tensor(blk[k], device=DEV) for k in ['ctx', 'strain', 'chem', 'obs']}
        X['isctl'] = torch.tensor(d['enc'].is_control_feat(m_test), device=DEV)

        preds = []
        with torch.no_grad():
            for model in models:
                pred_i = model(X['ctx'], X['strain'], X['chem'], X['obs'], X['isctl'])
                preds.append(pred_i.cpu().numpy() + center)

        pred = np.mean(preds, axis=0)  # Average predictions
        print(f'Ensemble 平均完成')
    else:
        model, center = load_model(model_path, d)
        pred, m_test, y_test, mask_test = predict_test(model, d, center)

    # 生成提交文件
    sub = pd.DataFrame(pred, index=m_test.index, columns=d['keep'][d['keep']].index)
    sub.index.name = 'sample_ID'

    # 验证
    assert not sub.isna().any().any(), '存在 NA'
    assert np.isfinite(sub.values).all(), '存在 inf'

    out_path = C.OUT / out_name
    sub.to_csv(out_path, encoding='utf-8')

    print(f'\n提交文件已生成: {out_path}')
    print(f'样本数: {len(sub)} | 蛋白列数: {sub.shape[1]}')
    print(f'预测值范围: [{pred.min():.2f}, {pred.max():.2f}]')
    print(f'尺度: log2 (声明 prediction_scale=log2)')

    # 本地评测
    print('\n本地测试集评测（如果有真值）:')
    try:
        import metrics as M
        from stage0_baseline import context_group

        isctl = C.is_control(m_test)
        ctl_tbl = C.control_table(m_test, y_test)

        for sc in C.TEST_SPLITS:
            sel = m_test['split_final'].eq(sc) & ~isctl
            idx = m_test.index[sel]
            pos = m_test.index.get_indexer(idx)
            yy, mm = y_test.loc[idx].values, mask_test.loc[idx].values
            ctl, _ = C.matched_control(m_test, idx, ctl_tbl)
            ctx = context_group(m_test, idx)
            chem = m_test.loc[idx, 'perturbation_no_concentration'].values

            res = M.evaluate(yy, pred[pos], mm, ctl, ctx, chem)
            print(f'  {sc}: fc_pcc={res["fc_pcc"]:.4f}, '
                  f'per_prot_r2={res["per_protein_r2"]:.4f}, '
                  f'global_r2={res["global_r2"]:.4f}')
    except Exception as e:
        print(f'本地评测失败: {e}')


if __name__ == '__main__':
    main()
