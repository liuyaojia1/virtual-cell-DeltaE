"""用 Uni-Mol 预训练模型为化合物生成 512 维表征，缓存到 缓存/unimol_repr.npz。

盐型处理：Uni-Mol 需要单一分子的 3D 构象，多组分 SMILES（盐/水合物）
取最大有机片段作为活性部分，反离子丢弃。Cisplatin 这类含金属配合物
保留原样，若构象生成失败则回落到 RDKit 指纹（见 features 中的 fallback 标记）。

网络：HuggingFace 直连不通时用 HF_ENDPOINT=https://hf-mirror.com，
并且必须设 HF_HUB_DISABLE_XET=1（xet 传输后端不在镜像代理范围内，返回 401）。
本脚本已在进程内设好这两项，无需外部导出。
"""
import os
import warnings

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import common as C

RDLogger.DisableLog('rdApp.*')
OUT_NPZ = C.CACHE / 'unimol_repr.npz'


def largest_fragment(smi):
    """多组分 SMILES 取原子数最大的片段（活性成分），单组分原样返回。"""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, 0
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return Chem.MolToSmiles(m), 1
    big = max(frags, key=lambda f: f.GetNumAtoms())
    return Chem.MolToSmiles(big), len(frags)


def main():
    df = pd.read_csv(C.CACHE / 'smiles.csv')
    df = df[df.smiles.notna()].copy()

    prep, names, nfrag = [], [], []
    for _, r in df.iterrows():
        s, nf = largest_fragment(r.smiles)
        if s is None:
            print(f'  跳过（RDKit 解析失败）: {r["name"]}')
            continue
        names.append(r['name'])
        prep.append(s)
        nfrag.append(nf)
    print(f'待编码 {len(prep)} 个化合物，其中 {sum(1 for n in nfrag if n > 1)} 个为多组分（已取最大片段）')

    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type='molecule', remove_hs=False,
                     model_name='unimolv1', model_size='84m')

    # 逐个编码：构象生成偶发失败，单点失败不应拖垮整批
    vecs, ok_names, failed = [], [], []
    for i, (nm, smi) in enumerate(zip(names, prep), 1):
        try:
            r = clf.get_repr([smi], return_atomic_reprs=False)
            v = np.asarray(r, dtype=np.float32).reshape(-1)
            if v.size == 0 or not np.isfinite(v).all():
                raise ValueError('非有限值')
            vecs.append(v)
            ok_names.append(nm)
        except Exception as e:
            failed.append((nm, f'{type(e).__name__}: {str(e)[:80]}'))
        if i % 10 == 0 or i == len(names):
            print(f'  进度 {i}/{len(names)}  成功 {len(ok_names)}  失败 {len(failed)}')

    if not vecs:
        raise SystemExit('全部编码失败，检查网络与权重')

    X = np.vstack(vecs).astype(np.float32)
    np.savez_compressed(OUT_NPZ, names=np.array(ok_names, dtype=object),
                        X=X, smiles=np.array(
                            [prep[names.index(n)] for n in ok_names], dtype=object))
    print(f'\n表征矩阵 {X.shape}，已保存: {OUT_NPZ}')
    if failed:
        print('编码失败（将回落到 RDKit 指纹）:')
        for n, e in failed:
            print(f'   - {n}: {e}')

    # 化学合理性抽查：结构相近的分子在表征空间应更接近
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    idx = {n: i for i, n in enumerate(ok_names)}
    pairs = [('Tamoxifen', '4-Hydroxytamoxifen'),
             ('Fluconazole', 'Clotrimazole'),
             ('Neomycin B', 'G418'),
             ('Amphotericin B', 'Nystatin dihydrate'),
             ('Tamoxifen', 'H2O2')]
    print('\n余弦相似度抽查（前四对应偏高，末对应偏低）:')
    for a, b in pairs:
        if a in idx and b in idx:
            print(f'  {a:22s} ~ {b:22s} {S[idx[a], idx[b]]: .3f}')
    off = S[~np.eye(len(S), dtype=bool)]
    print(f'  全体非对角相似度: 均值 {off.mean():.3f}  中位 {np.median(off):.3f}')


if __name__ == '__main__':
    main()
