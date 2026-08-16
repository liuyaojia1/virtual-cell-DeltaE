"""条件编码：生物条件 + 观测过程分两路（PDF 5.3.3 / 5.4.3）。

关键纪律：观测过程字段（仪器/板/来源）不进生物编码，否则模型会把
仪器偏好当生物规律学走，评测时仪器分布一变就崩。
"""
import hashlib
import numpy as np
import pandas as pd

import common as C

UNIMOL_NPZ = C.CACHE / 'unimol_repr.npz'


def load_unimol():
    """读取 build_unimol.py 产出的 512 维表征。缺失则返回 None（自动回落 hash）。"""
    if not UNIMOL_NPZ.exists():
        return None
    z = np.load(UNIMOL_NPZ, allow_pickle=True)
    return {str(n): v for n, v in zip(z['names'], z['X'])}


def _onehot(s, levels):
    idx = pd.Categorical(s, categories=levels).codes
    out = np.zeros((len(s), len(levels)), dtype=np.float32)
    ok = idx >= 0
    out[np.arange(len(s))[ok], idx[ok]] = 1.0   # 未见实体 → 全零
    return out


def hash_encode(name, dim=32):
    """化合物名 hash：新化合物也有非零特征，比 one-hot 全零好（PDF 3.1.5c 特征 4）。"""
    h = hashlib.md5(str(name).encode()).hexdigest()
    h = (h * ((dim * 2) // len(h) + 1))[:dim * 2]
    return np.array([int(h[i * 2:i * 2 + 2], 16) / 255.0 for i in range(dim)],
                    dtype=np.float32)


class Encoder:
    """在训练集上 fit，对任意行 transform。

    use_unimol=True 时化合物走 Uni-Mol 预训练表征（可迁移到未见化合物），
    否则回落到 one-hot + hash（hash 对未见化合物是任意值，学不到化学规律）。
    """

    def __init__(self, use_unimol=True, n_mol=32):
        self.use_unimol = use_unimol
        self.n_mol = n_mol
        self._mol = None

    def _fit_mol(self, m_train):
        """在训练集出现的化合物上 fit 标准化 + PCA，避免用未见化合物的分布信息。"""
        tab = load_unimol()
        if tab is None:
            self.use_unimol = False
            return
        col = 'perturbation_no_concentration'
        tr_names = [n for n in sorted(m_train[col].astype(str).unique()) if n in tab]
        if len(tr_names) < 4:
            self.use_unimol = False
            return
        X = np.vstack([tab[n] for n in tr_names]).astype(np.float64)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        mu = X.mean(0)
        Xc = X - mu
        # 训练化合物数 (~40) 远小于 512 维，取前 k 个主成分即可无损覆盖其张成空间
        k = int(min(self.n_mol, len(tr_names) - 1, Xc.shape[1]))
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:k].T
        Z = Xc @ W
        self._mol = dict(tab=tab, mu=mu, W=W,
                         sd=(Z.std(0) + 1e-6), k=k)

    def mol(self, m):
        """化合物 Uni-Mol 表征 + 有效性标记。未知/非分子扰动 → 零向量 + flag=0。"""
        col = 'perturbation_no_concentration'
        p = self._mol
        n = len(m)
        out = np.zeros((n, p['k'] + 1), dtype=np.float32)
        names = m[col].astype(str).values
        have = np.array([x in p['tab'] for x in names])
        if have.any():
            X = np.vstack([p['tab'][x] for x in names[have]]).astype(np.float64)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            Z = ((X - p['mu']) @ p['W']) / p['sd']
            out[have, :p['k']] = Z.astype(np.float32)
        out[:, p['k']] = have.astype(np.float32)
        return out

    def fit(self, m_train):
        if self.use_unimol:
            self._fit_mol(m_train)
        self.levels = {c: sorted(m_train[c].astype(str).unique()) for c in C.BIO_COLS}
        self.obs_levels = {c: sorted(m_train[c].astype(str).unique()) for c in C.OBS_COLS}
        # 条件交叉特征（PDF 3.1.5c 特征 1）
        self.cross = {}
        for a, b in [('Strains', 'Medium'), ('perturbation_no_concentration', 'Temperature')]:
            key = f'{a}×{b}'
            self.cross[key] = (a, b, sorted((m_train[a].astype(str) + '_'
                                             + m_train[b].astype(str)).unique()))
        self.tmax = float(m_train['pert_time'].astype(float).max())
        return self

    def bio(self, m):
        parts = [_onehot(m[c].astype(str), self.levels[c]) for c in C.BIO_COLS]
        for a, b, lv in self.cross.values():
            parts.append(_onehot(m[a].astype(str) + '_' + m[b].astype(str), lv))
        t = m['pert_time'].astype(float).values / self.tmax
        n = 2 * np.pi * t
        parts.append(np.stack([np.sin(n), np.cos(n), t], axis=1).astype(np.float32))
        if self.use_unimol:
            parts.append(self.mol(m))
        else:
            parts.append(np.stack([hash_encode(x) for x in
                                   m['perturbation_no_concentration']]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def obs(self, m):
        return np.concatenate(
            [_onehot(m[c].astype(str), self.obs_levels[c]) for c in C.OBS_COLS],
            axis=1).astype(np.float32)

    # ---- 分块编码：供阶段 3 残差分解使用 ----
    # 每块只喂给对应的分支，组件的可辨识性靠输入隔离保证。

    CTX_COLS = ['Medium', 'Temperature']

    def ctx(self, m):
        """上下文（不含化合物、不含菌株）：培养基 + 温度 + 时间周期。"""
        parts = [_onehot(m[c].astype(str), self.levels[c]) for c in self.CTX_COLS]
        t = m['pert_time'].astype(float).values / self.tmax
        n = 2 * np.pi * t
        parts.append(np.stack([np.sin(n), np.cos(n), t], axis=1).astype(np.float32))
        parts.append(_onehot(m['pert_time'].astype(str), self.levels['pert_time']))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def strain(self, m):
        """菌株：one-hot（新菌株全零）。阶段 2 将替换为基因组表征。"""
        return _onehot(m['Strains'].astype(str), self.levels['Strains'])

    def chem(self, m):
        """化合物：one-hot + Uni-Mol 表征。

        one-hot 对已见化合物更强，Uni-Mol 则是未见化合物唯一的可迁移信号——
        阶段 3 的 chem/smod 分支在 val_both 上失效正是因为缺后者。
        """
        col = 'perturbation_no_concentration'
        parts = [_onehot(m[col].astype(str), self.levels[col])]
        parts.append(self.mol(m) if self.use_unimol
                     else np.stack([hash_encode(x) for x in m[col]]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def is_control_feat(self, m):
        """对照标记：让共享分支知道该样本是否为无扰动基线。"""
        return C.is_control(m).values.astype(np.float32)[:, None]

    def blocks(self, m):
        return {'ctx': self.ctx(m), 'strain': self.strain(m),
                'chem': self.chem(m), 'obs': self.obs(m)}
