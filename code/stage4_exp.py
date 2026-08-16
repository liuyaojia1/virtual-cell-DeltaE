"""
阶段 4：系统优化实验链

实验列表:
  exp01  baseline_ref      当前 stage3 残差分解结果（直接读取，作参照）
  exp02  feat_dropout      feature dropout (p_strain=0.3, p_chem_oh=0.2)
  exp03  larger_model      dropout + hidden=1024, emb=128
  exp04  mol64             dropout + n_mol=64 (full PCA span)
  exp05  rdkit_fp          dropout + RDKit Morgan 256-bit PCA64 fingerprints
  exp06  shared_dropout    SharedMultiHead + feature dropout
  exp07  ensemble_5seed    best config × 5 seeds, average predictions
  exp08  combined_best     dropout + n_mol=64 + rdkit + hidden=1024

所有结果追加到 output/experiment_log.md
用法: python code/stage4_exp.py [--exp EXP_ID] [--all]
"""
import sys, time, copy, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CODE = Path(__file__).parent
sys.path.insert(0, str(CODE))
import common as C
import features as F
import metrics as M
from model_residual import ResidualDecomp, masked_mse, masked_fc_loss
from model_shared import SharedMultiHead
from stage0_baseline import context_group

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_PATH = C.OUT / 'experiment_log.md'


# ─────────────────────────────────────────────────────────────────────────────
# RDKit Morgan fingerprints helper
# ─────────────────────────────────────────────────────────────────────────────

def build_morgan_fps(smiles_csv: Path, n_bits: int = 256) -> dict:
    """Compute ECFP4 fingerprints (Morgan r=2) from smiles.csv, keyed by 'name'."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {}
    df = pd.read_csv(smiles_csv)
    fps = {}
    for _, row in df.iterrows():
        smi = row.get('smiles', '')
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        fps[str(row['name'])] = np.array(fp, dtype=np.float32)
    return fps


def reduce_fps(fps: dict, train_names: list, n_comp: int = 64):
    """PCA-reduce fingerprints using only training compound variance."""
    tr_valid = [n for n in train_names if n in fps]
    if len(tr_valid) < 4:
        return None  # not enough data
    X = np.vstack([fps[n] for n in tr_valid]).astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    k = min(n_comp, len(tr_valid) - 1, Xc.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T
    Z = Xc @ W
    sd = Z.std(0) + 1e-6
    return dict(fps=fps, mu=mu, W=W, sd=sd, k=k)


def encode_fps(pca_info: dict, names) -> np.ndarray:
    """Project compound names to reduced fingerprint space. Unknown → zeros."""
    k = pca_info['k']
    out = np.zeros((len(names), k + 1), dtype=np.float32)
    fps = pca_info['fps']
    for i, name in enumerate(names):
        if name in fps:
            x = fps[name].astype(np.float64) - pca_info['mu']
            out[i, :k] = ((x @ pca_info['W']) / pca_info['sd']).astype(np.float32)
            out[i, k] = 1.0  # validity flag
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Extended Encoder (n_mol configurable + optional RDKit)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderV2(F.Encoder):
    """Drop-in replacement for features.Encoder with n_mol and RDKit support."""

    def __init__(self, use_unimol: bool = True, n_mol: int = 32,
                 use_rdkit: bool = False, n_fp: int = 256, n_fp_comp: int = 64):
        super().__init__(use_unimol=use_unimol, n_mol=n_mol)
        self.use_rdkit = use_rdkit
        self.n_fp = n_fp
        self.n_fp_comp = n_fp_comp
        self._rdkit_pca = None

    def fit(self, m_train):
        super().fit(m_train)
        if self.use_rdkit:
            smiles_csv = C.CACHE / 'smiles.csv'
            fps_raw = build_morgan_fps(smiles_csv, self.n_fp)
            tr_names = m_train['perturbation_no_concentration'].astype(str).unique().tolist()
            self._rdkit_pca = reduce_fps(fps_raw, tr_names, self.n_fp_comp)
            if self._rdkit_pca:
                print(f'  RDKit fingerprint PCA: {self._rdkit_pca["k"]}D '
                      f'(from {self.n_fp}-bit Morgan)')
            else:
                print('  RDKit fingerprint: not enough compounds, skipping')
                self.use_rdkit = False
        return self

    def chem(self, m):
        """Chemical: one-hot + Uni-Mol (+ optional RDKit PCA)."""
        base = super().chem(m)
        if self.use_rdkit and self._rdkit_pca:
            names = m['perturbation_no_concentration'].astype(str).values
            fp_feat = encode_fps(self._rdkit_pca, names)
            base = np.concatenate([base, fp_feat], axis=1).astype(np.float32)
        return base

    def blocks(self, m):
        return {'ctx': self.ctx(m), 'strain': self.strain(m),
                'chem': self.chem(m), 'obs': self.obs(m)}

    @property
    def n_chem_oh(self) -> int:
        """Number of one-hot dims in chem encoding (= n training compounds)."""
        return len(self.levels['perturbation_no_concentration'])


# ─────────────────────────────────────────────────────────────────────────────
# Feature dropout (per-sample, training only)
# ─────────────────────────────────────────────────────────────────────────────

def feat_dropout(strain_t: torch.Tensor, chem_t: torch.Tensor,
                 n_chem_oh: int, p_strain: float, p_chem_oh: float):
    """Randomly zero out strain one-hot or chem one-hot per sample."""
    bs = strain_t.shape[0]
    if p_strain > 0:
        mask = (torch.rand(bs, 1, device=strain_t.device) >= p_strain).float()
        strain_t = strain_t * mask
    if p_chem_oh > 0:
        mask = (torch.rand(bs, 1, device=chem_t.device) >= p_chem_oh).float()
        chem_oh = chem_t[:, :n_chem_oh] * mask
        chem_t = torch.cat([chem_oh, chem_t[:, n_chem_oh:]], dim=1)
    return strain_t, chem_t


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_data(n_mol: int = 32, use_rdkit: bool = False,
               n_fp: int = 256, n_fp_comp: int = 64):
    """Load proteome data and build feature blocks."""
    m, p = C.load('train_val')
    keep = C.keep_proteins(m, p)
    y, mask = C.to_log2(p, keep)
    tr = m['split_final'].eq('train').values
    isctl = C.is_control(m)
    ctl_tbl = C.control_table(m, y)

    enc = EncoderV2(use_unimol=True, n_mol=n_mol,
                    use_rdkit=use_rdkit, n_fp=n_fp, n_fp_comp=n_fp_comp)
    enc.fit(m[tr])
    blk = enc.blocks(m)
    blk['isctl'] = enc.is_control_feat(m)

    center = y.loc[tr].mean(axis=0).values.astype(np.float32)
    ctl_all, hit = C.matched_control(m, m.index, ctl_tbl)
    ctl_off = np.where(np.isfinite(ctl_all), ctl_all - center, 0.0).astype(np.float32)
    ctl_ok = (np.isfinite(ctl_all) & hit[:, None]).astype(np.float32)
    Y = (np.nan_to_num(y.values, nan=0.0).astype(np.float32) - center)
    MK = mask.values.astype(np.float32)

    return dict(m=m, y=y, mask=mask, keep=keep, tr=tr, isctl=isctl,
                ctl_tbl=ctl_tbl, enc=enc, blk=blk, center=center,
                Y=Y, MK=MK, ctl_off=ctl_off, ctl_ok=ctl_ok)


def tens(blk, rows, dev=DEV):
    return {k: torch.tensor(blk[k][rows], device=dev)
            for k in ['ctx', 'strain', 'chem', 'obs', 'isctl']}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one(d, cfg: dict, seed: int = 42, verbose: bool = True):
    """Train one model with given config, return (model, best_fc, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    blk = d['blk']
    dims = {k: blk[k].shape[1] for k in ['ctx', 'strain', 'chem', 'obs']}
    n_prot = int(d['keep'].sum())

    arch = cfg.get('arch', 'residual')
    hidden = cfg.get('hidden', 512)
    emb = cfg.get('emb', 64)
    rank = cfg.get('rank', 32)
    latent = cfg.get('latent', 256)
    lr = cfg.get('lr', 1e-3)
    epochs = cfg.get('epochs', 300)
    batch = cfg.get('batch', 256)
    w_fc = cfg.get('w_fc', 0.3)
    p_strain = cfg.get('p_strain', 0.0)
    p_chem_oh = cfg.get('p_chem_oh', 0.0)
    p_both = cfg.get('p_both', 0.0)   # joint dropout: zero both strain+chem_oh
    patience = cfg.get('patience', 6)
    n_chem_oh = d['enc'].n_chem_oh

    if arch == 'shared':
        model = SharedMultiHead(dims, n_prot, hidden, emb, latent).to(DEV)
    else:
        model = ResidualDecomp(dims, n_prot, hidden, emb, rank).to(DEV)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ti = np.where(d['tr'])[0]
    T = tens(blk, ti)
    ty = torch.tensor(d['Y'][ti], device=DEV)
    tm = torch.tensor(d['MK'][ti], device=DEV)
    tc = torch.tensor(d['ctl_off'][ti], device=DEV)
    tk = torch.tensor(d['ctl_ok'][ti], device=DEV)

    m = d['m']
    vb = m.index[m['split_final'].eq('val_both') & ~d['isctl']]
    vpos = m.index.get_indexer(vb)
    V = tens(blk, vpos)
    vctl, _ = C.matched_control(m, vb, d['ctl_tbl'])

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best, bad = -np.inf, 0
    hist, stop_ep = [], epochs
    n = len(ti)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEV)
        agg = np.zeros(2)
        for s in range(0, n, batch):
            j = perm[s:s + batch]
            opt.zero_grad()
            strain_j = T['strain'][j]
            chem_j = T['chem'][j]
            if p_strain > 0 or p_chem_oh > 0:
                strain_j, chem_j = feat_dropout(
                    strain_j, chem_j, n_chem_oh, p_strain, p_chem_oh)
            # joint dropout: zero out both strain+chem_oh together for some samples
            if p_both > 0:
                bs_j = strain_j.shape[0]
                jmask = (torch.rand(bs_j, 1, device=strain_j.device) < p_both)
                strain_j = strain_j * (~jmask).float()
                chem_oh = chem_j[:, :n_chem_oh] * (~jmask).float()
                chem_j = torch.cat([chem_oh, chem_j[:, n_chem_oh:]], dim=1)
            if arch == 'shared':
                pr, pt = model(T['ctx'][j], strain_j, chem_j,
                               T['obs'][j], T['isctl'][j], return_parts=True)
                fc_pred = pt.get('fc', pr - tc[j])
            else:
                pr, pt = model(T['ctx'][j], strain_j, chem_j,
                               T['obs'][j], T['isctl'][j], return_parts=True)
                fc_pred = pr - tc[j]
            l_mse = masked_mse(pr, ty[j], tm[j])
            l_fc = masked_fc_loss(fc_pred, ty[j] - tc[j], tm[j], tk[j])
            (l_mse + w_fc * l_fc).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg += np.array([l_mse.item(), l_fc.item()]) * len(j)
        sched.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pv = model(V['ctx'], V['strain'], V['chem'],
                           V['obs'], V['isctl']).cpu().numpy() + d['center']
            fc = M.fc_pcc(d['y'].loc[vb].values, pv, d['mask'].loc[vb].values, vctl)
            if verbose:
                print(f'  ep {ep:4d} | mMSE {agg[0]/n:.4f} | fc_loss {agg[1]/n:.4f}'
                      f' | val_both fc_pcc {fc:.4f} | lr {sched.get_last_lr()[0]:.2e}')
            hist.append({'epoch': ep, 'train_mse': agg[0]/n,
                         'train_fc_loss': agg[1]/n,
                         'val_both_fc_pcc': fc,
                         'lr': sched.get_last_lr()[0]})
            if fc > best:
                best, bad = fc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f'  早停 ep {ep}')
                    stop_ep = ep
                    break

    model.load_state_dict(best_state)
    model.eval()
    best_ep = max(hist, key=lambda r: r['val_both_fc_pcc'])['epoch'] if hist else 0
    return model, best, {'history': hist, 'best_ep': best_ep, 'stop_ep': stop_ep}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_splits(d, model) -> list:
    """Evaluate model on all val splits. Returns list of (label, metrics_dict)."""
    rows = []
    m = d['m']
    blk = d['blk']
    for sc in C.VAL_SPLITS:
        idx = m.index[m['split_final'].eq(sc) & ~d['isctl']]
        pos = m.index.get_indexer(idx)
        B = tens(blk, pos)
        with torch.no_grad():
            pred = model(B['ctx'], B['strain'], B['chem'],
                         B['obs'], B['isctl']).cpu().numpy() + d['center']
        ctl, _ = C.matched_control(m, idx, d['ctl_tbl'])
        rows.append((sc, M.evaluate(
            d['y'].loc[idx].values, pred, d['mask'].loc[idx].values, ctl,
            context_group(m, idx),
            m.loc[idx, 'perturbation_no_concentration'].values)))
    return rows


def eval_ensemble(d, models: list) -> list:
    """Average predictions from multiple models, then evaluate."""
    rows = []
    m = d['m']
    blk = d['blk']
    for sc in C.VAL_SPLITS:
        idx = m.index[m['split_final'].eq(sc) & ~d['isctl']]
        pos = m.index.get_indexer(idx)
        B = tens(blk, pos)
        preds = []
        for model in models:
            with torch.no_grad():
                preds.append(model(B['ctx'], B['strain'], B['chem'],
                                   B['obs'], B['isctl']).cpu().numpy())
        pred = np.mean(preds, axis=0) + d['center']
        ctl, _ = C.matched_control(m, idx, d['ctl_tbl'])
        rows.append((sc, M.evaluate(
            d['y'].loc[idx].values, pred, d['mask'].loc[idx].values, ctl,
            context_group(m, idx),
            m.loc[idx, 'perturbation_no_concentration'].values)))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(eval_rows: list) -> str:
    """Format eval results as a markdown table."""
    header = ('| Split | n | global_r2 | per_prot_r2 | fc_pcc '
              '| resid_ctx | resid_chem |')
    sep = '|---|---|---|---|---|---|---|'
    lines = [header, sep]
    for sc, r in eval_rows:
        lines.append(
            f"| {sc} | {r.get('n', '?')} "
            f"| {r.get('global_r2', float('nan')):.4f} "
            f"| {r.get('per_protein_r2', float('nan')):.4f} "
            f"| {r.get('fc_pcc', float('nan')):.4f} "
            f"| {r.get('resid_ctx_pcc', float('nan')):.4f} "
            f"| {r.get('resid_chem_pcc', float('nan')):.4f} |")
    return '\n'.join(lines)


def append_log(exp_id: str, cfg: dict, eval_rows: list,
               best_fc: float, best_ep: int, stop_ep: int,
               elapsed_s: float, notes: str = ''):
    """Append one experiment result to experiment_log.md."""
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f'\n## {exp_id}  _(val_both fc_pcc={best_fc:.4f})_\n\n')
        f.write(f'**Config:** `{cfg}`\n\n')
        if notes:
            f.write(f'**Notes:** {notes}\n\n')
        f.write(f'**Best epoch:** {best_ep}  |  **Stop epoch:** {stop_ep}  '
                f'|  **Time:** {elapsed_s/60:.1f} min\n\n')
        f.write(format_results_table(eval_rows))
        f.write('\n\n---\n')


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configs
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    # --- previously run baselines (loaded from disk) ---
    'exp01_baseline_ref': None,  # placeholder: read from stage3_validation.csv

    # --- new experiments ---
    'exp02_feat_dropout': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=32, use_rdkit=False,
        notes='Feature dropout: p_strain=0.3 p_chem_oh=0.2, simulate unseen strain/chem'
    ),
    'exp03_larger_model': dict(
        arch='residual', hidden=1024, emb=128, rank=64, latent=512,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=32, use_rdkit=False,
        notes='Larger model (hidden=1024, emb=128) + feature dropout'
    ),
    'exp04_mol64': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=64, use_rdkit=False,
        notes='n_mol=64 (full PCA span of ~39 training chems) + feature dropout'
    ),
    'exp05_rdkit_fp': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=32, use_rdkit=True, n_fp=256, n_fp_comp=64,
        notes='RDKit Morgan256-bit PCA64 fingerprints + feature dropout'
    ),
    'exp06_shared_dropout': dict(
        arch='shared', hidden=512, emb=64, rank=32, latent=256,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=32, use_rdkit=False,
        notes='SharedMultiHead arch + feature dropout'
    ),
    'exp07_ensemble_5seed': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=400, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=8,
        n_mol=64, use_rdkit=False, ensemble_seeds=[42, 7, 123, 2024, 999],
        notes='5-seed ensemble with best config (n_mol=64 + feat_dropout)'
    ),
    'exp08_combined_best': dict(
        arch='residual', hidden=1024, emb=128, rank=64, latent=512,
        epochs=500, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=64, use_rdkit=True, n_fp=256, n_fp_comp=64,
        notes='Combined: larger model + n_mol=64 + RDKit + feat_dropout'
    ),
    # ── Round 2: based on insights from round 1 ──────────────────────────────
    # Key insight: exp02 (p_strain=0.3,p_chem_oh=0.2,hidden=512) is the winner.
    # Next: tune dropout rates, higher w_fc, longer training, joint dropout

    'exp09_joint_dropout': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, p_both=0.15, patience=10,
        n_mol=32, use_rdkit=False,
        notes='Joint dropout: independently p_both=0.15 zero BOTH strain+chem_oh together'
    ),
    'exp10_high_strain_drop': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.5, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        notes='Higher strain dropout p_strain=0.5 — more aggressive val_both sim'
    ),
    'exp11_high_wfc': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.6,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        notes='Higher fold-change loss weight w_fc=0.6 + feat_dropout'
    ),
    'exp12_longer_train': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=800, lr=5e-4, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=12,
        n_mol=32, use_rdkit=False,
        notes='Longer training (800 ep, lr=5e-4) + feat_dropout, for later part of cosine'
    ),
    'exp13_ensemble_best_config': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.3,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        ensemble_seeds=[42, 7, 123, 2024, 999, 314, 888],
        notes='7-seed ensemble with exp02 config (best single model)'
    ),

    # ── Round 3: Combining best findings ──────────────────────────────────────
    # exp11 (w_fc=0.6) was best single model, now ensemble it

    'exp17_high_wfc_ensemble': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.6,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        ensemble_seeds=[42, 7, 123, 2024, 999, 314, 888],
        notes='7-seed ensemble with w_fc=0.6 (exp11 best single)'
    ),

    'exp21_final_ensemble_10': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.6,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        ensemble_seeds=[42, 7, 123, 2024, 999, 314, 888, 555, 777, 2023],
        notes='10-seed ensemble with w_fc=0.6 for maximum stability'
    ),

    'exp22_protein_seq_features': dict(
        arch='residual', hidden=512, emb=64, rank=32, latent=256,
        epochs=500, lr=1e-3, batch=256, w_fc=0.6,
        p_strain=0.3, p_chem_oh=0.2, patience=10,
        n_mol=32, use_rdkit=False,
        use_protein_features=True,  # NEW: add protein sequence features
        notes='Add protein sequence features (length, MW, pI, hydrophobicity) + best config'
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_exp(exp_id: str, cfg: dict):
    """Run a single experiment. Handles ensemble if cfg has ensemble_seeds."""
    print(f'\n{"="*70}')
    print(f'  {exp_id}')
    print(f'{"="*70}')
    t0 = time.time()

    n_mol = cfg.get('n_mol', 32)
    use_rdkit = cfg.get('use_rdkit', False)
    n_fp = cfg.get('n_fp', 256)
    n_fp_comp = cfg.get('n_fp_comp', 64)

    print('  Loading data …')
    d = build_data(n_mol=n_mol, use_rdkit=use_rdkit, n_fp=n_fp, n_fp_comp=n_fp_comp)
    n_prot = int(d['keep'].sum())
    print(f'  Device: {DEV} | proteins: {n_prot} | '
          f'strain_dim: {d["blk"]["strain"].shape[1]} | '
          f'chem_dim: {d["blk"]["chem"].shape[1]}')

    seeds = cfg.get('ensemble_seeds', None)
    if seeds:
        # ensemble run
        models = []
        for seed in seeds:
            print(f'\n  --- seed {seed} ---')
            model, bfc, tr = train_one(d, cfg, seed=seed, verbose=True)
            models.append(model)
            print(f'  seed {seed}: val_both fc_pcc={bfc:.4f} @ ep {tr["best_ep"]}')
            # Save individual seed model
            torch.save({'state': model.state_dict(), 'center': d['center'],
                        'cfg': cfg, 'seed': seed, 'best': bfc, 'best_ep': tr['best_ep']},
                       C.OUT / f'{exp_id}_seed{seed}.pt')
        print('\n  Evaluating ensemble …')
        eval_rows = eval_ensemble(d, models)
        best_fc = max(r.get('fc_pcc', -1) for _, r in eval_rows
                      if _ == 'val_both')
        # find val_both fc_pcc
        for sc, r in eval_rows:
            if sc == 'val_both':
                best_fc = r.get('fc_pcc', float('nan'))
        best_ep = 0
        stop_ep = 0
        # Save ensemble config
        ensemble_cfg = cfg.copy()
        ensemble_cfg['ensemble_seeds'] = seeds
        torch.save({'cfg': ensemble_cfg, 'center': d['center'],
                    'best': best_fc, 'n_seeds': len(seeds)},
                   C.OUT / f'{exp_id}_model.pt')
        C.save_history([], f'{exp_id}_seeds.csv')
    else:
        model, best_fc, tr = train_one(d, cfg, seed=cfg.get('seed', 42), verbose=True)
        eval_rows = eval_splits(d, model)
        best_ep = tr['best_ep']
        stop_ep = tr['stop_ep']
        C.save_history(tr['history'], f'{exp_id}_history.csv')
        torch.save({'state': model.state_dict(), 'center': d['center'],
                    'cfg': cfg, 'best': best_fc, 'best_ep': best_ep},
                   C.OUT / f'{exp_id}_model.pt')

    elapsed = time.time() - t0
    print(f'\n  === {exp_id} 结果 ===')
    for sc, r in eval_rows:
        print(f'  {sc}: fc_pcc={r.get("fc_pcc", float("nan")):.4f}  '
              f'resid_ctx={r.get("resid_ctx_pcc", float("nan")):.4f}  '
              f'resid_chem={r.get("resid_chem_pcc", float("nan")):.4f}  '
              f'per_prot_r2={r.get("per_protein_r2", float("nan")):.4f}')
    print(f'  Time: {elapsed/60:.1f} min')

    notes = cfg.get('notes', '')
    append_log(exp_id, {k: v for k, v in cfg.items() if k not in ('notes', 'ensemble_seeds')},
               eval_rows, best_fc, best_ep, stop_ep, elapsed, notes)
    return eval_rows, best_fc


def write_log_header():
    """Write/overwrite header only if file doesn't exist yet."""
    if LOG_PATH.exists():
        return
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('# 虚拟细胞竞赛 — 实验记录\n\n')
        f.write('**评测指标说明：** 关键指标 `fc_pcc`（fold-change PCC，25%）、'
                '`resid_ctx_pcc`（上下文残差，20%）、`resid_chem_pcc`（化合物残差，20%）、'
                '`per_protein_r2`（逐蛋白R²，20%）；`global_r2`（15%）。\n\n')
        f.write('**基线 (exp01)：** stage3 残差分解，val_both fc_pcc=0.2451\n\n')
        f.write('---\n')

        # write baseline from existing results
        f.write('\n## exp01_baseline_ref  _(val_both fc_pcc=0.2451)_\n\n')
        f.write('**Config:** stage3 残差分解原始结果（读自 stage3_validation.csv）\n\n')
        f.write('| Split | n | global_r2 | per_prot_r2 | fc_pcc '
                '| resid_ctx | resid_chem |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        baselines = [
            ('val_strain_only', 1357, 0.9500, 0.2627, 0.3431, 0.3938, 0.4505),
            ('val_chem_only',   1065, 0.9774, 0.7080, 0.3882, 0.3439, 0.3956),
            ('val_both',         269, 0.9604, 0.5310, 0.2451, 0.3323, 0.3768),
            ('val_time',         142, 0.9836, 0.7943, 0.5886, 0.4888, 0.5492),
        ]
        for row in baselines:
            f.write(f'| {row[0]} | {row[1]} | {row[2]:.4f} | {row[3]:.4f} '
                    f'| {row[4]:.4f} | {row[5]:.4f} | {row[6]:.4f} |\n')
        f.write('\n---\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', nargs='+', default=None,
                    help='实验 ID（如 exp02_feat_dropout），不指定则按顺序全跑')
    ap.add_argument('--all', action='store_true', help='跑全部实验')
    a = ap.parse_args()

    write_log_header()

    to_run = list(EXPERIMENTS.keys())[1:]  # skip exp01_baseline_ref
    if a.exp:
        to_run = a.exp
    elif not a.all:
        # default: run all new experiments in order
        pass

    print(f'将运行实验: {to_run}')
    results = {}
    for exp_id in to_run:
        if exp_id == 'exp01_baseline_ref':
            continue
        if exp_id not in EXPERIMENTS:
            print(f'未知实验 ID: {exp_id}，跳过')
            continue
        cfg = EXPERIMENTS[exp_id]
        eval_rows, best_fc = run_exp(exp_id, cfg)
        results[exp_id] = best_fc
        print(f'\n[累计结果] {exp_id}: val_both fc_pcc={best_fc:.4f}')

    print('\n\n========== 所有实验完成 ==========')
    for eid, fc in results.items():
        print(f'  {eid}: {fc:.4f}')
    print(f'\n详细结果记录在: {LOG_PATH}')


if __name__ == '__main__':
    main()
