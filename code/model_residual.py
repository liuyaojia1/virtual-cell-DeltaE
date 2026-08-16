"""残差分解架构（PDF 5.4.1）。

ŷ = 共享应激响应(上下文+菌株基线)          → 对齐绝对丰度 (20%)
  + 化合物特异残差(chem 驱动)              → 对齐上下文均值残差 模块3 (20%)
  + 菌株调制残差(strain×chem 驱动)         → 对齐药物均值残差 模块4 (20%)
  + 菌株×化合物 低秩交互                   → 对齐双重未知 模块5
  + 观测校准 offset(仪器/板/来源)          → 与生物信号分离，不参与泛化

组件的可辨识性靠输入隔离保证：共享分支看不到化合物，
化合物分支看不到菌株，因此各项无法互相吸收对方的信号。
"""
import torch
import torch.nn as nn


def mlp(d_in, hidden, d_out, p_drop=0.1, depth=2):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class ResidualDecomp(nn.Module):
    def __init__(self, dims, n_prot, hidden=512, emb=64, rank=32,
                 p_drop=0.1, use=('shared', 'chem', 'smod', 'inter', 'calib')):
        super().__init__()
        self.use = set(use)
        d_ctx, d_str, d_chm, d_obs = (dims['ctx'], dims['strain'],
                                     dims['chem'], dims['obs'])

        # 共享应激响应：上下文 + 菌株 + 是否对照（不含化合物身份）
        self.shared = mlp(d_ctx + d_str + 1, hidden, n_prot, p_drop)

        # 实体 embedding，供残差与交互项共用
        self.chem_emb = mlp(d_chm, hidden, emb, p_drop, depth=1)
        self.strain_emb = mlp(d_str, hidden // 2, emb, p_drop, depth=1)

        # 化合物特异残差：化合物 + 上下文（跨菌株共享的扰动效应）
        self.chem_res = mlp(emb + d_ctx, hidden, n_prot, p_drop)
        self._zero_last(self.chem_res)

        # 菌株调制残差：菌株如何改变该化合物的效应
        self.smod = mlp(emb * 2 + d_ctx, hidden, n_prot, p_drop)
        self._zero_last(self.smod)

        # 低秩双线性交互：外积压到 rank 维再映射，参数量可控
        self.inter_s = nn.Linear(emb, rank, bias=False)
        self.inter_c = nn.Linear(emb, rank, bias=False)
        self.inter_out = nn.Linear(rank, n_prot, bias=False)
        nn.init.zeros_(self.inter_out.weight)

        # 观测校准：零初始化的 additive offset
        self.calib = nn.Linear(d_obs, n_prot, bias=False)
        nn.init.zeros_(self.calib.weight)

    @staticmethod
    def _zero_last(seq):
        """残差分支末层零初始化：训练初期等价于纯共享模型，收敛更稳。"""
        nn.init.zeros_(seq[-1].weight)
        nn.init.zeros_(seq[-1].bias)

    def parts(self, ctx, strain, chem, obs, isctl):
        ce = self.chem_emb(chem)
        se = self.strain_emb(strain)
        out = {}
        out['shared'] = self.shared(torch.cat([ctx, strain, isctl], 1))
        out['chem'] = self.chem_res(torch.cat([ce, ctx], 1))
        out['smod'] = self.smod(torch.cat([se, ce, ctx], 1))
        out['inter'] = self.inter_out(self.inter_s(se) * self.inter_c(ce))
        out['calib'] = self.calib(obs)
        return out

    def forward(self, ctx, strain, chem, obs, isctl, return_parts=False):
        parts = self.parts(ctx, strain, chem, obs, isctl)
        total = sum(v for k, v in parts.items() if k in self.use)
        return (total, parts) if return_parts else total


def masked_mse(pred, y, mk):
    return (((pred - y) ** 2) * mk).sum() / mk.sum().clamp(min=1)


def masked_fc_loss(pred, y, mk, ctl_ok, min_n=50):
    """逐样本 fold change Pearson → 1 - r。

    pred/y 已是相对 center 的偏移，ctl 亦然，故 Δ 直接相减即可。
    ctl_ok 标记该行是否有 matched control。
    """
    v = mk * ctl_ok
    n = v.sum(1)
    keep = n >= min_n
    if keep.sum() == 0:
        return pred.new_zeros(())
    a, b, w = pred[keep], y[keep], v[keep]
    n = w.sum(1, keepdim=True)
    am = (a * w).sum(1, keepdim=True) / n
    bm = (b * w).sum(1, keepdim=True) / n
    a, b = (a - am) * w, (b - bm) * w
    num = (a * b).sum(1)
    # clamp 必须在 sqrt 之内：sqrt(0) 的梯度为 inf，会立刻污染成 NaN
    den = torch.sqrt(((a * a).sum(1) * (b * b).sum(1)).clamp(min=1e-8))
    return 1.0 - (num / den).mean()
