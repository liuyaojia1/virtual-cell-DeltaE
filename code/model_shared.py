"""共享 encoder + 多头输出（阶段 3 残差分解的替代架构）。

残差分解用输入隔离换可辨识性，代价是每个分支只看到部分条件；消融显示
chem/smod 分支在 val_both 上反而是负担。这里换相反的假设：单个 encoder
吃全部生物条件得到 z，再由多个**监督目标不同**的头去读它——

  head_abs : z → 绝对丰度偏移，受 masked MSE 监督（评测走这条路）
  head_fc  : z → 相对 control 的 Δ，受 fold-change loss 监督（辅助任务）
  calib    : obs → additive offset，零初始化

多头的意义在于目标不同：两个线性头读同一个 z 再相加等价于单头，只有监督
信号不同才真正塑形 z。obs 不进 trunk，仪器/板变异只能被 calib 吸收——这条
隔离是原架构里唯一被消融证明有效的部分，保留。
"""
import torch
import torch.nn as nn

from model_residual import mlp


class SharedMultiHead(nn.Module):
    def __init__(self, dims, n_prot, hidden=512, emb=64, latent=256,
                 p_drop=0.1, use=('abs', 'fc', 'calib')):
        super().__init__()
        self.use = set(use)
        d_ctx, d_str, d_chm, d_obs = (dims['ctx'], dims['strain'],
                                      dims['chem'], dims['obs'])

        # 实体压缩：chem 块含 one-hot + Uni-Mol，直接进 trunk 会被维度淹没
        self.chem_emb = mlp(d_chm, hidden, emb, p_drop, depth=1)
        self.strain_emb = mlp(d_str, hidden // 2, emb, p_drop, depth=1)

        self.trunk = mlp(d_ctx + emb * 2 + 1, hidden, latent, p_drop)
        self.act = nn.Sequential(nn.ReLU(), nn.Dropout(p_drop))

        self.head_abs = nn.Linear(latent, n_prot)
        self.head_fc = nn.Linear(latent, n_prot)
        nn.init.zeros_(self.head_fc.weight)
        nn.init.zeros_(self.head_fc.bias)

        self.calib = nn.Linear(d_obs, n_prot, bias=False)
        nn.init.zeros_(self.calib.weight)

    def forward(self, ctx, strain, chem, obs, isctl, return_parts=False):
        z = self.act(self.trunk(torch.cat(
            [ctx, self.strain_emb(strain), self.chem_emb(chem), isctl], 1)))
        parts = {'abs': self.head_abs(z), 'calib': self.calib(obs),
                 'fc': self.head_fc(z)}
        # fc 头目标是 Δ，不进绝对丰度预测，只经辅助 loss 反传塑形 z
        total = sum(v for k, v in parts.items()
                    if k in self.use and k != 'fc')
        return (total, parts) if return_parts else total
