"""为 57 个 perturbation 取 SMILES。

策略：PubChem PUG-REST 按名称查询 → 失败则用内置的人工校对表兜底。
结果缓存到 缓存/smiles.csv，只联网一次。

名称清洗要点：数据里的名称带盐型、水合物、立体标注（如
"Amiodarone hydrochloride"、"Nystatin dihydrate"、"(1R, 2S, 5R) - (-) - Menthol"），
PubChem 多数能直接命中；命中失败时逐级剥离盐型/水合物后重试。
"""
import re
import time
import pandas as pd
import requests

import common as C

PUG = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}'
       '/property/SMILES,ConnectivitySMILES,MolecularWeight/JSON')

# 非小分子扰动：不走分子编码，单独作为标记特征处理
NON_MOLECULAR = {'Quality Control'}

# 溶剂/对照：有明确结构，保留（模型需要它们作为基线）
# 人工校对的兜底表，覆盖 PubChem 易查不到或歧义的条目
FALLBACK = {
    'CHX': 'C[C@@H]1C[C@H](C)C(=O)[C@H](C1=O)[C@@H](O)C[C@H]2CC(=O)NC(=O)C2',  # Cycloheximide
    'MMS': 'COS(=O)(=O)C',                    # Methyl methanesulfonate
    'SDS': 'CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]',
    'EDTA': 'OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O',
    'H2O2': 'OO',
    'NaCl': '[Na+].[Cl-]',
    'Water': 'O',
    'DMSO': 'CS(=O)C',
    'Sorbitol': 'OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO',
    'FCCP': 'N#CC(C#N)=NNc1ccc(OC(F)(F)F)cc1',
    'G418': None,       # Geneticin，氨基糖苷，PubChem 名称可查
    'Hoechst 33258': None,
    'LY 294002 hydrochloride': None,
    'U-73122': None,
    'Neomycin B': None,
    '1-10 Phenanthroline monohydrate': 'c1cnc2c(c1)ccc1cccnc12.O',
    '(1R, 2S, 5R) - (-) - Menthol': 'C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1',
    '(S)-(+)-Camptothecin': None,
}

# 名称歧义时改用的查询别名
ALIAS = {
    'CHX': 'Cycloheximide',
    'MMS': 'Methyl methanesulfonate',
    'SDS': 'Sodium dodecyl sulfate',
    'G418': 'Geneticin',
    'FCCP': 'Carbonyl cyanide 4-(trifluoromethoxy)phenylhydrazone',
    'LY 294002 hydrochloride': 'LY294002',
    'U-73122': 'U-73122',
    '(S)-(+)-Camptothecin': 'Camptothecin',
    '(1R, 2S, 5R) - (-) - Menthol': 'L-Menthol',
    '1-10 Phenanthroline monohydrate': '1,10-Phenanthroline',
    'Anisomycin': 'Anisomycin',
    'Nystatin dihydrate': 'Nystatin',
    'Doxycycline hyclate': 'Doxycycline',
    'Harmine hydrochloride': 'Harmine',
    # 商品试剂是同系物混合物，PubChem 无混合物条目；取主组分 A 作代表结构
    'Oligomycin': 'Oligomycin A',
    'Tunicamycin': 'Tunicamycin A',
}


def strip_salt(name):
    """逐级剥离盐型/水合物后缀，产出候选查询名。"""
    out = [name]
    n = re.sub(r'\s*\b(hydrochloride|dihydrochloride|hyclate|citrate|isethionate'
               r'|monohydrate|dihydrate|hydrate|sodium|methyl)\b', '', name,
               flags=re.I).strip()
    if n and n != name:
        out.append(n)
    out.append(re.sub(r'[\(\)\[\],]', ' ', name).strip())
    return list(dict.fromkeys(out))


def query(name, tries=3):
    for cand in ([ALIAS[name]] if name in ALIAS else []) + strip_salt(name):
        url = PUG.format(requests.utils.quote(cand))
        for k in range(tries):
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    p = r.json()['PropertyTable']['Properties'][0]
                    smi = p.get('SMILES') or p.get('ConnectivitySMILES')
                    if smi:
                        return smi, p.get('CID'), cand
                    break
                if r.status_code == 404:
                    break
                time.sleep(1.5 * (k + 1))
            except Exception:
                time.sleep(1.5 * (k + 1))
        time.sleep(0.25)   # PubChem 限速：≤5 req/s
    return None, None, None


def main():
    fs = sorted(C.DATA.glob('*metadata*.csv'))
    al = pd.concat([pd.read_csv(f) for f in fs])
    names = sorted(al['perturbation_no_concentration'].astype(str).unique())

    rows = []
    for i, nm in enumerate(names, 1):
        if nm in NON_MOLECULAR:
            rows.append(dict(name=nm, smiles=None, cid=None,
                             source='non_molecular', query=None))
            print(f'  {i:2d}/{len(names)} {nm:45s} → 非分子扰动')
            continue
        fb = FALLBACK.get(nm)
        smi, cid, used = query(nm)
        src = 'pubchem'
        if not smi and fb:
            smi, src, used = fb, 'fallback', nm
        rows.append(dict(name=nm, smiles=smi, cid=cid, source=src, query=used))
        flag = 'OK ' if smi else '缺失'
        print(f'  {i:2d}/{len(names)} {nm:45s} → {flag} [{src}]'
              + (f' cid={cid}' if cid else ''))

    df = pd.DataFrame(rows)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    out = C.CACHE / 'smiles.csv'
    df.to_csv(out, index=False, encoding='utf-8-sig')
    ok = df.smiles.notna().sum()
    print(f'\n命中 {ok}/{len(df)}，缺失 {len(df) - ok}')
    miss = df[df.smiles.isna() & df.source.ne('non_molecular')]
    if len(miss):
        print('未命中（需人工补 FALLBACK）:')
        for n in miss.name:
            print('   -', n)
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()
