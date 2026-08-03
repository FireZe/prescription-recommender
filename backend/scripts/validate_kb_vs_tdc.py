"""
Validação da base de conhecimento (deteção de interações) contra o TDC / DrugBank-DDI
=====================================================================================

Verifica, para TODAS as classes terapêuticas (cobertura total), se cada regra de
interação da knowledge_base.json é RECONHECIDA por uma fonte externa independente:
o dataset DrugBank-DDI do Therapeutics Data Commons (TDC).

  Referência citável: Huang et al., "Artificial intelligence foundation for
  therapeutic science", Nature Chemical Biology, 2022; e Huang et al., NeurIPS 2021.

NOTA: o TDC/DrugBank-DDI confirma a EXISTÊNCIA/tipo da interação (deteção), não a
severidade clínica (Major/Moderate/Minor). A severidade é validada à parte, contra
fontes clínicas (RCM/Infarmed, DGS, Stockley's/BNF) — ver tabela de severidade.

PRÉ-REQUISITO:
    pip install PyTDC

COMO USAR:
    python backend/scripts/validate_kb_vs_tdc.py

SAÍDA:
    - resumo no terminal (taxa de deteção)
    - backend/data/kb_vs_tdc_concordancia.csv  (tabela detalhada para a tese)
"""

import csv
import json
import os
from itertools import product
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
OUT_CSV = BASE_DIR / "data" / "kb_vs_tdc_concordancia.csv"
TDC_DATA_DIR = BASE_DIR / "data" / "tdc"

# ── Mapeamento dos identificadores internos da KB -> DrugBank Accession ──────────
# (VERIFICAR os assinalados com '?'; os restantes são amplamente conhecidos)
DRUGBANK_ID = {
    "ibuprofen": "DB01050", "naproxen": "DB00788", "paracetamol": "DB00316",
    "acetylsalicylic_acid": "DB00945", "clopidogrel": "DB00758",
    "warfarin": "DB00682", "acenocoumarol": "DB01418", "apixaban": "DB06605",
    "enoxaparin": "DB01225", "enalapril": "DB00584", "ramipril": "DB00178",
    "losartan": "DB00678", "valsartan": "DB00177", "furosemide": "DB00695",
    "hydrochlorothiazide": "DB00999", "simvastatin": "DB00641",
    "atorvastatin": "DB01076", "azithromycin": "DB00207", "clarithromycin": "DB01211",
    "sertraline": "DB01104", "escitalopram": "DB01175", "amitriptyline": "DB00321",
    "digoxin": "DB00390", "amiodarone": "DB01118", "metoprolol": "DB00264",
    "bisoprolol": "DB00612", "carvedilol": "DB01136", "atenolol": "DB00335",
    "nebivolol": "DB04861", "metformin": "DB00331", "gliclazide": "DB01120",
    "amoxicillin": "DB01060", "amoxicillin_clavulanate": "DB01060",  # amoxicilina
    "omeprazole": "DB00338", "tramadol": "DB00193", "salbutamol": "DB01001",
    "budesonide_formoterol": "DB00983",  # formoterol (componente com mais DDIs)
}


def load_kb():
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    meds = kb.get("medications", {})
    class_to_drugs = {}
    for drug_id, info in meds.items():
        class_to_drugs.setdefault(info.get("therapeutic_class", "?"), []).append(drug_id)
    rules = kb.get("interaction_rules", []) or kb.get("interactions", [])
    return class_to_drugs, rules


def load_tdc_pairs():
    """Conjunto de frozenset({DrugBankID_a, DrugBankID_b}) reconhecidos pelo TDC."""
    # O tdc.multi_pred importa, em bloco, carregadores que dependem de pacotes
    # pesados e irrelevantes para o DDI (single-cell, rdkit, etc.). Simulam-se
    # esses módulos para o import do DDI passar sem os instalar.
    import sys, types

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            full = f"{self.__name__}.{name}"
            m = _StubModule(full)
            sys.modules[full] = m
            return m

    # Apenas módulos que bloqueiam o import e que NÃO são introspecionados por
    # outras libs (não simular torch: o scipy inspeciona-o e quebra).
    for _m in ["tiledbsoma", "cellxgene_census", "gget"]:
        sys.modules.setdefault(_m, _StubModule(_m))

    try:
        from tdc.multi_pred import DDI
    except Exception:
        import traceback
        print("[ERRO] Falha ao importar o TDC. Causa real abaixo:")
        traceback.print_exc()
        print("\n-> Instala o módulo em falta indicado acima (ex.: pip install <modulo>) e repete.")
        return None
    TDC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = DDI(name="DrugBank", path=str(TDC_DATA_DIR))
    df = data.get_data()
    # colunas típicas: Drug1_ID, Drug1, Drug2_ID, Drug2, Y
    id_a = "Drug1_ID" if "Drug1_ID" in df.columns else df.columns[0]
    id_b = "Drug2_ID" if "Drug2_ID" in df.columns else df.columns[2]
    pairs = set()
    for a, b in zip(df[id_a].astype(str), df[id_b].astype(str)):
        pairs.add(frozenset({a, b}))
    print(f"TDC/DrugBank-DDI carregado: {len(df)} associações | {len(pairs)} pares únicos")
    return pairs


def rule_pairs(rule, class_to_drugs):
    def members(side):
        med = rule.get(f"medication_{side}")
        cls = rule.get(f"class_{side}")
        if med:
            return [med]
        if cls:
            return class_to_drugs.get(cls, [])
        return []
    out = []
    for a, b in product(members("a"), members("b")):
        if a != b:
            out.append((a, b))
    return out


def main():
    class_to_drugs, rules = load_kb()
    tdc_pairs = load_tdc_pairs()
    if tdc_pairs is None:
        return

    rows, n_aplicaveis, n_detetadas, unmapped = [], 0, 0, set()

    for rule in rules:
        rid = rule.get("rule_id", "—")
        kb_sev = rule.get("severity", "—")
        label = f"{rule.get('medication_a') or rule.get('class_a')} × {rule.get('medication_b') or rule.get('class_b')}"
        pairs = rule_pairs(rule, class_to_drugs)
        if not pairs:  # regra por atributo (ex.: qt_qt) — sem par específico
            rows.append([rid, label, kb_sev, "—", "regra_por_atributo"])
            continue

        n_aplicaveis += 1
        detected, example = False, pairs[0]
        for a, b in pairs:
            ia, ib = DRUGBANK_ID.get(a), DRUGBANK_ID.get(b)
            if not ia:
                unmapped.add(a)
            if not ib:
                unmapped.add(b)
            if ia and ib and frozenset({ia, ib}) in tdc_pairs:
                detected, example = True, (a, b)
                break
        if detected:
            n_detetadas += 1
        rows.append([rid, label, kb_sev,
                     f"{example[0]}×{example[1]}",
                     "detetada" if detected else "nao_detetada"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rule_id", "regra", "severidade_KB", "par_exemplo", "estado_TDC"])
        w.writerows(rows)

    print("\n================ RESUMO (deteção) ================")
    print(f"Regras aplicáveis (com par): {n_aplicaveis}")
    print(f"  reconhecidas pelo TDC:     {n_detetadas}")
    if n_aplicaveis:
        print(f"  taxa de deteção:           {n_detetadas/n_aplicaveis*100:.0f}%")
    if unmapped:
        print(f"\n[AVISO] fármacos sem DrugBank ID mapeado: {sorted(unmapped)}")
        print("        adiciona-os ao dicionário DRUGBANK_ID e volta a correr.")
    print(f"\nTabela detalhada: {OUT_CSV}")


if __name__ == "__main__":
    main()
