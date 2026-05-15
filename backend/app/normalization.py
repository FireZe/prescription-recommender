import re
from typing import Optional


MEDICATION_SYNONYMS = {
    "ibuprofen": [
        "ibuprofen", "ibuprofeno", "ibuprofen 400 mg", "ibuprofen 400 mg oral tablet",
        "ibuprofen 400 mg oral tablet [ibu]"
    ],
    "naproxen": [
        "naproxen", "naproxeno", "naproxeno generis", "naproxen 250 mg"
    ],
    "paracetamol": [
        "paracetamol", "acetaminophen", "paracetamol generis", "paracetamol 1000 mg"
    ],
    "acetylsalicylic_acid": [
        "ácido acetilsalicílico", "acido acetilsalicilico", "aspirin", "aspirina",
        "aas", "acetylsalicylic acid", "ácido acetilsalicílico generis"
    ],
    "clopidogrel": [
        "clopidogrel", "clopidogrel 75 mg", "clopidogrel 75 mg oral tablet"
    ],
    "warfarin": [
        "warfarin", "varfarina", "varfarina sódica", "varfine"
    ],
    "enalapril": [
        "enalapril", "enalapril vitória", "maleato de enalapril"
    ],
    "ramipril": [
        "ramipril", "ramipril generis"
    ],
    "losartan": [
        "losartan", "losartan potássico", "losartan de potássio", "losartan generis", "losartan jaba"
    ],
    "valsartan": [
        "valsartan", "valsartan generis"
    ],
    "furosemide": [
        "furosemide", "furosemida", "furosemida generis"
    ],
    "hydrochlorothiazide": [
        "hydrochlorothiazide", "hidroclorotiazida", "hidroclorotiazida generis", "hctz"
    ],
    "simvastatin": [
        "simvastatin", "sinvastatina", "sinvastatina generis", "sinvastatina bluepharma"
    ],
    "atorvastatin": [
        "atorvastatin", "atorvastatina", "atorvastatina alter"
    ],
    "azithromycin": [
        "azithromycin", "azitromicina", "azitromicina generis"
    ],
    "clarithromycin": [
        "clarithromycin", "claritromicina", "claritromicina generis"
    ],
    "sertraline": [
        "sertraline", "sertralina", "sertralina generis"
    ],
    "amitriptyline": [
        "amitriptyline", "amitriptilina", "cloridrato de amitriptilina", "adt"
    ],
    "digoxin": [
        "digoxin", "digoxina", "lanoxin"
    ],
    "amiodarone": [
        "amiodarone", "amiodarona", "amiodarona generis"
    ],
    "metoprolol": [
        "metoprolol", "metoprolol succinate", "metoprolol succinate extended release"
    ],
}


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
    value = value.replace("é", "e").replace("ê", "e")
    value = value.replace("í", "i")
    value = value.replace("ó", "o").replace("õ", "o").replace("ô", "o")
    value = value.replace("ú", "u")
    value = value.replace("ç", "c")
    value = re.sub(r"[^a-z0-9\s\[\]/.-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_medication_id(raw_name: str) -> Optional[str]:
    if not raw_name:
        return None

    text = normalize_text(raw_name)

    for med_id, synonyms in MEDICATION_SYNONYMS.items():
        for synonym in synonyms:
            syn = normalize_text(synonym)
            if syn == text or syn in text:
                return med_id

    return None

def normalize_condition_id(description: str) -> str:
    if not description:
        return ""

    text = normalize_text(description)

    if "renal" in text or "kidney" in text:
        return "renal_disease"

    if "hypertension" in text or "high blood pressure" in text:
        return "hypertension"

    if "diabetes" in text:
        return "diabetes"

    if "heart failure" in text or "insuficiencia cardiaca" in text:
        return "heart_failure"

    if "myocardial infarction" in text or "enfarte" in text:
        return "myocardial_infarction"

    if "stroke" in text or "avc" in text:
        return "stroke"

    if "ulcer" in text or "ulcera" in text:
        return "active_gi_ulcer"

    if "bleeding" in text or "hemorragia" in text:
        return "active_bleeding"

    if "pain" in text or "dor" in text:
        return "pain"

    if "arthritis" in text or "inflammation" in text or "inflamacao" in text:
        return "inflammation"

    if "fever" in text or "febre" in text:
        return "fever"

    return text.strip()


def infer_main_problem(conditions: list[str]) -> str:
    joined = " ".join(conditions).lower()

    if "pain" in joined or "dor" in joined:
        return "pain"

    if "inflammation" in joined or "arthritis" in joined or "inflamacao" in joined:
        return "inflammation"

    if "infection" in joined or "infecao" in joined:
        return "infection"

    if "fever" in joined or "febre" in joined:
        return "fever"

    if "hypertension" in joined or "hipertensao" in joined:
        return "hypertension"

    if "diabetes" in joined:
        return "diabetes"

    if "heart_failure" in joined or "insuficiencia cardiaca" in joined:
        return "heart_failure"

    return "unspecified"

def normalize_main_problem(raw_problem: str | None) -> str:
    if not raw_problem:
        return "unspecified"

    text = normalize_text(raw_problem)

    mapping = {
        "dor": "pain",
        "pain": "pain",
        "analgesia": "pain",
        "cefaleia": "pain",
        "lombalgia": "pain",

        "inflamacao": "inflammation",
        "inflamação": "inflammation",
        "inflammation": "inflammation",
        "artrite": "inflammation",
        "arthritis": "inflammation",

        "infecao": "infection",
        "infeção": "infection",
        "infection": "infection",
        "infeccao": "infection",
        "infeccão": "infection",

        "febre": "fever",
        "fever": "fever",

        "hipertensao": "hypertension",
        "hipertensão": "hypertension",
        "hypertension": "hypertension",

        "diabetes": "diabetes",

        "insuficiencia cardiaca": "heart_failure",
        "insuficiência cardíaca": "heart_failure",
        "heart failure": "heart_failure",

        "arritmia": "arrhythmia",
        "arrhythmia": "arrhythmia",
    }

    return mapping.get(text, text)