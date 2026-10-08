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
        "acido acetilsalicilico", "aspirin", "aspirina",
        "aas", "acetylsalicylic acid"
    ],
    "clopidogrel": [
        "clopidogrel", "clopidogrel 75 mg", "clopidogrel 75 mg oral tablet"
    ],
    "warfarin": [
        "warfarin", "varfarina", "varfarina sodica", "varfine"
    ],
    "acenocoumarol": [
        "acenocoumarol", "acenocumarol", "sintrom"
    ],
    "enalapril": [
        "enalapril", "enalapril vitoria", "maleato de enalapril"
    ],
    "ramipril": [
        "ramipril", "ramipril generis"
    ],
    "losartan": [
        "losartan", "losartan potassico", "losartan de potassio", "losartan generis", "losartan jaba"
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
        "metoprolol", "metoprolol succinate", "metoprolol succinate extended release",
        "tartarato de metoprolol", "metoprolol aurobindo"
    ],
    "bisoprolol": [
        "bisoprolol", "bisoprolol fumarato", "bisoprolol generis", "bisoprolol generis phar"
    ],
    "carvedilol": [
        "carvedilol", "carvedilol generis"
    ],
    "atenolol": [
        "atenolol", "atenolol generis"
    ],
    "metformin": [
        "metformin", "metformina", "metformina basi", "metformina generis"
    ],
    "gliclazide": [
        "gliclazide", "gliclazida", "gliclazida generis", "diamicron"
    ],
    "amoxicillin_clavulanate": [
        "amoxicillin_clavulanate", "amoxicilina + acido clavulanico",
        "amoxicilina mais acido clavulanico", "amoxicilina/acido clavulanico",
        "amoxicilina e acido clavulanico", "acido clavulanico", "clavulanico",
        "augmentin", "co-amoxiclav"
    ],
    "amoxicillin": [
        "amoxicillin", "amoxicilina", "amoxicilina generis", "amoxil"
    ],
    "omeprazole": [
        "omeprazole", "omeprazol", "omeprazol generis", "losec"
    ],
    "tramadol": [
        "tramadol", "tramadol generis", "cloridrato de tramadol"
    ],
    "apixaban": [
        "apixaban", "apixabano", "eliquis"
    ],
    "enoxaparin": [
        "enoxaparin", "enoxaparina", "enoxaparina sodica", "lovenox"
    ],
    "escitalopram": [
        "escitalopram", "escitalopram alter", "oxalato de escitalopram"
    ],
    "nebivolol": [
        "nebivolol", "nebivolol generis", "cloridrato de nebivolol"
    ],
    "salbutamol": [
        "salbutamol", "ventilan", "sulfato de salbutamol"
    ],
    "budesonide_formoterol": [
        "budesonide_formoterol", "budesonida + formoterol", "budesonida/formoterol",
        "budesonida e formoterol", "symbicort"
    ],
        "mexazolam": [
        "mexazolam", "sedoxil"
    ],
    "levothyroxine": [
        "levothyroxine", "levothyroxine sodium", "levotiroxina",
        "levotiroxina sodica", "eutirox", "letter", "tirosint", "synthroid"
    ],
    "allopurinol": [
        "allopurinol", "alopurinol", "zyloric"
    ],
    "colchicine": [
        "colchicine", "colchicina"
    ],
    "loratadine": [
        "loratadine", "loratadina", "claritine"
    ],
    "cetirizine": [
        "cetirizine", "cetirizina", "dicloridrato de cetirizina", "zyrtec"
    ],
    "tamsulosin": [
        "tamsulosin", "tamsulosin hydrochloride", "tansulosina", "tamsulosina",
        "cloridrato de tansulosina", "omnic"
    ],
    "finasteride": [
        "finasteride", "finasterida", "proscar", "propecia"
    ],
    "ciprofloxacin": [
        "ciprofloxacin", "ciprofloxacina", "cloridrato de ciprofloxacina",
        "ciproxina", "cipro"
    ],
    "fosfomycin": [
        "fosfomycin", "fosfomicina", "fosfomicina trometamol", "monuril"
    ],
    "nitrofurantoin": [
        "nitrofurantoin", "nitrofurantoina", "furadantina",
        "macrobid", "macrodantin"
    ],
    "alendronic_acid": [
        "alendronic acid", "alendronate", "alendronate sodium", "alendronato",
        "alendronato de sodio", "acido alendronico", "fosamax"
    ],
    "calcium_vitamin_d": [
        "calcium_vitamin_d", "calcium vitamin d",
        "carbonato de calcio + colecalciferol", "calcio + vitamina d",
        "calcio e vitamina d", "calcium carbonate / cholecalciferol",
        "cholecalciferol", "colecalciferol", "caltrate"
    ],
    "fluoxetine": [
        "fluoxetine", "fluoxetina", "cloridrato de fluoxetina", "prozac"
    ],
    "trazodone": [
        "trazodone", "trazodona", "cloridrato de trazodona",
        "triticum", "desyrel"
    ],
    "diclofenac": [
        "diclofenac", "diclofenaco", "diclofenac sodium",
        "diclofenac de sodio", "voltaren", "voltarene", "cataflam"
    ],
    "codeine": [
        "codeine", "codeina", "codeine phosphate", "fosfato de codeina"
    ],
    "pantoprazole": [
        "pantoprazole", "pantoprazol", "pantoprazol generis",
        "protonix", "controloc", "pantoc"
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

# Termos clínicos portugueses correntes que remetem para os identificadores
# de condição usados na base de conhecimento.
CONDITION_SYNONYMS = {
    "hipertensao": "hypertension",
    "hipertensao arterial": "hypertension",
    "hta": "hypertension",
    "insuficiencia renal": "renal_disease",
    "doenca renal cronica": "renal_disease",
    "drc": "renal_disease",
    "insuficiencia cardiaca": "heart_failure",
    "fibrilhacao auricular": "atrial_fibrillation",
    "enfarte agudo do miocardio": "recent_myocardial_infarction",
    "insuficiencia hepatica grave": "severe_hepatic_impairment",
    "doenca hepatica grave": "severe_hepatic_impairment",
    "ulcera gastrica ativa": "active_gi_ulcer",
    "ulcera peptica ativa": "active_gi_ulcer",
    "hemorragia ativa": "active_bleeding",
    "gravidez": "pregnancy",
    "gravida": "pregnancy",
    "amamentacao": "breastfeeding",
    "aleitamento": "breastfeeding",
    "bradicardia": "bradycardia",
    "hipocaliemia": "hypokalemia",
    "hipocalemia": "hypokalemia",
    "hipomagnesemia": "hypomagnesemia",
    "hipercalcemia": "hypercalcemia",
    "desidratacao": "dehydration",
    "prolongamento do qt": "qt_prolongation",
    "sindrome do qt longo": "qt_prolongation",
    "miastenia gravis": "myasthenia_gravis",
    "apneia do sono": "sleep_apnea",
    "epilepsia nao controlada": "uncontrolled_epilepsy",
    "doenca arterial periferica": "peripheral_arterial_disease",
    "cardiopatia isquemica": "ischemic_heart_disease",
    "angioedema por ieca": "history_acei_angioedema",
    "dislipidemia": "dyslipidemia",
}


def normalize_condition_id(description: str) -> str:
    if not description:
        return ""

    raw = description.strip().lower()

    # Identificadores internos escritos diretamente, por exemplo
    # "active_gi_ulcer". Sao devolvidos tal e qual, sem passar pela
    # heuristica de texto livre.
    if "_" in raw and " " not in raw:
        return raw

    text = normalize_text(description)

    if text in CONDITION_SYNONYMS:
        return CONDITION_SYNONYMS[text]

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

    if "hypothyroid" in text or "hipotiroidismo" in text:
        return "hypothyroidism"

    if "gout" in text or "gota" in text:
        return "gout"

    if "hyperuricemia" in text or "hiperuricemia" in text:
        return "hyperuricemia"

    if "urinary tract infection" in text or "cystitis" in text \
            or "cistite" in text or "infecao urinaria" in text:
        return "urinary_tract_infection"

    if "prostatic hyperplasia" in text or "hiperplasia benigna" in text:
        return "benign_prostatic_hyperplasia"

    if "osteoporosis" in text or "osteoporose" in text:
        return "osteoporosis"

    if "vitamin d deficiency" in text or "deficiencia de vitamina d" in text \
            or "calcium deficiency" in text:
        return "calcium_vitamin_d_deficiency"

    if "allergic rhinitis" in text or "rinite" in text or "hay fever" in text:
        return "allergic_rhinitis"

    if "urticaria" in text or "hives" in text:
        return "urticaria"

    if "insomnia" in text or "insonia" in text:
        return "insomnia"

    if "reflux" in text or "refluxo" in text or "gerd" in text or "drge" in text:
        return "gerd"

    if "asthma" in text or "asma" in text:
        return "asthma"

    if "chronic obstructive" in text or "copd" in text or "dpoc" in text:
        return "copd"

    if "cough" in text or "tosse" in text:
        return "cough"

    if "depress" in text:
        return "depression"

    if "anxiety" in text or "ansiedade" in text:
        return "anxiety"

    if "pain" in text or "dor" in text:
        return "pain"

    if "arthritis" in text or "inflammation" in text or "inflamacao" in text:
        return "inflammation"

    if "fever" in text or "febre" in text:
        return "fever"

    return text.strip()


def infer_main_problem(conditions: list[str]) -> str:
    joined = " ".join(conditions).lower()

    if "gout" in joined or "gota" in joined:
        return "gout"

    if "urinary_tract_infection" in joined or "cistite" in joined:
        return "urinary_tract_infection"

    if "hypothyroidism" in joined or "hipotiroidismo" in joined:
        return "hypothyroidism"

    if "benign_prostatic_hyperplasia" in joined or "hiperplasia benigna" in joined:
        return "benign_prostatic_hyperplasia"

    if "osteoporosis" in joined or "osteoporose" in joined:
        return "osteoporosis"

    if "allergic_rhinitis" in joined or "rinite" in joined:
        return "allergic_rhinitis"

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
        "inflammation": "inflammation",
        "artrite": "inflammation",
        "arthritis": "inflammation",

        "infecao": "infection",
        "infection": "infection",
        "infeccao": "infection",

        "febre": "fever",
        "fever": "fever",

        "hipertensao": "hypertension",
        "hypertension": "hypertension",

        "diabetes": "diabetes",

        "insuficiencia cardiaca": "heart_failure",
        "heart failure": "heart_failure",

        "arritmia": "arrhythmia",
        "arrhythmia": "arrhythmia",

        "ulcera": "active_gi_ulcer",
        "ulcer": "active_gi_ulcer",
        "protecao gastrica": "gastric_protection",
        "refluxo": "gerd",
        "gerd": "gerd",
        "drge": "gerd",

        "gota": "gout",
        "gout": "gout",
        "hiperuricemia": "hyperuricemia",

        "infecao urinaria": "urinary_tract_infection",
        "itu": "urinary_tract_infection",
        "cistite": "urinary_tract_infection",
        "urinary tract infection": "urinary_tract_infection",

        "hipotiroidismo": "hypothyroidism",
        "hypothyroidism": "hypothyroidism",

        "hiperplasia benigna da prostata": "benign_prostatic_hyperplasia",
        "hbp": "benign_prostatic_hyperplasia",

        "osteoporose": "osteoporosis",
        "osteoporosis": "osteoporosis",

        "rinite alergica": "allergic_rhinitis",
        "rinite": "allergic_rhinitis",
        "urticaria": "urticaria",

        "insonia": "insomnia",
        "insomnia": "insomnia",

        "depressao": "depression",
        "depression": "depression",
        "ansiedade": "anxiety",
        "anxiety": "anxiety",

        "asma": "asthma",
        "asthma": "asthma",
        "dpoc": "copd",
        "copd": "copd",

        "tosse": "cough",
        "cough": "cough",        
    }

    return mapping.get(text, text)

# Alergias declaradas ao nível da classe terapêutica. Aceitam-se variantes de
# língua, de número e com prefixo, remetendo todas para o identificador da
# classe usado na base de conhecimento.
ALLERGY_CLASS_SYNONYMS = {
    "penicilina": "penicilina",
    "penicilinas": "penicilina",
    "penicillin": "penicilina",
    "penicillins": "penicilina",
    "betalactamico": "penicilina",
    "betalactamicos": "penicilina",
    "beta lactamicos": "penicilina",
    "macrolido": "macrolido",
    "macrolidos": "macrolido",
    "macrolide": "macrolido",
    "quinolona": "fluoroquinolona",
    "quinolonas": "fluoroquinolona",
    "fluoroquinolona": "fluoroquinolona",
    "fluoroquinolonas": "fluoroquinolona",
    "aine": "aine",
    "aines": "aine",
    "nsaid": "aine",
    "anti inflamatorio nao esteroide": "aine",
    "opioide": "opioide",
    "opioides": "opioide",
    "estatina": "estatina",
    "estatinas": "estatina",
    "sulfonilureia": "sulfonilureia",
    "sulfonilureias": "sulfonilureia",
}

_ALLERGY_PREFIXES = (
    "alergia a", "alergia ao", "alergia as", "alergia aos", "alergia",
    "alergico a", "alergica a", "hipersensibilidade a", "hipersensibilidade",
)


def normalize_allergy_terms(raw: str) -> set:
    """Devolve as formas sob as quais uma alergia declarada deve ser procurada:
    o texto original em minusculas, o texto normalizado, o texto sem prefixos
    correntes e, quando aplicavel, o identificador da classe terapeutica."""
    if not raw or not raw.strip():
        return set()

    terms = {raw.strip().lower(), normalize_text(raw)}

    for prefix in _ALLERGY_PREFIXES:
        for term in list(terms):
            if term.startswith(prefix + " "):
                terms.add(term[len(prefix) + 1:].strip())

    for term in list(terms):
        if term in ALLERGY_CLASS_SYNONYMS:
            terms.add(ALLERGY_CLASS_SYNONYMS[term])

    return {t for t in terms if t}
