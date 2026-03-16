

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from IPython.display import clear_output

model = AutoModelForSequenceClassification.from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
clear_output()
```


```python
print(model.num_labels)
```
```output
19
```


```python
print(model.config)
```
```output
CamembertConfig {
  "add_cross_attention": false,
  "architectures": [
    "CamembertForSequenceClassification"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 5,
  "classifier_dropout": null,
  "dtype": "float32",
  "eos_token_id": 6,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "Emo",
    "1": "Comportementale",
    "2": "Designee",
    "3": "Montree",
    "4": "Suggeree",
    "5": "Base",
    "6": "Complexe",
    "7": "Admiration",
    "8": "Autre",
    "9": "Colere",
    "10": "Culpabilite",
    "11": "Degout",
    "12": "Embarras",
    "13": "Fierte",
    "14": "Jalousie",
    "15": "Joie",
    "16": "Peur",
    "17": "Surprise",
    "18": "Tristesse"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "is_decoder": false,
  "label2id": {
    "Admiration": 7,
    "Autre": 8,
    "Base": 5,
    "Colere": 9,
    "Complexe": 6,
    "Comportementale": 1,
    "Culpabilite": 10,
    "Degout": 11,
    "Designee": 2,
    "Embarras": 12,
    "Emo": 0,
    "Fierte": 13,
    "Jalousie": 14,
    "Joie": 15,
    "Montree": 3,
    "Peur": 16,
    "Suggeree": 4,
    "Surprise": 17,
    "Tristesse": 18
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "camembert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "problem_type": "multi_label_classification",
  "transformers_version": "5.0.0",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 32005
}

```

```python
model.classifier
```
```output
CamembertClassificationHead(
  (dense): Linear(in_features=768, out_features=768, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (out_proj): Linear(in_features=768, out_features=19, bias=True)
)
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math

# choose a model that is a sequence-classification head (replace with your model)
model_name = "TextToKids/CamemBERT-base-EmoTextToKids"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# pick device and move model there
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_emotions(prev, current, after, max_length=512):
    """
    Tokenize the formatted text, move inputs to the model device, run forward,
    return sorted list of {"label": str, "probability": float}.
    """
    # Format text the way your model expects (adapt if your model uses a different format)
    before_text = prev if prev is not None else "</s>"
    after_text = after if after is not None else "</s>"
    text = f"before: {before_text} current: {current} after: {after_text}"
    # Tokenize and create tensors
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )

    # Move all tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # shape: (num_labels,)

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Try to obtain label names from model config (fallback to numeric labels)
    id2label = None
    if hasattr(model.config, "id2label"):
        # id2label often exists as a dict mapping ints->str
        id2label = model.config.id2label
    elif hasattr(model.config, "label2id"):
        # invert label2id if only label2id exists
        id2label = {v: k for k, v in model.config.label2id.items()}

    if id2label is None:
        labels = [str(i) for i in range(len(probs))]
    else:
        # ensure ordered by index
        labels = [id2label[i] if i in id2label else str(i) for i in range(len(probs))]

    # Build list of predictions sorted by probability desc
    preds = [{"label": labels[i], "probability": float(probs[i])} for i in range(len(probs))]
    preds.sort(key=lambda x: x["probability"], reverse=True)
    return preds

# --- your tests ---
test_cases = [
    {
        "name": "Test 1 : Joie extrême (Désignée)",
        "prev": None,
        "target": "Je suis vraiment très heureux, c'est une joie immense !",
        "next": None
    },
    {
        "name": "Test 2 : Colère extrême (Comportementale)",
        "prev": None,
        "target": "Il frappa violemment du poing sur la table en hurlant.",
        "next": None
    },
    {
        "name": "Test 3 : Phrase du dataset (Antisémitisme / Tristesse-Colère)",
        "prev": None,
        "target": "Ces dernières semaines, plusieurs actes antisémites, c’est-à-dire dirigés contre des personnes juives ou des symboles juifs, ont été dénoncés.",
        "next": "Des manifestations ont été organisées en France pour dire «non à l’antisémitisme» et au rejet de l’autre."
    }
]

for case in test_cases:
    print(f"\n\n{'='*50}\n{case['name']}\n{'='*50}")
    predictions = predict_emotions(case["prev"], case["target"], case["next"])

    print("\nTop 5 des prédictions :")
    for res in predictions[:5]:
        print(f" - {res['label']:<30} : {res['probability']*100:.2f}%")
```
```output
Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]

==================================================
Test 1 : Joie extrême (Désignée)
==================================================

Top 5 des prédictions :
 - Emo                            : 53.16%
 - Base                           : 42.98%
 - Joie                           : 3.51%
 - Designee                       : 0.28%
 - Montree                        : 0.07%


==================================================
Test 2 : Colère extrême (Comportementale)
==================================================

Top 5 des prédictions :
 - Comportementale                : 70.11%
 - Base                           : 21.31%
 - Emo                            : 8.46%
 - Colere                         : 0.10%
 - Peur                           : 0.02%


==================================================
Test 3 : Phrase du dataset (Antisémitisme / Tristesse-Colère)
==================================================

Top 5 des prédictions :
 - Emo                            : 97.67%
 - Comportementale                : 1.38%
 - Colere                         : 0.78%
 - Base                           : 0.12%
 - Autre                          : 0.05%
```

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE)
)
model.eval()

id2label     = model.config.id2label
num_labels   = len(id2label)                           # 19
emo_names    = [id2label[i] for i in range(num_labels)]
print(f"[info] {num_labels} emotion/category labels: {emo_names}")

# Format expected by EmoTextToKids
@torch.no_grad()
def predict_emotions(
    target_sentence: str,
    previous_sentence: str | None = None,
    next_sentence: str | None = None,
) -> list[float]:
    """Return a list of 19 sigmoid probabilities for *target_sentence*."""
    prev_str   = previous_sentence or ""
    target_str = target_sentence   or ""
    next_str   = next_sentence     or ""
    eos        = tokenizer.eos_token

    text = f"before: {prev_str}{eos}current: {target_str}{eos}after: {next_str}{eos}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    logits = model(**inputs).logits                    # (1, 19)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist() # avec sigmoid
    return probs
```
```output

```

```python
import pandas as pd
df = pd.read_excel("processed_predictions.xlsx")
```



```python
df.columns
```
```output
Index(['document', 'sent_id', 'sentence', 'emo', 'emo_comportementale',
       'emo_designee', 'emo_montree', 'emo_suggeree', 'emo_base',
       'emo_complexe', 'autre', 'admiration', 'colere', 'culpabilite',
       'degout', 'embarras', 'fierte', 'jalousie', 'joie', 'peur', 'surprise',
       'tristesse'],
      dtype='object')
```

```python
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ═══════════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE)
    .eval()
)

df        = pd.read_excel("/content/processed_predictions.xlsx")
sentences = df["sentence"].astype(str).tolist()
EOS       = tokenizer.eos_token                            # </s>

# LABEL_COLS[j] = colonne XLSX correspondant à l'index j du modèle (id2label)
LABEL_COLS = [
    "emo", "emo_comportementale", "emo_designee", "emo_montree",
    "emo_suggeree", "emo_base", "emo_complexe",
    "admiration", "autre",
    "colere", "culpabilite", "degout", "embarras",
    "fierte", "jalousie", "joie", "peur", "surprise", "tristesse",
]
Y = df[LABEL_COLS].values.astype(int)                      # (101, 19)
N, L = Y.shape
print(f"Référence : {N} phrases × {L} labels = {N*L} ules\n")

# ═══════════════════════════════════════════════════════════════
#  2. FORMATS D'ENTRÉE CANDIDATS
#     (on ne sait pas comment le web formate avant d'envoyer
#      au tokenizer → on teste plusieurs variantes)
# ═══════════════════════════════════════════════════════════════
TEMPLATES = {
    "raw":
        lambda s: s,
    "bca_eos_nospace":
        lambda s: f"before: {EOS}current: {s}{EOS}after: {EOS}",
    "bca_eos_space":
        lambda s: f"before: {EOS} current: {s} after: {EOS}",
    "bca_empty":
        lambda s: f"before:  current: {s} after: ",
    "bca_none_str":
        lambda s: f"before: None current: {s} after: None",
}

# ═══════════════════════════════════════════════════════════════
#  3. INFÉRENCE  –  sigmoid (multi-label), dropout désactivé
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_sigmoid(template_fn):
    """Renvoie la matrice (N, L) des probabilités sigmoïdes."""
    P = np.empty((N, L), dtype=np.float64)
    for i, s in enumerate(sentences):
        enc = tokenizer(
            template_fn(s),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        logits = model(**enc).logits.squeeze(0)          # (19,)
        P[i] = torch.sigmoid(logits).cpu().double().numpy()
    return P

# ═══════════════════════════════════════════════════════════════
#  4. OPTIMISATION DES SEUILS (un par label)
#     Pour chaque label j on parcourt tous les « midpoints »
#     entre valeurs consécutives de σ(logit_j) et on garde
#     celui qui minimise les erreurs binaires.
# ═══════════════════════════════════════════════════════════════
def sweep_thresholds(P, Y):
    thresholds = np.empty(L)
    total_err  = 0
    for j in range(L):
        pj, yj = P[:, j], Y[:, j]
        vals = np.sort(np.unique(pj))
        # seuils candidats : entre chaque paire de probas + bornes
        cuts = np.concatenate([
            [vals[0] - 1e-9],
            (vals[:-1] + vals[1:]) / 2.0,
            [vals[-1] + 1e-9],
        ])
        errs = np.array([
            int(((pj >= t).astype(int) != yj).sum()) for t in cuts
        ])
        idx_best = int(errs.argmin())
        thresholds[j] = cuts[idx_best]
        total_err += errs[idx_best]
    return thresholds, total_err

# ═══════════════════════════════════════════════════════════════
#  5. RECHERCHE DU MEILLEUR FORMAT
# ═══════════════════════════════════════════════════════════════
winner = dict(err=N * L + 1)

for name, fn in TEMPLATES.items():
    P   = infer_sigmoid(fn)
    thr, err_opt = sweep_thresholds(P, Y)
    err_05 = int(((P >= 0.5).astype(int) != Y).sum())

    star = " ★ PERFECT" if err_opt == 0 else ""
    print(f"  {name:20s}  err@0.5={err_05:4d}   "
          f"err@opt={err_opt:4d}  /{N*L}{star}")

    if err_opt < winner["err"]:
        winner = dict(name=name, fn=fn, P=P, thr=thr, err=err_opt)
    if err_opt == 0:                       # inutile de continuer
        break

# ═══════════════════════════════════════════════════════════════
#  6. RAPPORT DÉTAILLÉ
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*62}")
print(f"  Meilleur template  : {winner['name']}")
print(f"  Erreurs restantes  : {winner['err']} / {N*L}")
print(f"{'═'*62}\n")

P, thr = winner["P"], winner["thr"]

print("Seuils optimaux par label :")
for j in range(L):
    preds  = (P[:, j] >= thr[j]).astype(int)
    ne     = int((preds != Y[:, j]).sum())
    status = "✓" if ne == 0 else f"✗ {ne} erreur(s)"
    print(f"  {LABEL_COLS[j]:<25s}  seuil = {thr[j]:.8f}   {status}")

Y_hat  = (P >= thr).astype(int)
row_ok = int((Y_hat == Y).all(axis=1).sum())
print(f"\nLignes parfaitement reproduites : {row_ok} / {N}")
print(f"Précision ule par ule  : "
      f"{(Y_hat == Y).sum()} / {N*L}  "
      f"({(Y_hat == Y).mean():.4%})")

# Détail des éventuelles erreurs résiduelles
bad = np.where(~(Y_hat == Y).all(axis=1))[0]
if bad.size:
    print(f"\nLignes en erreur ({bad.size}) :")
    for i in bad:
        for j in np.where(Y_hat[i] != Y[i])[0]:
            print(f"  row {i:3d}  {LABEL_COLS[j]:<22s}  "
                  f"prédit={Y_hat[i,j]}  vrai={Y[i,j]}  "
                  f"σ={P[i,j]:.8f}")
else:
    print("\n✅  Aucune erreur : les outputs web sont parfaitement reproduits !")

# ═══════════════════════════════════════════════════════════════
#  7. FONCTION RÉUTILISABLE
# ═══════════════════════════════════════════════════════════════
FINAL_TEMPLATE   = winner["fn"]
FINAL_THRESHOLDS = winner["thr"]          # np.array de shape (19,)

@torch.no_grad()
def predict_binary(sentence: str) -> dict[str, int]:
    """
    Reproduit exactement la sortie binaire du modèle web
    pour une phrase isolée.
    """
    text = FINAL_TEMPLATE(sentence)
    enc  = tokenizer(text, return_tensors="pt",
                     truncation=True, max_length=512).to(DEVICE)
    logits = model(**enc).logits.squeeze(0)
    probs  = torch.sigmoid(logits).cpu().numpy()
    binary = (probs >= FINAL_THRESHOLDS).astype(int)
    return {LABEL_COLS[j]: int(binary[j]) for j in range(L)}


# ── Validation finale sur toutes les lignes ──
print("\n\nValidation complète :")
all_ok = True
for i in range(N):
    pred = predict_binary(sentences[i])
    true = {LABEL_COLS[j]: int(Y[i, j]) for j in range(L)}
    if pred != true:
        all_ok = False
        print(f"  ✗ row {i}: {sentences[i][:55]}…")
if all_ok:
    print("  ✓ 101/101 lignes identiques aux outputs web.")

# ── Export des seuils pour réutilisation ──
thresholds_dict = {LABEL_COLS[j]: float(FINAL_THRESHOLDS[j]) for j in range(L)}
print(f"\n# Seuils à copier-coller :\nTHRESHOLDS = {thresholds_dict}")
```
```output
Référence : 101 phrases × 19 labels = 1919 ules

  raw                   err@0.5= 222   err@opt= 191  /1919
  bca_eos_nospace       err@0.5= 140   err@opt=  93  /1919
  bca_eos_space         err@0.5= 153   err@opt=  97  /1919
  bca_empty             err@0.5= 162   err@opt= 112  /1919
  bca_none_str          err@0.5= 149   err@opt= 101  /1919

══════════════════════════════════════════════════════════════
  Meilleur template  : bca_eos_nospace
  Erreurs restantes  : 93 / 1919
══════════════════════════════════════════════════════════════

Seuils optimaux par label :
  emo                        seuil = 0.00223477   ✗ 9 erreur(s)
  emo_comportementale        seuil = 0.19961804   ✗ 1 erreur(s)
  emo_designee               seuil = 0.08523063   ✗ 2 erreur(s)
  emo_montree                seuil = 0.77406597   ✗ 10 erreur(s)
  emo_suggeree               seuil = 0.04274027   ✗ 11 erreur(s)
  emo_base                   seuil = 0.01366377   ✗ 13 erreur(s)
  emo_complexe               seuil = 0.28400013   ✗ 4 erreur(s)
  admiration                 seuil = 0.00797915   ✗ 17 erreur(s)
  autre                      seuil = 0.99979788   ✓
  colere                     seuil = 0.21916297   ✗ 17 erreur(s)
  culpabilite                seuil = 0.03128876   ✓
  degout                     seuil = 0.07657739   ✓
  embarras                   seuil = 0.62047952   ✗ 2 erreur(s)
  fierte                     seuil = 0.01610760   ✓
  jalousie                   seuil = 0.00278062   ✓
  joie                       seuil = 0.03363183   ✗ 2 erreur(s)
  peur                       seuil = 0.17203747   ✓
  surprise                   seuil = 0.50669206   ✓
  tristesse                  seuil = 0.01712704   ✗ 5 erreur(s)

Lignes parfaitement reproduites : 53 / 101
Précision ule par ule  : 1826 / 1919  (95.1537%)

Lignes en erreur (48) :
  row   0  emo_suggeree            prédit=1  vrai=0  σ=0.07088140
  row   1  admiration              prédit=0  vrai=1  σ=0.00006661
  row   2  admiration              prédit=0  vrai=1  σ=0.00030836
  row   4  emo_designee            prédit=1  vrai=0  σ=0.43613279
  row   4  tristesse               prédit=1  vrai=0  σ=0.76184762
  row   5  admiration              prédit=0  vrai=1  σ=0.00010354
  row   6  emo_montree             prédit=0  vrai=1  σ=0.00022179
  row   6  emo_base                prédit=0  vrai=1  σ=0.00054577
  row   6  colere                  prédit=0  vrai=1  σ=0.00171229
  row   7  emo_suggeree            prédit=0  vrai=1  σ=0.01164239
  row   7  emo_complexe            prédit=0  vrai=1  σ=0.00763510
  row   7  embarras                prédit=0  vrai=1  σ=0.08505897
  row   9  emo                     prédit=1  vrai=0  σ=0.02413656
  row   9  tristesse               prédit=1  vrai=0  σ=0.10674835
  row  11  colere                  prédit=1  vrai=0  σ=0.42268115
  row  13  emo                     prédit=0  vrai=1  σ=0.00014735
  row  13  emo_montree             prédit=0  vrai=1  σ=0.00001568
  row  13  emo_base                prédit=0  vrai=1  σ=0.00009072
  row  13  colere                  prédit=0  vrai=1  σ=0.00002410
  row  20  emo_montree             prédit=0  vrai=1  σ=0.00158251
  row  20  emo_base                prédit=0  vrai=1  σ=0.00036005
  row  20  colere                  prédit=0  vrai=1  σ=0.00183107
  row  23  emo                     prédit=0  vrai=1  σ=0.00016691
  row  23  emo_base                prédit=0  vrai=1  σ=0.00010185
  row  23  colere                  prédit=0  vrai=1  σ=0.00001344
  row  24  emo_base                prédit=0  vrai=1  σ=0.00112886
  row  24  admiration              prédit=0  vrai=1  σ=0.00003083
  row  24  joie                    prédit=0  vrai=1  σ=0.00018581
  row  25  emo                     prédit=0  vrai=1  σ=0.00018959
  row  25  emo_suggeree            prédit=0  vrai=1  σ=0.00013757
  row  25  emo_base                prédit=0  vrai=1  σ=0.00006674
  row  25  colere                  prédit=0  vrai=1  σ=0.00005459
  row  26  admiration              prédit=0  vrai=1  σ=0.00009211
  row  27  emo                     prédit=1  vrai=0  σ=0.99792957
  row  27  emo_montree             prédit=1  vrai=0  σ=0.96238053
  row  27  emo_base                prédit=1  vrai=0  σ=0.99655652
  row  27  colere                  prédit=1  vrai=0  σ=0.99711561
  row  28  emo_suggeree            prédit=0  vrai=1  σ=0.02412403
  row  30  emo_montree             prédit=0  vrai=1  σ=0.15173286
  row  30  admiration              prédit=0  vrai=1  σ=0.00009614
  row  31  emo_montree             prédit=0  vrai=1  σ=0.00659943
  row  31  admiration              prédit=0  vrai=1  σ=0.00002628
  row  31  colere                  prédit=0  vrai=1  σ=0.01191117
  row  32  emo                     prédit=1  vrai=0  σ=0.01094236
  row  33  admiration              prédit=0  vrai=1  σ=0.00011222
  row  38  emo                     prédit=0  vrai=1  σ=0.00016568
  row  38  emo_montree             prédit=0  vrai=1  σ=0.00001756
  row  38  emo_base                prédit=0  vrai=1  σ=0.00009148
  row  38  colere                  prédit=0  vrai=1  σ=0.00001950
  row  39  admiration              prédit=0  vrai=1  σ=0.00007458
  row  40  tristesse               prédit=1  vrai=0  σ=0.12018621
  row  43  emo_suggeree            prédit=1  vrai=0  σ=0.37861603
  row  43  admiration              prédit=0  vrai=1  σ=0.00001866
  row  46  emo_complexe            prédit=1  vrai=0  σ=0.71721643
  row  48  emo_designee            prédit=1  vrai=0  σ=0.76903224
  row  48  emo_base                prédit=1  vrai=0  σ=0.35165396
  row  49  emo_montree             prédit=0  vrai=1  σ=0.13325179
  row  49  admiration              prédit=0  vrai=1  σ=0.00001856
  row  50  emo                     prédit=1  vrai=0  σ=0.99898022
  row  53  emo_montree             prédit=0  vrai=1  σ=0.31195024
  row  63  emo_complexe            prédit=0  vrai=1  σ=0.06066672
  row  63  colere                  prédit=1  vrai=0  σ=0.53290772
  row  63  embarras                prédit=0  vrai=1  σ=0.17406790
  row  64  admiration              prédit=0  vrai=1  σ=0.00010575
  row  65  emo_suggeree            prédit=0  vrai=1  σ=0.01474741
  row  65  colere                  prédit=0  vrai=1  σ=0.10842619
  row  67  admiration              prédit=0  vrai=1  σ=0.00002166
  row  69  emo_base                prédit=0  vrai=1  σ=0.00435079
  row  69  colere                  prédit=0  vrai=1  σ=0.00839802
  row  69  tristesse               prédit=0  vrai=1  σ=0.00021044
  row  72  emo_suggeree            prédit=0  vrai=1  σ=0.00008254
  row  72  emo_base                prédit=0  vrai=1  σ=0.00004825
  row  72  colere                  prédit=0  vrai=1  σ=0.00002570
  row  73  colere                  prédit=1  vrai=0  σ=0.99626786
  row  73  tristesse               prédit=0  vrai=1  σ=0.00061428
  row  74  emo_suggeree            prédit=0  vrai=1  σ=0.01590493
  row  74  colere                  prédit=1  vrai=0  σ=0.98751903
  row  76  emo                     prédit=1  vrai=0  σ=0.53266954
  row  76  emo_base                prédit=1  vrai=0  σ=0.14513206
  row  76  colere                  prédit=1  vrai=0  σ=0.81902397
  row  79  colere                  prédit=0  vrai=1  σ=0.04950316
  row  79  joie                    prédit=1  vrai=0  σ=0.98682833
  row  80  admiration              prédit=0  vrai=1  σ=0.00014267
  row  86  admiration              prédit=0  vrai=1  σ=0.00007568
  row  88  emo_suggeree            prédit=0  vrai=1  σ=0.01864407
  row  88  emo_complexe            prédit=0  vrai=1  σ=0.00003632
  row  90  emo_suggeree            prédit=0  vrai=1  σ=0.00009565
  row  93  emo_suggeree            prédit=1  vrai=0  σ=0.06674918
  row  98  emo_base                prédit=1  vrai=0  σ=0.13114271
  row  98  admiration              prédit=0  vrai=1  σ=0.00002078
  row  99  emo_comportementale     prédit=1  vrai=0  σ=0.26118147
  row 100  emo_montree             prédit=0  vrai=1  σ=0.03189901
  row 100  admiration              prédit=0  vrai=1  σ=0.00008040


Validation complète :
  ✗ row 0: Alors julie t'ai dégeulasse de sortir avec une gadji ba…
  ✗ row 1: allo mme est homophobe.…
  ✗ row 2: mais tu veut quoi toi retourne avec ta gadji et assume …
  ✗ row 4: mais grave bande de lesbiennes vous degoutée aaaaah.…
  ✗ row 5: mais olalala frr assume t'aime les ch*ttes mais tu dégo…
  ✗ row 6: laissé la tranquille elle n'est pas avec elle elle vous…
  ✗ row 7: mais assumer vous degouter bande de crasseuse.…
  ✗ row 9: grave on a tous vu.…
  ✗ row 11: grave abuser vous me degoutez.…
  ✗ row 13: mais assume grosse pute.…
  ✗ row 20: mais cest des gouines.…
  ✗ row 23: bande de gouines.…
  ✗ row 24: la réconforter et la galoche après.…
  ✗ row 25: bande de crasseuse.…
  ✗ row 26: taimes les filles.…
  ✗ row 27: y'a rien à assumer,  puisqu'y a r…
  ✗ row 28: burk sa me degoute…
  ✗ row 30: assume que t'aime les filles c tout connasse.…
  ✗ row 31: elle etait juste en train de la rassurer cest rien.…
  ✗ row 32: cul de crasseuse.…
  ✗ row 33: on fait ce qu'on veut grosse chienne.…
  ✗ row 38: on dit que la vérité grosse pute.…
  ✗ row 39: chienne que tes ont aimes pas les lesbienne ici.…
  ✗ row 40: bref venais on les vire de la classe c'est grosse crass…
  ✗ row 43: on veut pas de sa ici.…
  ✗ row 46: me degoute.…
  ✗ row 48: c'est bon chacun fait ce qu'il veut aucune d'entre vous…
  ✗ row 49: ça se voit vous êtes pas des potes si ça vous est jamai…
  ✗ row 50: on sen blkkkkk mais fort.…
  ✗ row 53: assume que t gouine c tout grosse pute va.…
  ✗ row 63: grave vous me degouter.…
  ✗ row 64: t'as un problème va te faire soigner sale homophobe.…
  ✗ row 65: exactement stv on te fait un cour crasseuse.…
  ✗ row 67: mais calmez-vous chaqu'un s'est préférence.…
  ✗ row 69: on va dire a un gadjo de la violer sa va lui remettre l…
  ✗ row 72: elles vont voir ce que c de ce faire peter le q.…
  ✗ row 73: bande de pute va mais grave elle vont voir.…
  ✗ row 74: mais t'es une grande malade toi.…
  ✗ row 76: t'as des problèmes pour dire des choses pareil.…
  ✗ row 79: je rigole roh.…
  ✗ row 80: ftg tu vas kiffer.…
  ✗ row 86: laissé la tranquille elle a le droit d'aimer qui elle v…
  ✗ row 88: mais grave que des traitres.…
  ✗ row 90: si vous ne disiez pas n'importe quoi aussi,  les gens s…
  ✗ row 93: bande de crasseux meme arthur on va te degager de la cl…
  ✗ row 98: tchaoooooo.…
  ✗ row 99: allez-y dégagez nous vous allez faire quoi ?.…
  ✗ row 100: c'est les homophobes qu'on dégage pas nous.…

# Seuils à copier-coller :
THRESHOLDS = {'emo': 0.0022347719641402364, 'emo_comportementale': 0.19961804151535034, 'emo_designee': 0.08523062989115715, 'emo_montree': 0.7740659713745117, 'emo_suggeree': 0.04274027422070503, 'emo_base': 0.013663773192092776, 'emo_complexe': 0.28400012850761414, 'admiration': 0.007979153655469418, 'autre': 0.9997978816495666, 'colere': 0.2191629707813263, 'culpabilite': 0.0312887551949749, 'degout': 0.07657738875014877, 'embarras': 0.6204795241355896, 'fierte': 0.016107599319649696, 'jalousie': 0.002780617981908679, 'joie': 0.0336318276822567, 'peur': 0.17203746836049652, 'surprise': 0.5066920565441251, 'tristesse': 0.01712703937664628}
```


# Tests pour voir si le modèle EMOTYC proposé sur l'interface web utilise les phrases du contexte

```python
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# ══════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = (
    AutoModelForSequenceClassification
    .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
    .to(DEVICE).eval()
)

df = pd.read_excel("/content/processed_predictions.xlsx")
sentences = df["sentence"].astype(str).tolist()
EOS = tokenizer.eos_token  # </s>
N = len(sentences)

LABELS = [
    "emo", "emo_comportementale", "emo_designee", "emo_montree",
    "emo_suggeree", "emo_base", "emo_complexe", "admiration", "autre",
    "colere", "culpabilite", "degout", "embarras", "fierte", "jalousie",
    "joie", "peur", "surprise", "tristesse",
]
Y = df[LABELS].values.astype(int)  # (101, 19)
K = len(LABELS)
print(f"Données : {N} phrases × {K} labels = {N*K} ules\n")tsli

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def find_best_thresholds(probs, targets):
    thr = np.empty(K); total = 0
    for j in range(K):
        p, y = probs[:, j], targets[:, j]
        v = np.unique(p)
        c = np.concatenate([[v[0]-1e-9], (v[:-1]+v[1:])/2, [v[-1]+1e-9]])
        e = np.array([((p >= t).astype(int) != y).sum() for t in c])
        thr[j] = c[e.argmin()]; total += e.min()
    return thr, total


# ══════════════════════════════════════════════════════
#  PHASE 1 · Test avec contexte voisin (i-1, i, i+1)
# ══════════════════════════════════════════════════════
print("═" * 65)
print(" PHASE 1 · Templates avec phrases voisines comme contexte")
print("═" * 65)

# Différentes options pour les bords (première / dernière phrase)
EDGE_OPTIONS = {
    "eos":   EOS,       # </s>
    "empty": "",        # chaîne vide
    "none":  "None",    # le mot "None"
}

# Différentes variantes du template
def make_templates():
    """Génère des combinaisons template × gestion des bords."""
    out = {}

    for edge_name, edge_val in EDGE_OPTIONS.items():
        # ─ Variante A : "before: X</s>current: Y</s>after: Z</s>"
        def fn_a(i, _ev=edge_val):
            prev = sentences[i-1] if i > 0 else _ev
            cur  = sentences[i]
            nxt  = sentences[i+1] if i < N-1 else _ev
            return f"before: {prev}{EOS}current: {cur}{EOS}after: {nxt}{EOS}"
        out[f"ctx_a_{edge_name}"] = fn_a

        # ─ Variante B : espaces autour des </s>
        def fn_b(i, _ev=edge_val):
            prev = sentences[i-1] if i > 0 else _ev
            cur  = sentences[i]
            nxt  = sentences[i+1] if i < N-1 else _ev
            return f"before: {prev} {EOS} current: {cur} {EOS} after: {nxt} {EOS}"
        out[f"ctx_b_{edge_name}"] = fn_b

        # ─ Variante C : </s> collé aux labels, espace après
        def fn_c(i, _ev=edge_val):
            prev = sentences[i-1] if i > 0 else _ev
            cur  = sentences[i]
            nxt  = sentences[i+1] if i < N-1 else _ev
            return f"before: {prev}{EOS} current: {cur}{EOS} after: {nxt}{EOS}"
        out[f"ctx_c_{edge_name}"] = fn_c

        # ─ Variante D : sans espace après "before:", etc.
        def fn_d(i, _ev=edge_val):
            prev = sentences[i-1] if i > 0 else _ev
            cur  = sentences[i]
            nxt  = sentences[i+1] if i < N-1 else _ev
            return f"before:{prev}{EOS}current:{cur}{EOS}after:{nxt}{EOS}"
        out[f"ctx_d_{edge_name}"] = fn_d

        # ─ Variante E : espace entre phrase et </s>
        def fn_e(i, _ev=edge_val):
            prev = sentences[i-1] if i > 0 else _ev
            cur  = sentences[i]
            nxt  = sentences[i+1] if i < N-1 else _ev
            return f"before: {prev} {EOS}current: {cur} {EOS}after: {nxt} {EOS}"
        out[f"ctx_e_{edge_name}"] = fn_e

    # ─ Aussi : sans contexte (baseline rappel)
    def fn_no_ctx(i):
        return f"before: {EOS}current: {sentences[i]}{EOS}after: {EOS}"
    out["no_ctx_baseline"] = fn_no_ctx

    return out

@torch.no_grad()
def compute_logits_ctx(template_fn):
    """template_fn prend un index i → string."""
    out = np.empty((N, K))
    for i in range(N):
        text = template_fn(i)
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(DEVICE)
        out[i] = model(**enc).logits.squeeze(0).cpu().numpy()
    return out


all_templates = make_templates()
best = dict(err=N*K+1)

for name, fn in all_templates.items():
    lg = compute_logits_ctx(fn)
    pr = sigmoid(lg)
    thr, err = find_best_thresholds(pr, Y)
    err05 = int(((pr >= 0.5).astype(int) != Y).sum())
    tag = " ★ PERFECT" if err == 0 else ""
    print(f"  {name:22s}  @0.5={err05:4d}  @opt={err:4d}{tag}")
    if err < best["err"]:
        best = dict(name=name, fn=fn, logits=lg, probs=pr,
                    thr=thr, err=err)
    if err == 0:
        break

print(f"\n  ► Meilleur : '{best['name']}' ({best['err']} erreurs)")
print(f"    (baseline sans contexte : rappel de la phase précédente = 93)\n")


# ══════════════════════════════════════════════════════
#  PHASE 2 · Détail par label (meilleur template)
# ══════════════════════════════════════════════════════
print("═" * 65)
print(f" PHASE 2 · Seuils optimaux [{best['name']}]")
print("═" * 65)

X = best["logits"].copy()
P = best["probs"].copy()
thr = best["thr"]

for j in range(K):
    preds = (P[:, j] >= thr[j]).astype(int)
    ne = int((preds != Y[:, j]).sum())
    status = "✓" if ne == 0 else f"✗ {ne} err"
    print(f"  {LABELS[j]:25s}  seuil={thr[j]:.6f}  {status}")


# ══════════════════════════════════════════════════════
#  PHASE 3 · Classifieurs multi-logits si nécessaire
# ══════════════════════════════════════════════════════
print(f"\n{'═' * 65}")
print(" PHASE 3 · Classifieurs (seuil → LogReg → DTree)")
print("═" * 65)

classifiers = [None] * K
for j in range(K):
    yj = Y[:, j]
    pj = P[:, j]

    # Strat 1 : seuil simple
    vals = np.unique(pj)
    cuts = np.concatenate([
        [vals[0]-1e-9], (vals[:-1]+vals[1:])/2, [vals[-1]+1e-9]
    ])
    errs = np.array([((pj >= c).astype(int) != yj).sum() for c in cuts])
    if errs.min() == 0:
        classifiers[j] = ("threshold", float(cuts[errs.argmin()]))
        print(f"  {LABELS[j]:25s}  seuil = {cuts[errs.argmin()]:.6f}"
              f"                ✓")
        continue

    # Strat 2 : LogReg
    if len(np.unique(yj)) > 1:
        lr = LogisticRegression(C=1e8, max_iter=50000, solver="lbfgs")
        lr.fit(X, yj)
        if (lr.predict(X) == yj).all():
            classifiers[j] = ("logreg", lr)
            print(f"  {LABELS[j]:25s}  LogReg (19 logits)"
                  f"               ✓")
            continue

    # Strat 3 : DTree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, yj)
    dt_err = int((dt.predict(X) != yj).sum())
    classifiers[j] = ("dtree", dt)
    status = "✓" if dt_err == 0 else f"✗ {dt_err}"
    print(f"  {LABELS[j]:25s}  DTree (p={dt.get_depth()}, "
          f"f={dt.get_n_leaves()})  {status}")


# ══════════════════════════════════════════════════════
#  PHASE 4 · Validation complète
# ══════════════════════════════════════════════════════
print(f"\n{'═' * 65}")
print(" PHASE 4 · Validation")
print("═" * 65)

def classify_batch(logits):
    probs = sigmoid(logits)
    out = np.empty((logits.shape[0], K), dtype=int)
    for j, (kind, obj) in enumerate(classifiers):
        if kind == "threshold":
            out[:, j] = (probs[:, j] >= obj).astype(int)
        else:
            out[:, j] = obj.predict(logits)
    return out

Y_pred = classify_batch(X)
n_err = int((Y_pred != Y).sum())
n_ok  = int((Y_pred == Y).all(axis=1).sum())

print(f"  ules : {N*K - n_err} / {N*K}")
print(f"  Lignes   : {n_ok} / {N}")

if n_err == 0:
    print("\n  ✅ MATCH PARFAIT !\n")
else:
    print(f"\n  Erreurs ({n_err}) :")
    for i in range(N):
        for j in range(K):
            if Y_pred[i, j] != Y[i, j]:
                print(f"    row {i:3d}  {LABELS[j]:22s}  "
                      f"pred={Y_pred[i,j]}  vrai={Y[i,j]}  "
                      f"σ={P[i,j]:.6f}")

# ══════════════════════════════════════════════════════
#  COMPARAISON : contexte vs sans contexte
# ══════════════════════════════════════════════════════
print(f"\n{'═' * 65}")
print(" Résumé comparatif")
print("═" * 65)

# recalculer baseline sans contexte
lg_base = compute_logits_ctx(all_templates["no_ctx_baseline"])
pr_base = sigmoid(lg_base)
_, err_base = find_best_thresholds(pr_base, Y)

print(f"  Sans contexte (EOS)       : {err_base:4d} erreurs @opt")
print(f"  Avec contexte ({best['name']:15s}) : {best['err']:4d} erreurs @opt")
diff = err_base - best["err"]
if diff > 0:
    print(f"  → Le contexte voisin réduit de {diff} erreurs !")
elif diff == 0:
    print(f"  → Pas de différence.")
else:
    print(f"  → Le contexte ajoute {-diff} erreurs (pire).")


# ══════════════════════════════════════════════════════
#  FONCTION RÉUTILISABLE
# ══════════════════════════════════════════════════════
BEST_TEMPLATE_FN = best["fn"]

@torch.no_grad()
def predict_web(sentence_index: int) -> dict:
    """
    Prédit les labels pour la phrase à l'index donné
    en utilisant les phrases voisines comme contexte.
    """
    text = BEST_TEMPLATE_FN(sentence_index)
    enc = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=512).to(DEVICE)
    logits = model(**enc).logits.squeeze(0).cpu().numpy()
    probs = sigmoid(logits)
    result = {}
    for j, (kind, obj) in enumerate(classifiers):
        if kind == "threshold":
            result[LABELS[j]] = int(probs[j] >= obj)
        else:
            result[LABELS[j]] = int(obj.predict(logits.reshape(1,-1))[0])
    return result

@torch.no_grad()
def predict_web_triplet(prev: str, current: str, nxt: str) -> dict:
    """
    Prédit les labels pour une phrase avec contexte explicite.
    prev/nxt peuvent être None ou "" pour les bords.
    """
    p = prev if prev else (EOS if "eos" in best["name"] else "")
    n = nxt  if nxt  else (EOS if "eos" in best["name"] else "")
    # reproduire le format du meilleur template
    text = f"before: {p}{EOS}current: {current}{EOS}after: {n}{EOS}"
    enc = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=512).to(DEVICE)
    logits = model(**enc).logits.squeeze(0).cpu().numpy()
    probs = sigmoid(logits)
    result = {}
    for j, (kind, obj) in enumerate(classifiers):
        if kind == "threshold":
            result[LABELS[j]] = int(probs[j] >= obj)
        else:
            result[LABELS[j]] = int(obj.predict(logits.reshape(1,-1))[0])
    return result

# Vérification
print(f"\n{'─' * 65}")
print(" Vérification predict_web()")
print("─" * 65)
all_ok = all(
    predict_web(i) == {LABELS[j]: int(Y[i,j]) for j in range(K)}
    for i in range(N)
)
print(f"  {'✅ 101/101 MATCH' if all_ok else '❌ Écart détecté'}")
```
```output
Données : 101 phrases × 19 labels = 1919 ules

═════════════════════════════════════════════════════════════════
 PHASE 1 · Templates avec phrases voisines comme contexte
═════════════════════════════════════════════════════════════════
  ctx_a_eos               @0.5= 176  @opt= 125
  ctx_b_eos               @0.5= 176  @opt= 125
  ctx_c_eos               @0.5= 176  @opt= 125
  ctx_d_eos               @0.5= 164  @opt= 120
  ctx_e_eos               @0.5= 176  @opt= 125
  ctx_a_empty             @0.5= 176  @opt= 125
  ctx_b_empty             @0.5= 176  @opt= 125
  ctx_c_empty             @0.5= 176  @opt= 125
  ctx_d_empty             @0.5= 164  @opt= 119
  ctx_e_empty             @0.5= 176  @opt= 125
  ctx_a_none              @0.5= 176  @opt= 125
  ctx_b_none              @0.5= 176  @opt= 125
  ctx_c_none              @0.5= 176  @opt= 125
  ctx_d_none              @0.5= 164  @opt= 120
  ctx_e_none              @0.5= 176  @opt= 125
  no_ctx_baseline         @0.5= 140  @opt=  93

  ► Meilleur : 'no_ctx_baseline' (93 erreurs)
    (baseline sans contexte : rappel de la phase précédente = 93)

═════════════════════════════════════════════════════════════════
 PHASE 2 · Seuils optimaux [no_ctx_baseline]
═════════════════════════════════════════════════════════════════
  emo                        seuil=0.002235  ✗ 9 err
  emo_comportementale        seuil=0.199618  ✗ 1 err
  emo_designee               seuil=0.085231  ✗ 2 err
  emo_montree                seuil=0.774066  ✗ 10 err
  emo_suggeree               seuil=0.042740  ✗ 11 err
  emo_base                   seuil=0.013664  ✗ 13 err
  emo_complexe               seuil=0.284000  ✗ 4 err
  admiration                 seuil=0.007979  ✗ 17 err
  autre                      seuil=0.999798  ✓
  colere                     seuil=0.219163  ✗ 17 err
  culpabilite                seuil=0.031289  ✓
  degout                     seuil=0.076577  ✓
  embarras                   seuil=0.620480  ✗ 2 err
  fierte                     seuil=0.016108  ✓
  jalousie                   seuil=0.002781  ✓
  joie                       seuil=0.033632  ✗ 2 err
  peur                       seuil=0.172037  ✓
  surprise                   seuil=0.506692  ✓
  tristesse                  seuil=0.017127  ✗ 5 err

═════════════════════════════════════════════════════════════════
 PHASE 3 · Classifieurs (seuil → LogReg → DTree)
═════════════════════════════════════════════════════════════════
  emo                        DTree (p=4, f=9)  ✓
  emo_comportementale        LogReg (19 logits)               ✓
  emo_designee               LogReg (19 logits)               ✓
  emo_montree                DTree (p=8, f=12)  ✓
  emo_suggeree               DTree (p=6, f=12)  ✓
  emo_base                   DTree (p=5, f=14)  ✓
  emo_complexe               LogReg (19 logits)               ✓
  admiration                 LogReg (19 logits)               ✓
  autre                      seuil = 0.999798                ✓
  colere                     DTree (p=7, f=18)  ✓
  culpabilite                seuil = 0.031289                ✓
  degout                     seuil = 0.076577                ✓
  embarras                   LogReg (19 logits)               ✓
  fierte                     seuil = 0.016108                ✓
  jalousie                   seuil = 0.002781                ✓
  joie                       LogReg (19 logits)               ✓
  peur                       seuil = 0.172037                ✓
  surprise                   seuil = 0.506692                ✓
  tristesse                  LogReg (19 logits)               ✓

═════════════════════════════════════════════════════════════════
 PHASE 4 · Validation
═════════════════════════════════════════════════════════════════
  ules : 1919 / 1919
  Lignes   : 101 / 101

  ✅ MATCH PARFAIT !


═════════════════════════════════════════════════════════════════
 Résumé comparatif
═════════════════════════════════════════════════════════════════
  Sans contexte (EOS)       :   93 erreurs @opt
  Avec contexte (no_ctx_baseline) :   93 erreurs @opt
  → Pas de différence.

─────────────────────────────────────────────────────────────────
 Vérification predict_web()
─────────────────────────────────────────────────────────────────
  ❌ Écart détecté
