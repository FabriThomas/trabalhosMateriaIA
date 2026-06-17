"""
trainer.py — seção de treinamento de IA do Curador de Imagens.

Usa as imagens de processed/<categoria>/ como dataset rotulado (categoria = rótulo).
Cada imagem vira um vetor de features (RGB redimensionado p/ 16x16 = 768 dims).

Parte 1 (Supervisionado): Decision Tree, KNN, Naive Bayes, SVM, MLP/Perceptron,
  Random Forest, AdaBoost, PRISM (ilustrativo). Métricas: Acurácia, Precisão,
  Matriz de Confusão. Gráficos: comparação, matriz de confusão, curva de perda (MLP).
Parte 2 (Não supervisionado): K-Means, AGNES (aglomerativo/Ward), DIANA (divisivo,
  proxy bisecting). Métricas: Silhouette, Davies-Bouldin. Gráficos: dendrograma, PCA 2D.
Regras de associação: Apriori (ilustrativo sobre features discretizadas).
"""

import os
# --- IMPORTANTE: setar ANTES de numpy/scipy/sklearn p/ evitar o abort do OpenMP
# (OMP: Error #15 ... libiomp5) que derruba o servidor com ERR_CONNECTION_RESET.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import re
import base64

import numpy as np
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix,
                             silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             cohen_kappa_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

FEAT = 16

# representações de features disponíveis (para a UI)
FEATURE_TYPES = [
    {"id": "rgb", "name": "Cor RGB (rápido, fraco p/ formas)"},
    {"id": "gray", "name": "Tons de cinza"},
    {"id": "edges", "name": "Bordas (Sobel) — realça formato/pinos"},
    {"id": "gray_edges", "name": "Cinza + bordas (recomendado p/ peças)"},
    {"id": "hog", "name": "HOG (formato; usa scikit-image se houver)"},
]

# =====================================================================
# REGISTROS DE MÉTODOS  — para ADICIONAR UMA NOVA FUNÇÃO/TÉCNICA, basta
# acrescentar UMA linha aqui. Ela aparece automaticamente na interface
# (via /api/train/methods) e passa a ser treinável.
#
#   SUPERVISED["chave"] = ("Nome exibido", lambda min_c: EstimadorSklearn(...))
#   CLUSTERING["chave"] = ("Nome exibido", lambda k:     EstimadorSklearn(...))
#
# O estimador supervisionado precisa de .fit/.predict (API do scikit-learn);
# o de agrupamento precisa de .fit_predict.
# =====================================================================
SUPERVISED = {
    "dt":  ("Árvore de Decisão", lambda min_c: DecisionTreeClassifier(random_state=0)),
    "knn": ("KNN", lambda min_c: KNeighborsClassifier(n_neighbors=max(1, min(5, min_c - 1)))),
    "nb":  ("Naive Bayes", lambda min_c: GaussianNB()),
    "svm": ("SVM", lambda min_c: SVC(kernel="rbf", C=2.0)),
    "mlp": ("Rede Neural (MLP)", lambda min_c: MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0)),
    "rf":  ("Random Forest", lambda min_c: RandomForestClassifier(n_estimators=120, random_state=0)),
    "ada": ("AdaBoost", lambda min_c: AdaBoostClassifier(random_state=0)),
    # exemplo p/ adicionar: "gb": ("Gradient Boosting", lambda min_c: GradientBoostingClassifier()),
}


def _diana_estimator(k):
    try:
        from sklearn.cluster import BisectingKMeans
        return BisectingKMeans(n_clusters=k, random_state=0)
    except Exception:
        return KMeans(n_clusters=k, n_init=10, random_state=1)


CLUSTERING = {
    "kmeans": ("K-Means", lambda k: KMeans(n_clusters=k, n_init=10, random_state=0)),
    "agnes":  ("AGNES (aglomerativo)", lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
    "diana":  ("DIANA (divisivo, proxy)", lambda k: _diana_estimator(k)),
}

# PRISM é tratado à parte (não é um estimador padrão do scikit-learn)
EXTRA_SUPERVISED = [{"id": "prism", "name": "PRISM (ilustrativo)"}]


def list_methods():
    return {
        "supervised": [{"id": k, "name": v[0]} for k, v in SUPERVISED.items()] + EXTRA_SUPERVISED,
        "unsupervised": [{"id": k, "name": v[0]} for k, v in CLUSTERING.items()],
        "features": FEATURE_TYPES,
    }


# ------------------------------- dataset -------------------------------
def _img_feat(path, size=16, feature="rgb"):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    if feature == "rgb":
        return arr.reshape(-1)
    gray = arr.mean(2)
    if feature == "gray":
        return gray.reshape(-1)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    if feature == "edges":
        return mag.reshape(-1)
    if feature == "gray_edges":
        return np.concatenate([gray.reshape(-1), mag.reshape(-1)])
    if feature == "hog":
        try:
            from skimage.feature import hog
            ppc = max(2, size // 4)
            return hog(gray, orientations=8, pixels_per_cell=(ppc, ppc),
                       cells_per_block=(2, 2), feature_vector=True)
        except Exception:
            return np.concatenate([gray.reshape(-1), mag.reshape(-1)])
    return arr.reshape(-1)


def _group_id(cat, fn):
    """Identifica a imagem-base (variações de augmentation compartilham o grupo)."""
    base = os.path.splitext(fn)[0]
    base = re.sub(r"(__[mbh])(_\d+)?$", "", base)
    return f"{cat}/{base}"


def detect_devices():
    """Dispositivos disponíveis. scikit-learn roda só em CPU; CUDA/ROCm exigem torch."""
    devs = [{"id": "cpu", "label": "CPU", "available": True}]
    cuda = rocm = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        rocm = bool(getattr(torch.version, "hip", None))
    except Exception:
        pass
    devs.append({"id": "cuda", "label": "CUDA (NVIDIA)", "available": cuda})
    devs.append({"id": "rocm", "label": "ROCm (AMD)", "available": rocm})
    return devs


def resolve_device(requested):
    avail = {d["id"]: d["available"] for d in detect_devices()}
    req = (requested or "cpu").lower()
    if req in ("cuda", "rocm") and not avail.get(req):
        return "cpu", (f"{req.upper()} indisponível — usando CPU (fallback).")
    if req in ("cuda", "rocm"):
        return "cpu", (f"{req.upper()} detectado, mas os modelos clássicos do "
                       "scikit-learn rodam em CPU. Treino feito em CPU.")
    return "cpu", "CPU (scikit-learn)."


def build_dataset(proc_root, cats, max_per_cat=0, balance=False,
                  size=16, feature="rgb"):
    """Carrega features. Retorna (X, y, paths, groups).
    max_per_cat>0 limita por categoria; balance iguala as classes.
    groups agrupa variações de augmentation da mesma imagem (anti-vazamento)."""
    per_cat = {}
    for c in cats:
        d = os.path.join(proc_root, c)
        if not os.path.isdir(d):
            continue
        files = [fn for fn in sorted(os.listdir(d))
                 if os.path.isfile(os.path.join(d, fn))]
        if max_per_cat and max_per_cat > 0:
            files = files[:max_per_cat]
        per_cat[c] = files
    if balance and per_cat:
        m = min((len(v) for v in per_cat.values() if v), default=0)
        if m > 0:
            per_cat = {c: v[:m] for c, v in per_cat.items()}
    X, y, paths, groups = [], [], [], []
    for c, files in per_cat.items():
        d = os.path.join(proc_root, c)
        for fn in files:
            p = os.path.join(d, fn)
            try:
                X.append(_img_feat(p, size, feature)); y.append(c)
                paths.append(p); groups.append(_group_id(c, fn))
            except Exception:
                pass
    return ((np.array(X) if X else np.empty((0, 1))),
            np.array(y), paths, np.array(groups))


# ------------------------------- graficos -------------------------------
def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=92,
                facecolor="#161d1c")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    ax.set_facecolor("#0f1413")
    for s in ax.spines.values():
        s.set_color("#26302e")
    ax.tick_params(colors="#e8efed")
    ax.title.set_color("#e8efed")
    ax.xaxis.label.set_color("#e8efed"); ax.yaxis.label.set_color("#e8efed")


def _bar_chart(results):
    items = [(v["name"], v["accuracy"]) for v in results.values() if "accuracy" in v]
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6, 3.2)); _style(ax)
    ax.barh([i[0] for i in items], [i[1] for i in items], color="#2f8f6b")
    ax.set_xlim(0, 1); ax.set_xlabel("Acurácia"); ax.set_title("Comparação de modelos")
    for yi, (_, v) in enumerate(items):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", color="#e8efed", fontsize=9)
    return _b64(fig)


def _cm_chart(cm, classes, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4)); _style(ax)
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title("Confusão — " + title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="#111", fontsize=9)
    return _b64(fig)


def _loss_chart(loss):
    fig, ax = plt.subplots(figsize=(5, 3)); _style(ax)
    ax.plot(loss, color="#3aa0ff")
    ax.set_xlabel("Época"); ax.set_ylabel("Perda"); ax.set_title("Curva de perda — MLP")
    return _b64(fig)


def _dendro(X):
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(6, 3.2)); _style(ax)
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title("Dendrograma (Ward)"); ax.set_ylabel("Distância")
    return _b64(fig)


def _scatter(X, labels):
    comp = min(2, X.shape[1])
    P = PCA(n_components=comp, random_state=0).fit_transform(X)
    if P.shape[1] == 1:
        P = np.c_[P, np.zeros(len(P))]
    fig, ax = plt.subplots(figsize=(4.6, 3.4)); _style(ax)
    ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title("Clusters (PCA 2D)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    return _b64(fig)


# ------------------------------- supervisionado -------------------------------
def _make_model(key, min_c):
    spec = SUPERVISED.get(key)
    return spec[1](min_c) if spec else None


# ------------------------- busca automática de agrupamento -------------------------
def _subsample(X, y, groups, max_per_cat=0, balance=False):
    """Aplica máximo por classe e balanceamento DEPOIS da fusão (no rótulo já agrupado)."""
    idx = {}
    for i, c in enumerate(y):
        idx.setdefault(c, []).append(i)
    if max_per_cat and max_per_cat > 0:
        cap = max(2, int(max_per_cat))
        idx = {c: v[:cap] for c, v in idx.items()}
    if balance:
        m = min(len(v) for v in idx.values())
        if m >= 2:                       # não balanceia se sobraria <2 por classe
            idx = {c: v[:m] for c, v in idx.items()}
    keep = sorted(i for v in idx.values() for i in v)
    g = groups[keep] if groups is not None else None
    return X[keep], y[keep], g


def _cv_eval_single(X, y, groups, model_key, scale, cv_folds):
    classes = sorted(set(y.tolist()))
    counts = {c: int((y == c).sum()) for c in classes}
    min_c = min(counts.values())
    cv = max(2, min(int(cv_folds), min_c))
    use_groups = groups is not None and len(set(groups.tolist())) < len(y)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    if use_groups:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            ng = {c: len(set(groups[y == c].tolist())) for c in classes}
            cv = max(2, min(cv, min(ng.values())))
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=0)
        except Exception:
            use_groups = False
    clf = _make_model(model_key, min_c)
    if scale:
        clf = make_pipeline(StandardScaler(), clf)
    if use_groups:
        yp = cross_val_predict(clf, X, y, cv=splitter, groups=groups)
    else:
        yp = cross_val_predict(clf, X, y, cv=splitter)
    acc = float(accuracy_score(y, yp))
    kappa = float(cohen_kappa_score(y, yp))
    cm = confusion_matrix(y, yp, labels=classes)
    return acc, kappa, cm, classes


def _eval_range(X, y, groups, model_key, scale, cv_min, cv_max):
    accs, kappas, last_cm, last_classes = [], [], None, None
    for f in range(cv_min, cv_max + 1):
        a, k, cm, classes = _cv_eval_single(X, y, groups, model_key, scale, f)
        accs.append(a); kappas.append(k); last_cm, last_classes = cm, classes
    return float(np.mean(accs)), float(np.mean(kappas)), last_cm, last_classes


def _binom_p(correct, n, p0):
    """P(acertar >= correct por acaso), teste binomial unicaudal."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(int(correct), int(n), p0, alternative="greater").pvalue)
    except Exception:
        try:
            from scipy.stats import binom_test
            return float(binom_test(int(correct), int(n), p0, alternative="greater"))
        except Exception:
            return None


def auto_group(X, y, groups, model_keys=("svm",), scale=True, cv_min=4, cv_max=6,
               max_per_cat=0, balance=False):
    """Funde gulosamente os pares mais confundidos. Em cada agrupamento avalia TODOS
    os modelos pedidos na faixa de folds (média±desvio) e calcula significância:
      - baseline = 1/nº de classes (acaso);
      - p-valor (teste binomial) de o melhor modelo superar o acaso;
      - Δκ de cada junção e se o ganho está acima do ruído entre folds.
    Balanceamento e máximo de imagens são aplicados DEPOIS de cada fusão."""
    if isinstance(model_keys, str):
        model_keys = [model_keys]
    model_keys = [m for m in model_keys if m in SUPERVISED] or ["svm"]
    base = sorted(set(y.tolist()))
    members = {c: [c] for c in base}
    label_of = {c: c for c in base}

    def evaluate():
        yl = np.array([label_of[c] for c in y])
        Xs, ys, gs = _subsample(X, yl, groups, max_per_cat, balance)
        permodel = {}
        for m in model_keys:
            try:
                accs, kappas = [], []
                for f in range(cv_min, cv_max + 1):
                    a, k, _, _ = _cv_eval_single(Xs, ys, gs, m, scale, f)
                    accs.append(a); kappas.append(k)
                mean_acc = float(np.mean(accs))
                ncls0 = len(set(ys.tolist())); n0 = len(ys)
                pm = _binom_p(int(round(mean_acc * n0)), n0, 1.0 / ncls0)
                permodel[m] = {"name": SUPERVISED[m][0],
                               "acc": mean_acc, "acc_std": float(np.std(accs)),
                               "kappa": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
                               "p": pm}
            except Exception:
                continue
        if not permodel:
            raise ValueError("nenhum modelo treinou")
        bestm = max(permodel, key=lambda m: permodel[m]["kappa"])
        a, k, cm, classes = _cv_eval_single(Xs, ys, gs, bestm, scale, cv_max)
        n = len(ys); ncls = len(classes); baseline = 1.0 / ncls
        correct = int(round(permodel[bestm]["acc"] * n))
        return {"permodel": permodel, "best_model": bestm, "best_name": SUPERVISED[bestm][0],
                "acc": permodel[bestm]["acc"], "acc_std": permodel[bestm]["acc_std"],
                "kappa": permodel[bestm]["kappa"], "kappa_std": permodel[bestm]["kappa_std"],
                "baseline": baseline, "n": int(n), "p_value": _binom_p(correct, n, baseline),
                "cm": cm, "classes": classes}

    def snapshot():
        return {c: lbl for lbl, mem in members.items() if len(mem) > 1 for c in mem}

    def groups_view():
        return {lbl: sorted(mem) for lbl, mem in members.items() if len(mem) > 1}

    def make_step(ev, nmerges):
        eff = round(ev["acc"] * len(ev["classes"]), 3)   # "classes efetivas" = acur×k (linear em k)
        return {"merges": nmerges, "n_classes": len(members),
                "acc": round(ev["acc"], 4), "acc_std": round(ev["acc_std"], 4),
                "kappa": round(ev["kappa"], 4), "kappa_std": round(ev["kappa_std"], 4),
                "eff_classes": eff,
                "baseline": round(ev["baseline"], 4), "n": ev["n"],
                "p_value": ev["p_value"], "best_model": ev["best_name"],
                "permodel": {m: {"name": d["name"], "acc": round(d["acc"], 4),
                                 "acc_std": round(d["acc_std"], 4), "kappa": round(d["kappa"], 4),
                                 "kappa_std": round(d["kappa_std"], 4),
                                 "p": d.get("p")}
                             for m, d in ev["permodel"].items()},
                "grouping": snapshot(), "groups": groups_view()}

    steps = []
    ev = evaluate()
    steps.append(make_step(ev, 0))
    cm, classes = ev["cm"], ev["classes"]

    nmerges = 0
    while len(members) > 2:
        n = len(classes); rs = cm.sum(1)
        best_pair, best_rate = None, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                denom = rs[i] + rs[j]
                rate = (cm[i, j] + cm[j, i]) / denom if denom else 0
                if rate > best_rate:
                    best_rate = rate; best_pair = (classes[i], classes[j])
        if not best_pair or best_rate <= 0:
            break
        a, b = best_pair
        newmem = sorted(set(members[a] + members[b]))
        newlbl = "+".join(newmem)
        del members[a]; del members[b]; members[newlbl] = newmem
        for c in newmem:
            label_of[c] = newlbl
        nmerges += 1
        try:
            ev = evaluate()
        except Exception:
            break
        steps.append(make_step(ev, nmerges))
        cm, classes = ev["cm"], ev["classes"]

    # ganho marginal de cada junção e se está acima do ruído entre folds
    for i, s in enumerate(steps):
        if i == 0:
            s["d_kappa"] = None; s["sig"] = None; s["z"] = None
        else:
            dk = s["kappa"] - steps[i - 1]["kappa"]
            pooled = float(np.hypot(s["kappa_std"], steps[i - 1]["kappa_std"])) or 1e-9
            z = dk / pooled
            s["d_kappa"] = round(dk, 4); s["z"] = round(z, 2)
            s["sig"] = bool(dk > 0 and z >= 1.0)   # ganho real > ruído

    # recomendação: MENOS junções que chega a ~melhor kappa E supera o acaso (p<0.05)
    best_kappa = max(s["kappa"] for s in steps)
    near = [i for i, s in enumerate(steps)
            if s["kappa"] >= best_kappa - 0.02 and (s["p_value"] is None or s["p_value"] < 0.05)]
    best_i = (min(near, key=lambda i: steps[i]["merges"]) if near
              else max(range(len(steps)), key=lambda i: steps[i]["kappa"]))
    return {"models": [SUPERVISED[m][0] for m in model_keys],
            "cv_range": [cv_min, cv_max], "steps": steps,
            "best_index": best_i, "best_kappa": round(best_kappa, 4)}


def _suggest_merges(cm, classes, topn=4):
    """Pares de classes mais confundidas entre si (candidatos a fusão)."""
    cm = np.array(cm); n = len(classes)
    rowsum = cm.sum(1)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            conf = int(cm[i, j] + cm[j, i])
            denom = int(rowsum[i] + rowsum[j])
            rate = conf / denom if denom else 0.0
            if conf > 0:
                pairs.append((rate, conf, classes[i], classes[j]))
    pairs.sort(reverse=True)
    return [{"a": a, "b": b, "rate": round(r, 3), "count": c}
            for r, c, a, b in pairs[:topn]]


def run_supervised(X, y, model_keys, device="cpu", scale=False,
                   cv_folds=5, groups=None, weights=None):
    dev_used, dev_note = resolve_device(device)
    classes = sorted(set(y.tolist()))
    cw = None
    if weights:
        cw = {c: float(weights.get(c, 1) or 1) for c in classes}
    counts = {c: int((y == c).sum()) for c in classes}
    if len(classes) < 2:
        raise ValueError("Precisa de pelo menos 2 categorias com imagens processadas.")
    min_c = min(counts.values())
    if min_c < 2:
        raise ValueError("Cada categoria precisa de pelo menos 2 imagens processadas.")
    cv = max(2, min(int(cv_folds), min_c))

    # validação à prova de vazamento: se há variações de augmentation (mesmo grupo),
    # usa StratifiedGroupKFold p/ não deixar a base no treino e a variação no teste.
    use_groups = groups is not None and len(set(groups)) < len(y)
    leak_note = ""
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    if use_groups:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            ng = {c: len(set(np.array(groups)[y == c])) for c in classes}
            cv = max(2, min(cv, min(ng.values())))
            splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=0)
            leak_note = ("Validação por grupos (variações de augmentation ficam no "
                         "mesmo fold — evita vazamento e métricas infladas).")
        except Exception:
            use_groups = False

    def cvp(clf):
        if use_groups:
            return cross_val_predict(clf, X, y, cv=splitter, groups=groups)
        return cross_val_predict(clf, X, y, cv=splitter)

    def wrap(clf):
        return make_pipeline(StandardScaler(), clf) if scale else clf

    results = {}
    mlp_loss = None
    for key in model_keys:
        if key == "prism":
            continue
        clf = _make_model(key, min_c)
        if clf is None:
            continue
        if cw is not None:
            try:
                if "class_weight" in clf.get_params():
                    clf.set_params(class_weight=cw)
            except Exception:
                pass
        try:
            yp = cvp(wrap(clf))
            results[key] = {
                "name": SUPERVISED[key][0],
                "accuracy": float(accuracy_score(y, yp)),
                "precision": float(precision_score(y, yp, average="macro", zero_division=0)),
                "confusion": confusion_matrix(y, yp, labels=classes).tolist(),
            }
            if key == "mlp":
                m = _make_model("mlp", min_c)
                (make_pipeline(StandardScaler(), m) if scale else m).fit(X, y)
                mlp_loss = [float(v) for v in getattr(m, "loss_curve_", [])]
        except Exception as e:
            results[key] = {"name": SUPERVISED.get(key, (key,))[0], "error": str(e)[:140]}

    if "prism" in model_keys:
        try:
            results["prism"] = _prism(X, y, classes)
        except Exception as e:
            results["prism"] = {"name": "PRISM (ilustrativo)", "error": str(e)[:140]}

    charts = {"comparison": _bar_chart(results)}
    acc_keys = [k for k in results if "accuracy" in results[k]]
    best = max(acc_keys, key=lambda k: results[k]["accuracy"], default=None)
    if best:
        charts["confusion"] = _cm_chart(results[best]["confusion"], classes,
                                        results[best]["name"])
    if mlp_loss:
        charts["mlp_loss"] = _loss_chart(mlp_loss)
    baseline = 1.0 / len(classes)
    suggested = []
    if best and "confusion" in results[best]:
        suggested = _suggest_merges(results[best]["confusion"], classes)
    return {"classes": classes, "counts": counts, "cv": cv,
            "results": results, "best": best, "charts": charts,
            "device": dev_used, "device_note": dev_note,
            "baseline": baseline, "leak_note": leak_note,
            "n": int(len(y)), "scaled": bool(scale),
            "suggested_merges": suggested}


def _prism(X, y, classes):
    """PRISM simplificado (indução de regras por cobertura) sobre PCA+terciles."""
    from collections import Counter
    n = len(X)
    comp = min(6, X.shape[1], max(2, n - 1))
    Z = PCA(n_components=comp, random_state=0).fit_transform(X)
    edges = [np.quantile(Z[:, j], [1 / 3, 2 / 3]) for j in range(comp)]
    D = np.zeros_like(Z, dtype=int)
    for j in range(comp):
        D[:, j] = np.digitize(Z[:, j], edges[j])
    rules = []
    for cls in classes:
        target = (y == cls)
        covered = np.zeros(n, dtype=bool)
        for _ in range(4):
            if not (target & ~covered).any():
                break
            rule, mask = [], np.ones(n, dtype=bool)
            for _ in range(comp):
                best = None
                for j in range(comp):
                    if any(f == j for f, _ in rule):
                        continue
                    for val in (0, 1, 2):
                        m = mask & (D[:, j] == val)
                        tot = int(m.sum())
                        if tot == 0:
                            continue
                        p = float((target & m).sum()) / tot
                        if best is None or p > best[0]:
                            best = (p, j, val)
                if best is None:
                    break
                p, j, val = best
                rule.append((j, val)); mask = mask & (D[:, j] == val)
                if p >= 0.95 or int(mask.sum()) <= 2:
                    break
            rules.append((cls, rule, mask))
            covered |= mask & target
    default = Counter(y.tolist()).most_common(1)[0][0]

    def predict(i):
        for cls, rule, _ in rules:
            if all(D[i, j] == val for j, val in rule):
                return cls
        return default

    yp = np.array([predict(i) for i in range(n)])
    seen_txt, txt = set(), []
    for r in rules:
        if not r[1]:
            continue
        s = "SE " + " E ".join(f"dim{j}={v}" for j, v in r[1]) + f" ENTÃO {r[0]}"
        if s not in seen_txt:
            seen_txt.add(s); txt.append(s)
    return {"name": "PRISM (ilustrativo)",
            "accuracy": float(accuracy_score(y, yp)),
            "precision": float(precision_score(y, yp, average="macro", zero_division=0)),
            "confusion": confusion_matrix(y, yp, labels=classes).tolist(),
            "rules": txt}


def _k_chart(rows):
    ks = [r["k"] for r in rows if r["silhouette"] is not None]
    sil = [r["silhouette"] for r in rows if r["silhouette"] is not None]
    if not ks:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.2)); _style(ax)
    ax.plot(ks, sil, "-o", color="#2f8f6b")
    bi = int(np.argmax(sil))
    ax.plot(ks[bi], sil[bi], "o", color="#caa24a", markersize=11)
    ax.annotate(f"melhor K={ks[bi]}", (ks[bi], sil[bi]),
                color="#caa24a", fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("K (nº de clusters)"); ax.set_ylabel("Silhouette ↑")
    ax.set_title("Escolha de K (silhouette)"); ax.set_xticks(ks)
    return _b64(fig)


def best_k(X, k_min=2, k_max=8, method="kmeans"):
    """Roda o agrupamento p/ K no intervalo e escolhe o melhor por silhouette."""
    n = len(X)
    k_min = max(2, int(k_min))
    k_max = max(k_min, min(int(k_max), n - 1))
    spec = CLUSTERING.get(method) or CLUSTERING["kmeans"]
    rows, best = [], None
    for k in range(k_min, k_max + 1):
        sil = db = None
        try:
            lab = spec[1](k).fit_predict(X)
            sil = float(silhouette_score(X, lab))
            db = float(davies_bouldin_score(X, lab))
        except Exception:
            pass
        rows.append({"k": k, "silhouette": (round(sil, 4) if sil is not None else None),
                     "davies_bouldin": (round(db, 4) if db is not None else None)})
        if sil is not None and (best is None or sil > best[1]):
            best = (k, sil)
    return {"method": spec[0], "rows": rows,
            "best_k": (best[0] if best else k_min),
            "chart": _k_chart(rows)}


def learning_curve_data(X, y, model_key="svm", scale=True, cv=4):
    """Acurácia em função da quantidade de imagens (responde: mais dados ajudaria?)."""
    from sklearn.model_selection import learning_curve
    classes = sorted(set(y.tolist()))
    min_c = min(int((y == c).sum()) for c in classes)
    cv = max(2, min(int(cv), min_c))
    clf = _make_model(model_key, min_c) or _make_model("svm", min_c)
    if scale:
        clf = make_pipeline(StandardScaler(), clf)
    sizes, _, val = learning_curve(
        clf, X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.25, 1.0, 5), shuffle=True, random_state=0)
    vmean = val.mean(1); vstd = val.std(1)
    fig, ax = plt.subplots(figsize=(6, 3.4)); _style(ax)
    ax.plot(sizes, vmean, "-o", color="#2f8f6b")
    ax.fill_between(sizes, vmean - vstd, vmean + vstd, color="#2f8f6b", alpha=0.18)
    ax.axhline(1.0 / len(classes), ls="--", color="#caa24a", lw=1)
    ax.set_xlabel("Imagens de treino"); ax.set_ylabel("Acurácia (validação)")
    ax.set_title(f"Curva de aprendizado — {SUPERVISED.get(model_key, ('?',))[0]}")
    slope = float(vmean[-1] - vmean[-2]) if len(vmean) >= 2 else 0.0
    hint = ("ainda subindo — mais imagens devem ajudar" if slope > 0.02
            else "praticamente estável — mais imagens ajudam pouco")
    return {"chart": _b64(fig), "sizes": [int(s) for s in sizes],
            "scores": [round(float(v), 4) for v in vmean],
            "baseline": round(1.0 / len(classes), 4), "hint": hint}


def dataset_map(X, y):
    """Mapa 2D (PCA) de todas as imagens processadas, coloridas por categoria."""
    if len(X) < 3:
        return {"error": "Precisa de pelo menos 3 imagens."}
    pc = PCA(n_components=2).fit_transform(X)
    classes = sorted(set(y.tolist()))
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); _style(ax)
    cmap = plt.cm.get_cmap("tab20", max(3, len(classes)))
    for i, cl in enumerate(classes):
        m = y == cl
        ax.scatter(pc[m, 0], pc[m, 1], s=16, color=cmap(i), label=cl,
                   alpha=0.8, edgecolors="none")
    ax.legend(fontsize=7, loc="best", framealpha=0.15, labelcolor="#e8efed", ncol=2)
    ax.set_title("Mapa do dataset (PCA) — cores = categorias")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    return {"chart": _b64(fig), "n": int(len(X)), "classes": classes}


# ------------------------------- nao supervisionado -------------------------------
def run_unsupervised(X, k, methods, y=None):
    n = len(X)
    k = max(2, min(int(k), n - 1))

    yt = None
    if y is not None and len(y) == n:
        _, yt = np.unique(y, return_inverse=True)

    def met(lab):
        try:
            sil = float(silhouette_score(X, lab))
        except Exception:
            sil = None
        try:
            db = float(davies_bouldin_score(X, lab))
        except Exception:
            db = None
        ari = nmi = None
        if yt is not None:
            try:
                ari = float(adjusted_rand_score(yt, lab))
                nmi = float(normalized_mutual_info_score(yt, lab))
            except Exception:
                pass
        return sil, db, ari, nmi

    results, labelsets = {}, {}
    for key in methods:
        spec = CLUSTERING.get(key)
        if not spec:
            continue
        try:
            lab = spec[1](k).fit_predict(X)
        except Exception as e:
            results[key] = {"name": spec[0], "silhouette": None,
                            "davies_bouldin": None, "error": str(e)[:140]}
            continue
        s, d, ari, nmi = met(lab)
        results[key] = {"name": spec[0], "silhouette": s, "davies_bouldin": d,
                        "ari": ari, "nmi": nmi}
        labelsets[key] = lab

    charts = {"dendrogram": _dendro(X)}
    if labelsets:
        charts["scatter"] = _scatter(X, next(iter(labelsets.values())))
    return {"k": k, "n": n, "results": results, "charts": charts,
            "has_truth": yt is not None}


# ------------------------------- regras (apriori) -------------------------------
def run_apriori(X, min_support=0.3, min_conf=0.6):
    from itertools import combinations
    n = len(X)
    R = X[:, 0::3].mean(1); G = X[:, 1::3].mean(1); B = X[:, 2::3].mean(1)
    L = (R + G + B) / 3.0
    feats = {"R_alto": R > np.median(R), "G_alto": G > np.median(G),
             "B_alto": B > np.median(B), "Claro": L > np.median(L)}
    items = list(feats.keys())
    trans = [set(i for i in items if feats[i][k]) for k in range(n)]

    def sup(s):
        return sum(1 for t in trans if s <= t) / n

    freq = [({i}, sup({i})) for i in items if sup({i}) >= min_support]
    for combo in combinations(items, 2):
        s = set(combo)
        if sup(s) >= min_support:
            freq.append((s, sup(s)))

    rules = []
    for s, sp in freq:
        if len(s) < 2:
            continue
        for a in s:
            ante = s - {a}
            sa = sup(ante)
            if sa > 0:
                conf = sup(s) / sa
                if conf >= min_conf:
                    rules.append({"se": sorted(ante), "entao": a,
                                  "suporte": round(sp, 2), "confianca": round(conf, 2)})
    return {"itemsets": [{"itens": sorted(s), "suporte": round(sp, 2)} for s, sp in freq],
            "rules": rules}
