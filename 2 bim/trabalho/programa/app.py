"""
app.py — Curador de Imagens (servidor local)

Organizacao em PROJETOS:
  projects/<projeto>/
    project.json                 # {name, template, categories:[{name,query,goal}]}
    candidates/<categoria>/      # baixadas, aguardando curadoria
    processed/<categoria>/       # recortes + variacoes (o dataset)
    cache/<categoria>.json       # hashes ja vistos
    _trash/<categoria>/          # exclusoes (Ctrl+Z desfaz)

Recursos:
  - Template de busca com $cat$ (qualquer $...$ vira o nome da categoria)
  - CRUD de projeto e categoria (renomeia ate as pastas)
  - Auto-deteccao: pastas criadas na mao viram categorias automaticamente
  - Busca (DuckDuckGo / Google / Google API), curadoria, recorte, augmentation
  - Exportacao ZIP (atual / selecionadas / todas) com limite
  - Treinamento de IA (cap/balanceamento + selecao de dispositivo com fallback)

Rodar:  pip install -r requirements.txt  &&  python app.py  ->  http://127.0.0.1:5000
"""

import os
# evita o abort do OpenMP que derruba o servidor ao usar scikit-learn
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import sys
import io
import time
import json
import shutil
import zipfile
import tempfile
import subprocess

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, send_file, abort)
from PIL import Image, ImageOps, ImageEnhance

from crawler import fetch_candidates

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.join(BASE, "projects")
SETTINGS_FILE = os.path.join(BASE, "settings.json")

DEFAULT_TEMPLATE = "$cat$"
DEFAULT_PROJECT = "exemplo-sockets"
DEFAULT_PROJECT_DATA = {
    "name": DEFAULT_PROJECT, "template": "motherboard $cat$",
    "categories": [{"name": n, "query": "", "goal": 20}
                   for n in ["AM4", "AM5", "LGA1700"]],
}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+.\- ]{0,39}$")
AUG_MARK = ("__m", "__b", "__h")
UNDO = {}  # (projeto, categoria) -> [ [arquivos], ... ]

app = Flask(__name__)


# ============================ caminhos ============================
def p_dir(p): return os.path.join(PROJ_ROOT, p)
def cand_dir(p, c): return os.path.join(PROJ_ROOT, p, "candidates", c)
def proc_dir(p, c): return os.path.join(PROJ_ROOT, p, "processed", c)
def proc_root(p): return os.path.join(PROJ_ROOT, p, "processed")
def cache_file(p, c): return os.path.join(PROJ_ROOT, p, "cache", c + ".json")
def trash_dir(p, c): return os.path.join(PROJ_ROOT, p, "_trash", c)
def project_json(p): return os.path.join(PROJ_ROOT, p, "project.json")


# ============================ projetos ============================
def list_projects():
    """Auto-detecta projetos: toda pasta dentro de projects/ é um projeto."""
    os.makedirs(PROJ_ROOT, exist_ok=True)
    names = [d for d in sorted(os.listdir(PROJ_ROOT))
             if os.path.isdir(os.path.join(PROJ_ROOT, d)) and not d.startswith(".")]
    if not names:
        save_project(DEFAULT_PROJECT, DEFAULT_PROJECT_DATA)
        names = [DEFAULT_PROJECT]
    return names


def save_project(p, data):
    for sub in ("candidates", "processed", "cache", "_trash"):
        os.makedirs(os.path.join(PROJ_ROOT, p, sub), exist_ok=True)
    # garante a pasta de cada categoria (mantém disco e project.json coerentes)
    for c in data.get("categories", []):
        nm = c.get("name", "")
        if nm:
            os.makedirs(os.path.join(PROJ_ROOT, p, "processed", nm), exist_ok=True)
            os.makedirs(os.path.join(PROJ_ROOT, p, "candidates", nm), exist_ok=True)
    with open(project_json(p), "w", encoding="utf-8") as f:
        json.dump({"name": data.get("name", p),
                   "template": data.get("template", DEFAULT_TEMPLATE),
                   "categories": data.get("categories", [])},
                  f, ensure_ascii=False, indent=2)


def load_project(p):
    """Carrega o projeto e SINCRONIZA com o disco:
    - remove categorias cujas pastas (processed e candidates) não existem mais;
    - adiciona como categoria qualquer pasta nova achada em processed/ ou candidates/."""
    data = {"name": p, "template": DEFAULT_TEMPLATE, "categories": []}
    pj = project_json(p)
    if os.path.isfile(pj):
        try:
            d = json.load(open(pj, encoding="utf-8"))
            data["name"] = d.get("name", p)
            data["template"] = d.get("template", DEFAULT_TEMPLATE)
            cats = []
            for c in d.get("categories", []):
                nm = str(c.get("name", "")).strip()
                if nm:
                    cats.append({"name": nm, "query": str(c.get("query", "")).strip(),
                                 "goal": int(c.get("goal", 20) or 20)})
            data["categories"] = cats
        except Exception:
            pass
    # poda: categoria sem nenhuma pasta no disco deixa de existir
    data["categories"] = [c for c in data["categories"]
                          if os.path.isdir(proc_dir(p, c["name"]))
                          or os.path.isdir(cand_dir(p, c["name"]))]
    # auto-detecção de pastas (criadas manualmente ou por outro fluxo)
    known = {c["name"] for c in data["categories"]}
    for sub in ("processed", "candidates"):
        d = os.path.join(PROJ_ROOT, p, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if (os.path.isdir(os.path.join(d, fn)) and fn not in known
                        and not fn.startswith(".")):
                    data["categories"].append({"name": fn, "query": "", "goal": 20})
                    known.add(fn)
    save_project(p, data)
    return data


def effective_query(proj, catname):
    c = next((x for x in proj["categories"] if x["name"] == catname), None)
    if c and c.get("query"):
        return c["query"]
    tmpl = proj.get("template") or DEFAULT_TEMPLATE
    return re.sub(r"\$[^$]+\$", catname, tmpl).strip() or catname


# ============================ settings ============================
def load_settings():
    if os.path.isfile(SETTINGS_FILE):
        try:
            d = json.load(open(SETTINGS_FILE, encoding="utf-8"))
            return {"google_api_key": d.get("google_api_key", ""),
                    "google_cx": d.get("google_cx", "")}
        except Exception:
            pass
    return {"google_api_key": "", "google_cx": ""}


def save_settings(d):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"google_api_key": d.get("google_api_key", ""),
                   "google_cx": d.get("google_cx", "")}, f, indent=2)


# ============================ helpers ============================
def safe_proj(name):
    if name not in list_projects():
        abort(400, "projeto invalido")
    return name


def safe_cat(proj_name, cat):
    proj = load_project(proj_name)
    if cat not in [c["name"] for c in proj["categories"]]:
        abort(400, "categoria invalida")
    return cat


def safe_name(n):
    if not n or "/" in n or "\\" in n or ".." in n:
        abort(400, "nome invalido")
    return n


def ok_name(n):
    return bool(n) and "/" not in n and "\\" not in n and ".." not in n


def _listdir(d):
    return sorted(n for n in os.listdir(d)
                  if os.path.isfile(os.path.join(d, n))) if os.path.isdir(d) else []


def list_state(p, name):
    proj = load_project(p)
    c = next((x for x in proj["categories"] if x["name"] == name), None)
    goal = c["goal"] if c else 20
    return {"project": p, "category": name,
            "query": effective_query(proj, name), "goal": goal,
            "candidates": _listdir(cand_dir(p, name)),
            "processed": _listdir(proc_dir(p, name)),
            "processed_count": len(_listdir(proc_dir(p, name)))}


def _is_aug(fn):
    base = os.path.splitext(fn)[0]
    return any(base.endswith(m) for m in AUG_MARK)


# ============================ augmentation ============================
def _shift_hue(im, deg):
    h, s, v = im.convert("HSV").split()
    shift = int((int(deg) % 360) / 360.0 * 255)
    h = h.point(lambda px: (px + shift) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _aug_variants(im, ops):
    out = []
    if ops.get("mirror"):
        out.append(("__m", ImageOps.mirror(im)))
    b = ops.get("brightness")
    if b:
        try:
            out.append(("__b", ImageEnhance.Brightness(im).enhance(float(b))))
        except Exception:
            pass
    hue = ops.get("hue")
    if hue:
        try:
            out.append(("__h", _shift_hue(im, int(hue))))
        except Exception:
            pass
    return out


def _save_jpg(im, pdir, base):
    dst = os.path.join(pdir, base + ".jpg")
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(pdir, f"{base}_{i}.jpg"); i += 1
    im.convert("RGB").save(dst, "JPEG", quality=92)
    return dst


# ============================ paginas ============================
@app.route("/")
def index():
    projs = list_projects()
    return render_template("index.html",
                           projects=projs,
                           current=load_project(projs[0]),
                           settings=load_settings())


# ---- projetos CRUD ----
@app.route("/api/projects")
def api_projects():
    return jsonify({"projects": list_projects()})


@app.route("/api/project/<p>")
def api_project(p):
    return jsonify(load_project(safe_proj(p)))


@app.route("/api/projects/add", methods=["POST"])
def api_proj_add():
    b = request.json or {}
    name = (b.get("name") or "").strip()
    tmpl = (b.get("template") or DEFAULT_TEMPLATE).strip()
    if not NAME_RE.match(name):
        abort(400, "nome de projeto invalido")
    if os.path.isdir(p_dir(name)):
        abort(400, "ja existe")
    save_project(name, {"name": name, "template": tmpl, "categories": []})
    return jsonify({"projects": list_projects(), "added": name})


@app.route("/api/projects/update", methods=["POST"])
def api_proj_update():
    b = request.json or {}
    name = safe_proj((b.get("name") or "").strip())
    new_name = (b.get("new_name") or "").strip()
    proj = load_project(name)
    if "template" in b:
        proj["template"] = (b.get("template") or DEFAULT_TEMPLATE).strip()
    save_project(name, proj)
    final = name
    if new_name and new_name != name:
        if not NAME_RE.match(new_name) or os.path.isdir(p_dir(new_name)):
            abort(400, "novo nome invalido ou em uso")
        shutil.move(p_dir(name), p_dir(new_name))
        proj["name"] = new_name
        save_project(new_name, proj)
        final = new_name
    return jsonify({"projects": list_projects(), "name": final})


@app.route("/api/projects/remove", methods=["POST"])
def api_proj_remove():
    name = safe_proj((request.json or {}).get("name"))
    shutil.rmtree(p_dir(name), ignore_errors=True)
    return jsonify({"projects": list_projects()})


# ---- categorias CRUD ----
@app.route("/api/categories/add", methods=["POST"])
def api_cat_add():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    name = (b.get("name") or "").strip()
    query = (b.get("query") or "").strip()
    try:
        goal = max(1, int(b.get("goal", 20)))
    except (TypeError, ValueError):
        goal = 20
    if not NAME_RE.match(name):
        abort(400, "nome invalido")
    proj = load_project(p)
    if any(c["name"] == name for c in proj["categories"]):
        abort(400, "ja existe")
    proj["categories"].append({"name": name, "query": query, "goal": goal})
    save_project(p, proj)
    os.makedirs(cand_dir(p, name), exist_ok=True)
    os.makedirs(proc_dir(p, name), exist_ok=True)
    return jsonify(load_project(p))


@app.route("/api/categories/update", methods=["POST"])
def api_cat_update():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    name = safe_cat(p, (b.get("name") or "").strip())
    new_name = (b.get("new_name") or "").strip()
    proj = load_project(p)
    for c in proj["categories"]:
        if c["name"] == name:
            if "query" in b:
                c["query"] = (b.get("query") or "").strip()
            if "goal" in b:
                try:
                    c["goal"] = max(1, int(b.get("goal")))
                except (TypeError, ValueError):
                    pass
            if new_name and new_name != name:
                if not NAME_RE.match(new_name) or any(x["name"] == new_name for x in proj["categories"]):
                    abort(400, "novo nome invalido ou em uso")
                for mover in (cand_dir, proc_dir, trash_dir):
                    src = mover(p, name)
                    if os.path.isdir(src):
                        os.makedirs(os.path.dirname(mover(p, new_name)), exist_ok=True)
                        shutil.move(src, mover(p, new_name))
                cf = cache_file(p, name)
                if os.path.isfile(cf):
                    shutil.move(cf, cache_file(p, new_name))
                c["name"] = new_name
            break
    save_project(p, proj)
    return jsonify(load_project(p))


@app.route("/api/categories/remove", methods=["POST"])
def api_cat_remove():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    name = (b.get("name") or "").strip()
    proj = load_project(p)
    proj["categories"] = [c for c in proj["categories"] if c["name"] != name]
    save_project(p, proj)
    # apaga pastas (senao a auto-deteccao readiciona)
    for d in (cand_dir(p, name), proc_dir(p, name), trash_dir(p, name)):
        shutil.rmtree(d, ignore_errors=True)
    cf = cache_file(p, name)
    if os.path.isfile(cf):
        os.remove(cf)
    return jsonify(load_project(p))


# ============================ settings ============================
@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        save_settings(request.json or {})
    s = load_settings()
    return jsonify({"google_cx": s["google_cx"], "has_key": bool(s["google_api_key"]),
                    "google_api_key": s["google_api_key"]})


# ============================ busca ============================
@app.route("/api/fetch", methods=["POST"])
def api_fetch():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    name = safe_cat(p, b.get("category"))
    engine = b.get("engine", "duckduckgo")
    try:
        count = int(b.get("count", 60))
    except (TypeError, ValueError):
        count = 60
    count = max(1, min(count, 300))
    proj = load_project(p)
    s = load_settings()
    res = b.get("res")  # dict configurável (mode wh/flex + limites); None = >=1000x1000
    or_queries = [q for q in (b.get("or_queries") or []) if str(q).strip()]
    relevance = b.get("relevance", True)
    out = fetch_candidates(
        cand_dir(p, name), proc_dir(p, name), cache_file(p, name),
        effective_query(proj, name), count=count, engine=engine,
        api_key=s["google_api_key"], cx=s["google_cx"],
        res=res, relevance=bool(relevance), or_queries=or_queries)
    st = list_state(p, name)
    st["added"] = out.get("added", 0)
    st["report"] = out.get("report")
    return jsonify(st)


@app.route("/api/state/<p>/<name>")
def api_state(p, name):
    return jsonify(list_state(safe_proj(p), safe_cat(p, name)))


# ============================ excluir / undo / limpar ============================
def _to_trash(p, name, files):
    tdir = trash_dir(p, name); os.makedirs(tdir, exist_ok=True)
    moved = []
    for fn in files:
        if ok_name(fn):
            src = os.path.join(cand_dir(p, name), fn)
            if os.path.isfile(src):
                shutil.move(src, os.path.join(tdir, fn)); moved.append(fn)
    if moved:
        UNDO.setdefault((p, name), []).append(moved)
    return moved


@app.route("/api/delete_many", methods=["POST"])
def api_delete_many():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    _to_trash(p, name, b.get("filenames", []))
    return jsonify(list_state(p, name))


@app.route("/api/undo_delete", methods=["POST"])
def api_undo_delete():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    stack = UNDO.get((p, name), [])
    undone = 0
    if stack:
        batch = stack.pop()
        cdir = cand_dir(p, name); os.makedirs(cdir, exist_ok=True)
        for fn in batch:
            src = os.path.join(trash_dir(p, name), fn)
            if os.path.isfile(src):
                dst = os.path.join(cdir, fn); base, ext = os.path.splitext(fn); i = 1
                while os.path.exists(dst):
                    dst = os.path.join(cdir, f"{base}_r{i}{ext}"); i += 1
                shutil.move(src, dst); undone += 1
    st = list_state(p, name); st["undone"] = undone
    return jsonify(st)


@app.route("/api/clear_candidates", methods=["POST"])
def api_clear_candidates():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    _to_trash(p, name, _listdir(cand_dir(p, name)))
    return jsonify(list_state(p, name))


# ============================ processar + augmentation ============================
@app.route("/api/process", methods=["POST"])
def api_process():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    fn = safe_name(b.get("filename")); box = b.get("box"); aug = b.get("aug") or {}
    src = os.path.join(cand_dir(p, name), fn)
    if not os.path.isfile(src):
        abort(404, "candidato nao encontrado")
    pdir = proc_dir(p, name); os.makedirs(pdir, exist_ok=True)
    base = os.path.splitext(fn)[0]
    # guarda a ORIGINAL (inteira) indexada pelo nome-base, p/ recortes futuros
    odir = orig_dir(p, name); os.makedirs(odir, exist_ok=True)
    try:
        if not _find_original(p, name, base):
            shutil.copy2(src, os.path.join(odir, fn))
    except Exception:
        pass
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im); W, H = im.size
            if isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h")):
                x = max(0.0, min(1.0, float(box["x"]))); y = max(0.0, min(1.0, float(box["y"])))
                w = max(0.0, min(1.0, float(box["w"]))); h = max(0.0, min(1.0, float(box["h"])))
                l, t = int(x * W), int(y * H); r, bo = int((x + w) * W), int((y + h) * H)
                if r - l >= 10 and bo - t >= 10:
                    im = im.crop((l, t, r, bo))
            _save_jpg(im, pdir, base)
            for suf, var in _aug_variants(im, aug):
                _save_jpg(var, pdir, base + suf)
    except Exception as e:
        abort(500, f"falha ao processar: {e}")
    os.remove(src)
    return jsonify(list_state(p, name))


@app.route("/api/augment_all", methods=["POST"])
def api_augment_all():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    aug = b.get("aug") or {}
    pdir = proc_dir(p, name); created = 0
    for fn in _listdir(pdir):
        if _is_aug(fn):
            continue
        base = os.path.splitext(fn)[0]
        try:
            with Image.open(os.path.join(pdir, fn)) as im:
                im = im.convert("RGB")
                for suf, var in _aug_variants(im, aug):
                    out = os.path.join(pdir, base + suf + ".jpg")
                    if not os.path.exists(out):
                        var.convert("RGB").save(out, "JPEG", quality=92); created += 1
        except Exception:
            continue
    st = list_state(p, name); st["created"] = created
    return jsonify(st)


@app.route("/api/unprocess", methods=["POST"])
def api_unprocess():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    fn = safe_name(b.get("filename"))
    src = os.path.join(proc_dir(p, name), fn)
    if not os.path.isfile(src):
        abort(404)
    cdir = cand_dir(p, name); os.makedirs(cdir, exist_ok=True)
    shutil.move(src, os.path.join(cdir, fn))
    return jsonify(list_state(p, name))


# ============================ girar imagem ============================
@app.route("/api/rotate", methods=["POST"])
def api_rotate():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    fn = safe_name(b.get("filename"))
    where = b.get("where", "candidate")
    base = proc_dir if where == "processed" else cand_dir
    src = os.path.join(base(p, name), fn)
    if not os.path.isfile(src):
        abort(404)
    deg = -90 if b.get("dir") == "right" else 90  # PIL gira anti-horário
    try:
        with Image.open(src) as imf:
            im = ImageOps.exif_transpose(imf); im.load()
        im = im.rotate(deg, expand=True)
        ext = os.path.splitext(fn)[1].lower()
        if ext == ".png":
            im.save(src, "PNG")
        elif ext == ".webp":
            im.convert("RGB").save(src, "WEBP", quality=92)
        else:
            im.convert("RGB").save(src, "JPEG", quality=92)
    except Exception as e:
        abort(500, f"falha ao girar: {e}")
    return jsonify({"ok": True})


# ============================ modo revisão (re-recortar processadas) ============================
def orig_dir(p, c):
    return os.path.join(PROJ_ROOT, p, "originals", c)


def _find_original(p, c, base):
    d = orig_dir(p, c)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if os.path.splitext(fn)[0] == base and os.path.isfile(os.path.join(d, fn)):
            return os.path.join(d, fn)
    return None


def _review_source_path(p, name, fn, source):
    """Caminho da imagem-fonte: 'original' (inteira guardada) ou 'processed' (recorte atual)."""
    if source == "original":
        op = _find_original(p, name, os.path.splitext(fn)[0])
        if op:
            return op
    return os.path.join(proc_dir(p, name), fn)


def _load_rotated(path, angle):
    with Image.open(path) as imf:
        im = ImageOps.exif_transpose(imf).convert("RGB"); im.load()
    a = float(angle or 0) % 360
    if a:
        # -a p/ girar no sentido horário (igual ao transform CSS da interface)
        im = im.rotate(-a, expand=True, resample=Image.BICUBIC, fillcolor=(20, 25, 24))
    return im


@app.route("/api/review/preview")
def api_review_preview():
    p = safe_proj(request.args.get("project"))
    name = safe_cat(p, request.args.get("category"))
    fn = safe_name(request.args.get("filename"))
    source = request.args.get("source", "processed")
    try:
        angle = float(request.args.get("angle", "0"))
    except ValueError:
        angle = 0.0
    src = _review_source_path(p, name, fn, source)
    if not os.path.isfile(src):
        abort(404)
    im = _load_rotated(src, angle)
    flip = request.args.get("flip", "")
    if flip == "h":
        im = ImageOps.mirror(im)
    elif flip == "v":
        im = ImageOps.flip(im)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/review/has_original")
def api_review_has_original():
    p = safe_proj(request.args.get("project"))
    name = safe_cat(p, request.args.get("category"))
    fn = safe_name(request.args.get("filename"))
    return jsonify({"has": bool(_find_original(p, name, os.path.splitext(fn)[0]))})


def _archive_name(p, cat, fn):
    """Primeiro <cat>_old que ainda não contém esse arquivo (_old, _old2, ...)."""
    n = 1
    while True:
        arch = cat + "_old" + ("" if n == 1 else str(n))
        if not os.path.isfile(os.path.join(proc_dir(p, arch), fn)):
            return arch
        n += 1


@app.route("/api/review/recrop", methods=["POST"])
def api_review_recrop():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    fn = safe_name(b.get("filename")); box = b.get("box")
    src = os.path.join(proc_dir(p, name), fn)
    if not os.path.isfile(src):
        abort(404, "imagem não encontrada")
    if not (isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h"))):
        st = list_state(p, name); st["skipped"] = True
        return jsonify(st)
    # arquiva a versão anterior em <cat>_old[N]
    arch = _archive_name(p, name, fn)
    os.makedirs(proc_dir(p, arch), exist_ok=True)
    os.makedirs(cand_dir(p, arch), exist_ok=True)
    shutil.copy2(src, os.path.join(proc_dir(p, arch), fn))
    # recorta o original e sobrescreve a versão atual
    try:
        with Image.open(src) as imf:
            im = ImageOps.exif_transpose(imf); im.load()
        W, H = im.size
        x = max(0.0, min(1.0, float(box["x"]))); y = max(0.0, min(1.0, float(box["y"])))
        w = max(0.0, min(1.0, float(box["w"]))); h = max(0.0, min(1.0, float(box["h"])))
        l, t = int(x * W), int(y * H); r, bo = int((x + w) * W), int((y + h) * H)
        if r - l >= 10 and bo - t >= 10:
            im = im.crop((l, t, r, bo))
        im.convert("RGB").save(src, "JPEG", quality=92)
    except Exception as e:
        abort(500, f"falha ao recortar: {e}")
    st = list_state(p, name); st["archived"] = arch
    return jsonify(st)


@app.route("/api/review/transform", methods=["POST"])
def api_review_transform():
    """Rotação livre (graus) + recorte, podendo usar a imagem ORIGINAL (com margem).
    Arquiva a versão atual em <cat>_old[N] e grava o novo recorte como a processada."""
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    fn = safe_name(b.get("filename")); box = b.get("box")
    source = b.get("source", "processed")
    try:
        angle = float(b.get("angle", 0))
    except (TypeError, ValueError):
        angle = 0.0
    cur = os.path.join(proc_dir(p, name), fn)
    if not os.path.isfile(cur):
        abort(404, "imagem não encontrada")
    srcpath = _review_source_path(p, name, fn, source)
    if not os.path.isfile(srcpath):
        abort(404, "fonte não encontrada")
    # arquiva a processada atual
    arch = _archive_name(p, name, fn)
    os.makedirs(proc_dir(p, arch), exist_ok=True); os.makedirs(cand_dir(p, arch), exist_ok=True)
    shutil.copy2(cur, os.path.join(proc_dir(p, arch), fn))
    try:
        im = _load_rotated(srcpath, angle)   # já roda com expand (não perde conteúdo)
        flip = b.get("flip")
        if flip == "h":
            im = ImageOps.mirror(im)
        elif flip == "v":
            im = ImageOps.flip(im)
        W, H = im.size
        if isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h")):
            x = max(0.0, min(1.0, float(box["x"]))); y = max(0.0, min(1.0, float(box["y"])))
            w = max(0.0, min(1.0, float(box["w"]))); h = max(0.0, min(1.0, float(box["h"])))
            l, t = int(x * W), int(y * H); r, bo = int((x + w) * W), int((y + h) * H)
            if r - l >= 10 and bo - t >= 10:
                im = im.crop((l, t, r, bo))
        im.convert("RGB").save(cur, "JPEG", quality=92)
    except Exception as e:
        abort(500, f"falha ao transformar: {e}")
    st = list_state(p, name); st["archived"] = arch
    return jsonify(st)


# ============================ surpresa 1: detector de duplicatas ============================
def _dhash(path, size=8):
    import numpy as np
    with Image.open(path) as imf:
        im = ImageOps.exif_transpose(imf).convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


@app.route("/api/duplicates", methods=["POST"])
def api_duplicates():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    where = b.get("where", "candidate")
    d = (proc_dir if where == "processed" else cand_dir)(p, name)
    try:
        thr = int(b.get("threshold", 8))
    except (TypeError, ValueError):
        thr = 8
    hashes = {}
    for fn in _listdir(d):
        try:
            hashes[fn] = _dhash(os.path.join(d, fn))
        except Exception:
            pass
    names = list(hashes.keys()); used = set(); groups = []
    for i, fa in enumerate(names):
        if fa in used:
            continue
        grp = [fa]
        for fb in names[i + 1:]:
            if fb in used:
                continue
            if bin(hashes[fa] ^ hashes[fb]).count("1") <= thr:
                grp.append(fb); used.add(fb)
        if len(grp) > 1:
            used.add(fa); groups.append(grp)
    redundant = [f for g in groups for f in g[1:]]
    return jsonify({"groups": groups, "redundant": redundant, "count": len(redundant)})


# ============================ export ============================
@app.route("/api/export")
def api_export():
    p = safe_proj(request.args.get("project"))
    scope = request.args.get("scope", "all")
    cat = request.args.get("category", "")
    try:
        mx = int(request.args.get("max", "0"))
    except ValueError:
        mx = 0
    allcats = [c["name"] for c in load_project(p)["categories"]]
    if scope == "category" and cat in allcats:
        cats = [cat]
    elif scope == "selected":
        wanted = [c.strip() for c in request.args.get("cats", "").split(",") if c.strip()]
        cats = [c for c in wanted if c in allcats]
    else:
        cats = allcats
    tmp = tempfile.mkstemp(suffix=".zip")[1]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for c in cats:
            files = _listdir(proc_dir(p, c))
            if mx > 0:
                files = files[:mx]
            for fn in files:
                z.write(os.path.join(proc_dir(p, c), fn), arcname=f"{c}/{fn}")
    return send_file(tmp, as_attachment=True,
                     download_name=f"dataset_{p}.zip", mimetype="application/zip")


# ============================ abrir pasta ============================
@app.route("/api/open", methods=["POST"])
def api_open():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_cat(p, b.get("category"))
    path = proc_dir(p, name); os.makedirs(path, exist_ok=True)
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ============================ treinamento ============================
@app.route("/api/train/devices")
def api_train_devices():
    try:
        import trainer
        return jsonify({"devices": trainer.detect_devices()})
    except Exception:
        return jsonify({"devices": [{"id": "cpu", "label": "CPU", "available": True}]})


@app.route("/api/train/methods")
def api_train_methods():
    try:
        import trainer
        return jsonify(trainer.list_methods())
    except Exception:
        return jsonify({"supervised": [], "unsupervised": []})


@app.route("/api/train/info/<p>")
def api_train_info(p):
    p = safe_proj(p)
    info = [{"name": c["name"], "count": len(_listdir(proc_dir(p, c["name"])))}
            for c in load_project(p)["categories"]]
    return jsonify({"categories": info})


def _train_args(b):
    p = safe_proj(b.get("project"))
    cats = [c["name"] for c in load_project(p)["categories"]]
    chosen = [c for c in (b.get("cats") or []) if c in cats]
    try:
        mx = int(b.get("max_per_cat", 0))
    except (TypeError, ValueError):
        mx = 0
    try:
        size = int(b.get("img_size", 16))
    except (TypeError, ValueError):
        size = 16
    size = max(8, min(64, size))
    feature = b.get("feature", "rgb")
    return p, chosen, mx, bool(b.get("balance")), size, feature


def _apply_merges(y, merges):
    """Remapeia rótulos: categorias num mesmo grupo viram uma classe só."""
    if not merges:
        return y
    import numpy as np
    return np.array([merges.get(v, v) for v in y])


@app.route("/api/train/supervised", methods=["POST"])
def api_train_supervised():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        X, y, _, groups = trainer.build_dataset(proc_root(p), cats, mx, bal, size, feat)
        y = _apply_merges(y, b.get("merges") or {})
        if len(X) == 0:
            return jsonify({"error": "Sem imagens processadas nas categorias escolhidas."})
        try:
            cv = int(b.get("cv_folds", 5))
        except (TypeError, ValueError):
            cv = 5
        return jsonify(trainer.run_supervised(
            X, y, b.get("models") or [], device=b.get("device", "cpu"),
            scale=bool(b.get("scale")), cv_folds=cv, groups=groups,
            weights=b.get("weights") or None))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


@app.route("/api/train/learncurve", methods=["POST"])
def api_train_learncurve():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        X, y, _, _ = trainer.build_dataset(proc_root(p), cats, mx, bal, size, feat)
        y = _apply_merges(y, b.get("merges") or {})
        if len(X) < 6:
            return jsonify({"error": "Precisa de mais imagens processadas."})
        return jsonify(trainer.learning_curve_data(
            X, y, model_key=b.get("model", "svm"), scale=bool(b.get("scale", True)),
            cv=int(b.get("cv_folds", 4))))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


@app.route("/api/train/unsupervised", methods=["POST"])
def api_train_unsupervised():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        k = int(b.get("k", 3))
        X, y, _, _ = trainer.build_dataset(proc_root(p), cats, mx, bal, size, feat)
        y = _apply_merges(y, b.get("merges") or {})
        if len(X) < 2:
            return jsonify({"error": "Precisa de pelo menos 2 imagens processadas."})
        return jsonify(trainer.run_unsupervised(X, k, b.get("methods") or [], y=y))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


@app.route("/api/train/autogroup", methods=["POST"])
def api_train_autogroup():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        # dataset completo (sem cap/balance aqui — eles são aplicados PÓS-fusão)
        X, y, _, groups = trainer.build_dataset(proc_root(p), cats, 0, False, size, feat)
        if len(X) == 0:
            return jsonify({"error": "Sem imagens processadas nas categorias escolhidas."})
        if len(set(y.tolist())) < 3:
            return jsonify({"error": "Precisa de pelo menos 3 categorias para buscar agrupamentos."})
        model = b.get("model", "svm")
        models = b.get("models") or ([model] if model else ["svm"])
        try:
            cvmin = max(2, int(b.get("cv_min", 4)))
            cvmax = max(cvmin, int(b.get("cv_max", 6)))
        except (TypeError, ValueError):
            cvmin, cvmax = 4, 6
        return jsonify(trainer.auto_group(
            X, y, groups, model_keys=models, scale=bool(b.get("scale", True)),
            cv_min=cvmin, cv_max=cvmax, max_per_cat=mx, balance=bal))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


@app.route("/api/train/map", methods=["POST"])
def api_train_map():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        X, y, _, _ = trainer.build_dataset(proc_root(p), cats, mx, bal, size, feat)
        if len(X) < 3:
            return jsonify({"error": "Precisa de pelo menos 3 imagens processadas."})
        return jsonify(trainer.dataset_map(X, y))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


@app.route("/api/train/bestk", methods=["POST"])
def api_train_bestk():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        X, _, _, _ = trainer.build_dataset(proc_root(p), cats, mx, bal, size, feat)
        if len(X) < 3:
            return jsonify({"error": "Precisa de pelo menos 3 imagens processadas."})
        kmin = int(b.get("k_min", 2)); kmax = int(b.get("k_max", 8))
        return jsonify(trainer.best_k(X, kmin, kmax, b.get("method", "kmeans")))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


# ============================ resultados salvos (JSON) ============================
def results_dir(p):
    return os.path.join(PROJ_ROOT, p, "results")


def _list_results(p):
    d = results_dir(p)
    if not os.path.isdir(d):
        return []
    return sorted((os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".json")),
                  reverse=True)


@app.route("/api/results/save", methods=["POST"])
def api_results_save():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    raw = (b.get("name") or "").strip()
    name = re.sub(r"[^A-Za-z0-9_\- ]", "", raw)[:60] or time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(results_dir(p), exist_ok=True)
    with open(os.path.join(results_dir(p), name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "saved": time.time(), "data": b.get("data")},
                  f, ensure_ascii=False)
    return jsonify({"ok": True, "name": name, "list": _list_results(p)})


@app.route("/api/results/list/<p>")
def api_results_list(p):
    return jsonify({"list": _list_results(safe_proj(p))})


@app.route("/api/results/load", methods=["POST"])
def api_results_load():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_name(b.get("name"))
    fn = os.path.join(results_dir(p), name + ".json")
    if not os.path.isfile(fn):
        abort(404)
    return jsonify(json.load(open(fn, encoding="utf-8")))


@app.route("/api/results/delete", methods=["POST"])
def api_results_delete():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_name(b.get("name"))
    fn = os.path.join(results_dir(p), name + ".json")
    if os.path.isfile(fn):
        os.remove(fn)
    return jsonify({"list": _list_results(p)})


@app.route("/api/results/timeline/<p>")
def api_results_timeline(p):
    """Linha do tempo: resultados salvos com data e a melhor métrica de cada um."""
    p = safe_proj(p)
    out = []
    d = results_dir(p)
    for name in _list_results(p):
        try:
            j = json.load(open(os.path.join(d, name + ".json"), encoding="utf-8"))
            data = j.get("data") or {}
            metric, kind = None, ""
            if data.get("auto"):
                metric = data["auto"].get("best_kappa"); kind = "kappa(auto)"
            elif data.get("supervised", {}).get("results"):
                accs = [v.get("accuracy") for v in data["supervised"]["results"].values()
                        if isinstance(v, dict) and "accuracy" in v]
                if accs:
                    metric = round(max(accs), 4); kind = "acur(sup)"
            out.append({"name": name, "saved": j.get("saved"), "metric": metric, "kind": kind})
        except Exception:
            out.append({"name": name, "saved": None, "metric": None, "kind": ""})
    out.sort(key=lambda x: x["saved"] or 0)
    return jsonify({"timeline": out})


# presets de configuração (classes selecionadas, feature, modelos, etc.)
def presets_dir(p):
    return os.path.join(PROJ_ROOT, p, "presets")


def _list_presets(p):
    d = presets_dir(p)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d)) if os.path.isdir(d) else []


@app.route("/api/presets/list/<p>")
def api_presets_list(p):
    return jsonify({"list": _list_presets(safe_proj(p))})


@app.route("/api/presets/save", methods=["POST"])
def api_presets_save():
    b = request.json or {}
    p = safe_proj(b.get("project"))
    name = re.sub(r"[^A-Za-z0-9_\- ]", "", (b.get("name") or "").strip())[:50]
    if not name:
        abort(400, "nome inválido")
    os.makedirs(presets_dir(p), exist_ok=True)
    with open(os.path.join(presets_dir(p), name + ".json"), "w", encoding="utf-8") as f:
        json.dump(b.get("config") or {}, f, ensure_ascii=False)
    return jsonify({"ok": True, "name": name, "list": _list_presets(p)})


@app.route("/api/presets/load", methods=["POST"])
def api_presets_load():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_name(b.get("name"))
    fn = os.path.join(presets_dir(p), name + ".json")
    if not os.path.isfile(fn):
        abort(404)
    return jsonify({"config": json.load(open(fn, encoding="utf-8"))})


@app.route("/api/presets/delete", methods=["POST"])
def api_presets_delete():
    b = request.json or {}
    p = safe_proj(b.get("project")); name = safe_name(b.get("name"))
    fn = os.path.join(presets_dir(p), name + ".json")
    if os.path.isfile(fn):
        os.remove(fn)
    return jsonify({"list": _list_presets(p)})


@app.route("/api/train/rules", methods=["POST"])
def api_train_rules():
    b = request.json or {}
    try:
        import trainer
        p, cats, mx, bal, size, feat = _train_args(b)
        msup = float(b.get("min_support", 0.3)); mconf = float(b.get("min_conf", 0.6))
        X, _, _, _ = trainer.build_dataset(proc_root(p), cats, mx, bal, size, "rgb")
        if len(X) < 2:
            return jsonify({"error": "Precisa de pelo menos 2 imagens processadas."})
        return jsonify(trainer.run_apriori(X, msup, mconf))
    except ImportError:
        return jsonify({"error": "Instale: pip install scikit-learn numpy scipy matplotlib"})
    except Exception as e:
        return jsonify({"error": str(e)[:300]})


# ============================ servir imagens ============================
@app.route("/img/candidate/<p>/<name>/<fn>")
def img_candidate(p, name, fn):
    return send_from_directory(cand_dir(safe_proj(p), safe_cat(p, name)), safe_name(fn))


@app.route("/img/processed/<p>/<name>/<fn>")
def img_processed(p, name, fn):
    return send_from_directory(proc_dir(safe_proj(p), safe_cat(p, name)), safe_name(fn))


if __name__ == "__main__":
    list_projects()  # garante projeto default
    # threaded + sem reloader: requisicoes longas (treino) nao derrubam o server
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
