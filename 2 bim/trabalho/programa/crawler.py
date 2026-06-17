"""
crawler.py — Curador de Imagens
Busca imagens por uma QUERY livre (keyword definida pelo usuario) em 3 motores:
  - duckduckgo : via lib 'ddgs' (agrega DDG + Bing, pagina) — recomendado
  - google     : scrape best-effort (sem API key) — fragil/limitado
  - google_api : Google Custom Search JSON API (precisa key + cx) — confiavel

Recursos: filtro de relevancia baseado nas palavras da query, filtro de
resolucao >=1000x1000, download com Referer + retry, e cache por categoria.
"""

import os
import re
import json
import time
import shutil
import hashlib
import tempfile

import requests
from PIL import Image

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None

STOP = {"de", "da", "do", "the", "and", "of", "for", "com", "para", "with", "no", "na"}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
}
_EXT = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png",
        "webp": ".webp", "gif": ".gif", "bmp": ".bmp"}


# ------------------------------- relevancia -------------------------------
def query_words(query):
    words = [w for w in re.split(r"[^0-9a-zà-ú]+", (query or "").lower())
             if len(w) >= 3 and w not in STOP]
    return words


def _relevant(meta, words):
    """True se o resultado parece bater com a query. Sem texto (scrape) -> passa."""
    if not words:
        return True
    text = " ".join(str(meta.get(k, "")) for k in ("title", "source", "url")).lower().strip()
    if not text:
        return True
    return any(w in text for w in words)


def _meta_size(meta):
    try:
        return int(meta.get("width") or 0), int(meta.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0


# ------------------------------- cache -------------------------------
def _hash_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def load_seen(cache_file):
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_seen(cache_file, seen):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)


def _seed_seen(seen, *dirs):
    for d in dirs:
        if os.path.isdir(d):
            for n in os.listdir(d):
                p = os.path.join(d, n)
                if os.path.isfile(p):
                    h = _hash_file(p)
                    if h:
                        seen.add(h)


# ------------------------------- motores de busca -------------------------------
def _ddg_images(query, want, page=1):
    if DDGS is None:
        print("[erro] biblioteca 'ddgs' nao instalada (pip install ddgs)")
        return []
    attempts = [
        lambda d: d.images(query, max_results=want, page=page,
                           safesearch="off", region="us-en", backend="auto"),
        lambda d: d.images(query, max_results=want, page=page),
        lambda d: d.images(query, max_results=want),
        lambda d: d.images(query=query, max_results=want),
    ]
    for tryno in range(2):
        try:
            with DDGS() as d:
                last = None
                for call in attempts:
                    try:
                        return list(call(d))
                    except TypeError as te:
                        last = te
                        continue
                if last:
                    print(f"[aviso] assinatura ddg incompativel: {last}")
                    return []
        except Exception as e:
            print(f"[aviso] ddg '{query}' p{page} (tentativa {tryno+1}): {e}")
            time.sleep(2)
    return []


def _google_scrape(query, want):
    try:
        r = requests.get("https://www.google.com/search",
                         params={"q": query, "tbm": "isch", "hl": "en"},
                         headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"[aviso] google '{query}': HTTP {r.status_code}")
            return []
        html = r.text
    except Exception as e:
        print(f"[aviso] google '{query}': {e}")
        return []
    out, seen = [], set()
    for m in re.finditer(r'\["(https?://[^"]+?\.(?:jpg|jpeg|png|webp))",(\d+),(\d+)\]', html):
        u, h, w = m.group(1), int(m.group(2)), int(m.group(3))
        if "gstatic.com" in u or "google.com" in u or u in seen:
            continue
        seen.add(u)
        out.append({"image": u, "width": w, "height": h})
        if len(out) >= want:
            break
    return out


def _google_api(query, want, api_key, cx):
    """Google Custom Search JSON API (searchType=image). 10 por chamada, ate 100."""
    if not api_key or not cx:
        print("[erro] Google API requer 'key' e 'cx' (configure em Ajustes)")
        return []
    out, start, seen = [], 1, set()
    while len(out) < want and start <= 91:
        try:
            r = requests.get("https://www.googleapis.com/customsearch/v1",
                             params={"key": api_key, "cx": cx, "q": query,
                                     "searchType": "image", "num": 10,
                                     "start": start, "safe": "off"},
                             timeout=20)
            if r.status_code != 200:
                print(f"[aviso] google api HTTP {r.status_code}: {r.text[:160]}")
                break
            data = r.json()
        except Exception as e:
            print(f"[aviso] google api: {e}")
            break
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            link = it.get("link")
            if not link or link in seen:
                continue
            seen.add(link)
            im = it.get("image", {}) or {}
            out.append({"image": link,
                        "width": im.get("width"), "height": im.get("height"),
                        "title": it.get("title", ""),
                        "url": im.get("contextLink", ""),
                        "source": it.get("displayLink", "")})
        start += 10
    return out[:want]


def search_images(engine, query, want, page=1, api_key="", cx=""):
    e = (engine or "").lower().replace(" ", "_")
    if e == "google":
        return _google_scrape(query, want)
    if e in ("google_api", "googleapi"):
        return _google_api(query, want, api_key, cx)
    return _ddg_images(query, want, page)


# ------------------------------- download / validacao -------------------------------
def _download(url, dst, referer=None):
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    for _ in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=20, stream=True)
            if r.status_code == 200:
                with open(dst, "wb") as f:
                    for ch in r.iter_content(8192):
                        if ch:
                            f.write(ch)
                return os.path.getsize(dst) > 1024
        except Exception:
            pass
    return False


def _validate(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            fmt = (im.format or "JPEG").lower()
    except Exception:
        return None
    return _EXT.get(fmt, ".jpg"), w, h


def _passes_res(w, h, res):
    """Filtro de resolução configurável. res:
      mode 'wh'   -> min_w,min_h,max_w,max_h (0 = sem limite)
      mode 'flex' -> min_short,min_long,max_short,max_long (orientação livre: 500 pode ser X ou Y)
    """
    if not res:
        return w >= 1000 and h >= 1000
    g = lambda k, d=0: int(res.get(k, d) or 0)
    if res.get("mode") == "flex":
        short, long = min(w, h), max(w, h)
        if short < g("min_short") or long < g("min_long"):
            return False
        if g("max_short") and short > g("max_short"):
            return False
        if g("max_long") and long > g("max_long"):
            return False
        return True
    if w < g("min_w") or h < g("min_h"):
        return False
    if g("max_w") and w > g("max_w"):
        return False
    if g("max_h") and h > g("max_h"):
        return False
    return True


# ------------------------------- principal -------------------------------
def fetch_candidates(cand_dir, seed_dir, cache_file, query,
                     count=60, engine="duckduckgo", api_key="", cx="",
                     res=None, relevance=True, or_queries=None, min_size=None):
    """Baixa ate `count` imagens NOVAS. Retorna {added, report} com os motivos de descarte."""
    if res is None and min_size:
        res = {"mode": "wh", "min_w": min_size[0], "min_h": min_size[1]}
    os.makedirs(cand_dir, exist_ok=True)
    seen = load_seen(cache_file)
    _seed_seen(seen, cand_dir, seed_dir)

    queries = [query] + [q for q in (or_queries or []) if q.strip()]
    wordsets = [query_words(q) for q in queries] if relevance else []
    e = (engine or "").lower().replace(" ", "_")
    pages = 4 if e not in ("google", "google_api", "googleapi") else 1

    rep = {"found": 0, "rej_relevance": 0, "rej_meta_res": 0,
           "fail_download": 0, "rej_resolution": 0, "duplicates": 0, "added": 0}

    pool, seen_urls = [], set()
    target_pool = count * 6
    for q in queries:
        for page in range(1, pages + 1):
            if len(pool) >= target_pool:
                break
            for r in search_images(engine, q, 100, page, api_key, cx):
                img = r.get("image") or r.get("url")
                if not img or img in seen_urls:
                    continue
                seen_urls.add(img); rep["found"] += 1
                if relevance and not any(_relevant(r, ws) for ws in wordsets):
                    rep["rej_relevance"] += 1; continue
                w, h = _meta_size(r)
                if w and h and not _passes_res(w, h, res):
                    rep["rej_meta_res"] += 1; continue
                pool.append((img, r.get("url") or r.get("source") or ""))
            time.sleep(1)

    existing = [n for n in os.listdir(cand_dir)
                if os.path.isfile(os.path.join(cand_dir, n))]
    idx = len(existing)
    added = 0
    tmpdir = tempfile.mkdtemp(prefix="dl_")
    try:
        for img, page in pool:
            if added >= count:
                break
            tmp = os.path.join(tmpdir, "x.bin")
            if os.path.exists(tmp):
                os.remove(tmp)
            if not _download(img, tmp, referer=page or None):
                rep["fail_download"] += 1; continue
            v = _validate(tmp)
            if v is None:
                rep["fail_download"] += 1; continue
            ext, w, h = v
            if not _passes_res(w, h, res):
                rep["rej_resolution"] += 1; continue
            hsh = _hash_file(tmp)
            if not hsh or hsh in seen:
                rep["duplicates"] += 1; continue
            seen.add(hsh)
            idx += 1
            dst = os.path.join(cand_dir, f"{idx:05d}{ext}")
            while os.path.exists(dst):
                idx += 1
                dst = os.path.join(cand_dir, f"{idx:05d}{ext}")
            shutil.move(tmp, dst)
            added += 1
        save_seen(cache_file, seen)
        rep["added"] = added
        rep["pool"] = len(pool)
        return {"added": added, "report": rep}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
