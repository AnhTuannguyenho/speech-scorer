#!/usr/bin/env python3
# Speech Scorer — RunPod Serverless (phần mềm chấm phát âm, bản viết mới, gọn).
# Engine: faster-whisper (ASR) + wav2vec2 phoneme GOP forced-alignment.
# Handler TRỰC TIẾP (không Flask) — input JSON, audio base64.
#
# Input job:
#   {"input":{"route":"score|grade|grade_ph|transcribe|pron|health",
#             "text":"apple","audio_b64":"<base64 webm/mp3/wav/m4a...>",
#             "words":"", "lang":"en", "prompt":"", "fast":true}}
# Output: dict (score/status/band/heard/phones/fluency... tùy route).
import base64
import math
import os
import re
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.separator import Separator
import runpod

# ===== Cấu hình =====
MODEL_NAME = os.environ.get("ASR_MODEL", "medium.en")
W2V = os.environ.get("ASR_W2V", "facebook/wav2vec2-lv-60-espeak-cv-ft")
DEVICE = os.environ.get("ASR_DEVICE", "cuda")
COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if DEVICE == "cuda" else "int8")
MAX_SEC = int(os.environ.get("ASR_MAX_SEC", "90"))
MAX_BYTES = 30 * 1024 * 1024
PH_MIN = float(os.environ.get("ASR_PH_MIN", "0.45"))
GOP_P0 = float(os.environ.get("ASR_GOP_P0", "0.18"))   # tâm sigmoid hiệu chỉnh GOP
GOP_K = float(os.environ.get("ASR_GOP_K", "14"))       # độ dốc
PH_OK = float(os.environ.get("ASR_PH_OK", "0.60"))
PH_WARN = float(os.environ.get("ASR_PH_WARN", "0.30"))

# ===== Nạp model (1 lần lúc worker khởi động) =====
print(f"[engine] loading whisper {MODEL_NAME} on {DEVICE}/{COMPUTE}...", flush=True)
if DEVICE == "cuda":
    _model = WhisperModel(MODEL_NAME, device="cuda", compute_type=COMPUTE)
else:
    _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE, cpu_threads=4, num_workers=1)
print(f"[engine] loading wav2vec2 {W2V}...", flush=True)
torch.set_num_threads(2)
_proc = Wav2Vec2Processor.from_pretrained(W2V)
_w2v = Wav2Vec2ForCTC.from_pretrained(W2V).eval().to(DEVICE)
_VOCAB = _proc.tokenizer.get_vocab()
_BLANK = _proc.tokenizer.pad_token_id
print("[engine] models ready", flush=True)


def _calib(p):
    return 1.0 / (1.0 + math.exp(-GOP_K * (p - GOP_P0)))


# ===== Audio =====
def to_wav(src):
    dst = src + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", src, "-t", str(MAX_SEC), "-ar", "16000", "-ac", "1", "-f", "wav", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    return dst


# ===== Âm vị (espeak target + wav2vec2 nhận dạng) =====
def _strip_p(p):
    for c in ('ˈ', 'ˌ', 'ː', 'ˑ', 'ʰ', '̩', '̃', 'ʲ'):
        p = p.replace(c, '')
    return p.strip()


def target_phones(text):
    s = phonemize(text, language='en-us', backend='espeak',
                  separator=Separator(phone=' ', word=' | '), strip=True, with_stress=False, njobs=1)
    return [_strip_p(x) for x in s.split() if x != '|' and _strip_p(x)]


def _logits(wav):
    audio, sr = sf.read(wav)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(1)
    iv = _proc(audio, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        return _w2v(iv).logits[0].cpu()


def recog_from_logits(logits):
    ids = torch.argmax(logits, dim=-1).unsqueeze(0)
    txt = _proc.batch_decode(ids)[0]
    return [_strip_p(x) for x in txt.split() if _strip_p(x)]


def recog_phones(wav):
    return recog_from_logits(_logits(wav))


def _ctc_forced_align(logp, tokens, blank=0):
    T = logp.shape[0]; L = len(tokens); S = 2 * L + 1
    ext = [blank] * S
    for i, tk in enumerate(tokens):
        ext[2 * i + 1] = tk
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.full((T, S), -1, dtype=np.int64)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], s
            if s - 1 >= 0 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], s - 2
            if best <= NEG:
                continue
            dp[t, s] = best + logp[t, ext[s]]; bp[t, s] = arg
    s = (S - 2) if (S >= 2 and dp[T - 1, S - 2] > dp[T - 1, S - 1]) else (S - 1)
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s = int(bp[t, s])
    return ext, path


def _gop_targets(text):
    s = phonemize(text, language='en-us', backend='espeak',
                  separator=Separator(phone=' ', word=' | '), strip=True, with_stress=False, njobs=1)
    out = []
    for ph in s.split():
        if ph == '|':
            continue
        ph = ph.replace('ˈ', '').replace('ˌ', '').strip()
        if not ph:
            continue
        if ph in _VOCAB:
            out.append((ph, _VOCAB[ph]))
        else:
            st = _strip_p(ph)
            if st in _VOCAB:
                out.append((st, _VOCAB[st]))
    return out


def _gop_eval(wav, text):
    tg = _gop_targets(text)
    if not tg:
        return {"ok": False, "err": "no target phones"}
    tphones = [p for p, _ in tg]; toks = [t for _, t in tg]
    logits = _logits(wav)
    said = recog_from_logits(logits)
    logp = torch.log_softmax(logits, dim=-1).numpy()
    T = logp.shape[0]
    if T < len(toks):
        return {"ok": True, "accuracy": 0, "phones": [{"p": p, "status": "miss"} for p in tphones], "said": said}
    ext, path = _ctc_forced_align(logp, toks, blank=_BLANK)
    scores, phones = [], []
    for i, (tk, ph) in enumerate(zip(toks, tphones)):
        frames = [t for t in range(T) if path[t] == 2 * i + 1]
        if frames:
            raw = float(np.exp(np.mean([logp[t, tk] for t in frames])))
            sc = _calib(raw)
            status = "ok" if sc >= PH_OK else ("warn" if sc >= PH_WARN else "sub")
        else:
            sc, status = 0.0, "miss"
        scores.append(sc)
        phones.append({"p": ph, "status": status, "score": round(sc, 2)})
    return {"ok": True, "accuracy": round(float(np.mean(scores)) * 100), "phones": phones, "said": said}


def align_phones(tgt, hyp):
    la, lb = len(tgt), len(hyp)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): d[i][0] = i
    for j in range(lb + 1): d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if tgt[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    st = ['miss'] * la; i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tgt[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]: st[i - 1] = 'ok'; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1: st[i - 1] = 'sub'; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1: st[i - 1] = 'miss'; i -= 1
        else: j -= 1
    ok = sum(1 for s in st if s == 'ok')
    return st, ok, d[la][lb]


# ===== Whisper =====
def _whisper(wav, lang="en", hint="", fast=False):
    segments, info = _model.transcribe(
        wav, language=(None if lang == "auto" else lang),
        beam_size=(1 if fast else 5), temperature=0.0, condition_on_previous_text=False,
        initial_prompt=(hint or None), vad_filter=(not fast), word_timestamps=(not fast))
    parts, words = [], []
    for seg in segments:
        parts.append(seg.text)
        for wd in (seg.words or []):
            words.append({"w": wd.word.strip(), "start": round(wd.start, 2), "end": round(wd.end, 2)})
    return {"text": "".join(parts).strip(), "words": words, "duration": round(getattr(info, "duration", 0.0), 2)}


def _pron_eval(wav, text):
    tgt = target_phones(text); hyp = recog_phones(wav)
    if not tgt:
        return {"ok": False, "err": "no target phones"}
    st, ok, dist = align_phones(tgt, hyp)
    denom = max(len(tgt), len(hyp), 1)
    acc = max(0.0, 1 - dist / denom)
    return {"ok": True, "accuracy": round(acc * 100), "target": tgt, "said": hyp,
            "phones": [{"p": tgt[k], "status": st[k]} for k in range(len(tgt))]}


# ===== Tiện ích chấm câu =====
def _dnorm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _wsim(a, b):
    a = a.strip(); b = b.strip()
    if not a or not b: return 0.0
    if a == b: return 1.0
    la, lb = len(a), len(b)
    d = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = d[0]; d[0] = i
        for j in range(1, lb + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return max(0.0, 1.0 - d[lb] / max(la, lb))


def _align_words(ref, hyp):
    la, lb = len(ref), len(hyp)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): d[i][0] = i
    for j in range(lb + 1): d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    st = ['miss'] * la; i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]: st[i - 1] = 'ok'; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1: st[i - 1] = 'sub'; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1: st[i - 1] = 'miss'; i -= 1
        else: j -= 1
    ok = sum(1 for x in st if x == 'ok')
    return st, ok


def _band(s):
    if s >= 8: return {"color": "#16a34a", "label": "Đạt", "emoji": "✅"}
    return {"color": "#dc2626", "label": "Không đạt", "emoji": "❌"}


def _fluency(wpm):
    if wpm <= 0: return 0.0
    if 110 <= wpm <= 170: return 10.0
    if wpm < 110: return round(max(3.0, 10.0 - (110 - wpm) / 12.0), 1)
    return round(max(4.0, 10.0 - (wpm - 170) / 15.0), 1)


# ===== Orchestration từng route =====
def do_transcribe(wav, lang, hint, fast):
    r = _whisper(wav, lang, hint, fast)
    return {"ok": True, **r}


def do_pron(wav, text):
    r = _pron_eval(wav, text)
    return r if r.get("ok") else {"ok": False, "err": r.get("err", "pron")}


def do_grade(wav, text, hint):
    w = _whisper(wav, "en", hint, fast=True)
    p = _pron_eval(wav, text)
    out = {"ok": True, "text": w["text"], "duration": w["duration"]}
    if p.get("ok"):
        out.update(accuracy=p["accuracy"], target=p["target"], said=p["said"], phones=p["phones"])
    return out


def do_grade_ph(wav, current):
    r = _gop_eval(wav, current)
    if not r.get("ok"):
        return r
    r["word_ok"] = (r["accuracy"] / 100.0 >= PH_MIN) and bool(r.get("said"))
    return r


def do_score(wav, text, words_hint):
    tnorm = _dnorm(text)
    ntoks = len(tnorm.split()) if tnorm else 0
    hint = "" if ntoks > 1 else (words_hint if words_hint else text)
    w = _whisper(wav, "en", hint, fast=True)
    p = _pron_eval(wav, text)
    transcript = (w.get("text") or "").strip()
    pacc = p.get("accuracy") if p.get("ok") else None
    pacc = None if pacc is None else pacc / 100.0
    phones = p.get("phones") if p.get("ok") else []
    marks = None
    if not transcript:
        score, status = 0.0, "miss"
    elif ntoks > 1:
        ref = tnorm.split(); hyp = _dnorm(transcript).split()
        st, ok = _align_words(ref, hyp)
        wr = ok / max(1, len(ref))
        bonus = wr * (1.0 if pacc is None else 3.0 * pacc)
        score = round(min(10.0, wr * 7.0 + bonus), 1)
        status = "ok" if score >= 8.5 else ("warn" if score >= 7 else "sub")
        marks = [{"w": ref[k], "ok": st[k] == "ok"} for k in range(len(ref))]
    else:
        best = 0.0
        for wd in _dnorm(transcript).split():
            best = max(best, _wsim(wd, tnorm))
        if tnorm and _dnorm(transcript) == tnorm: best = 1.0
        if best >= 0.85:
            bonus = 1.0 if pacc is None else 3.0 * pacc
            score = round(min(10.0, 7.0 + bonus), 1)
            status = "ok" if score >= 8.5 else "warn"
        elif best >= 0.5:
            score = round(4.0 + 2.0 * best, 1); status = "sub"
        else:
            score = round(3.0 * best, 1); status = "sub"
    out = {"ok": True, "score": score, "status": status, "band": _band(score),
           "heard": transcript, "marks": marks, "phones": phones}
    # Fluency + completeness cho CÂU (IELTS)
    if ntoks > 1 and transcript:
        dur = float(w.get("duration") or 0)
        hw = len(_dnorm(transcript).split())
        wpm = round(hw / dur * 60) if dur > 0 else 0
        fl = _fluency(wpm)
        comp = round(min(1.0, hw / max(1, len(tnorm.split()))) * 100)
        out.update(fluency=fl, wpm=wpm, completeness=comp,
                   criteria={"accuracy": (None if pacc is None else round(pacc * 100)),
                             "fluency": fl, "completeness": comp, "wpm": wpm})
    return out


ROUTES = {"score", "grade", "grade_ph", "transcribe", "pron"}


def handler(job):
    inp = job.get("input") or {}
    route = (inp.get("route") or "score").strip()
    if route == "health":
        return {"ok": True, "model": MODEL_NAME, "w2v": W2V, "device": DEVICE}
    if route not in ROUTES:
        return {"ok": False, "err": f"unknown route '{route}'"}

    b64 = inp.get("audio_b64") or inp.get("audio")
    if not b64:
        return {"ok": False, "err": "no audio (cần 'audio_b64')"}
    try:
        audio = base64.b64decode(b64)
    except Exception as e:
        return {"ok": False, "err": f"base64 decode failed: {e}"}
    if not audio or len(audio) > MAX_BYTES:
        return {"ok": False, "err": "audio rỗng hoặc quá lớn"}

    text = (inp.get("text") or "").strip()
    if route in ("score", "grade", "grade_ph", "pron") and not text:
        return {"ok": False, "err": "no text"}

    tmpd = tempfile.mkdtemp(prefix="ss_")
    src = os.path.join(tmpd, "in")
    wav = None
    try:
        with open(src, "wb") as f:
            f.write(audio)
        wav = to_wav(src)
        lang = (inp.get("lang") or "en")
        hint = (inp.get("prompt") or "").strip()[:400]
        fast = str(inp.get("fast", "")).lower() in ("1", "true", "yes")
        if route == "transcribe":
            return do_transcribe(wav, lang, hint, fast)
        if route == "pron":
            return do_pron(wav, text)
        if route == "grade":
            return do_grade(wav, text, hint)
        if route == "grade_ph":
            return do_grade_ph(wav, text)
        return do_score(wav, text, (inp.get("words") or "").strip())
    except subprocess.CalledProcessError:
        return {"ok": False, "err": "audio decode failed"}
    except Exception as e:
        return {"ok": False, "err": str(e)}
    finally:
        for p in (src, wav):
            try:
                if p and os.path.exists(p): os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmpd)
        except Exception:
            pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
