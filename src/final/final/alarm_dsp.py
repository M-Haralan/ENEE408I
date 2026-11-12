#!/usr/bin/env python3
"""Shared DSP for the alarm detector: STFT band features, buzz segmentation and a
streaming detector. Also an offline replay tool, so you can test the detector on
WAV files without ROS:

    python3 alarm_dsp.py replay alarm_template.npz recording.wav [more.wav ...]

Requires only numpy (Python 3.8+).
"""
import argparse
import collections
import wave

import numpy as np

FRAME = 4096             # STFT window in samples (85.3 ms at 48 kHz, true 11.7 Hz resolution)
HOP = 1024               # STFT hop in samples (21.3 ms at 48 kHz)
LEGACY_FRAME_S = 0.050   # only used for templates built before frame/hop were stored in them
LEGACY_HOP_S = 0.025
DR_DB = 40.0             # band levels more than this far below the frame's loudest band are clamped
SIM_SMOOTH = 4           # similarity is averaged over this many frames (100 ms) before thresholding
NOISE_WINDOW_S = 30.0    # noise floor = 10th percentile of in-band level over this long


# ------------------------------------------------------------------ file I/O
def read_wav(path):
    """16-bit PCM WAV -> (int16 mono array, sample rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("%s: only 16-bit PCM WAV is supported" % path)
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    return x.astype(np.int16), rate


def load_template(path):
    d = np.load(path, allow_pickle=False)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


# ------------------------------------------------------------------ features
class Features:
    """Per-frame loudness (dB) and a gain-invariant log-band spectrum vector."""

    def __init__(self, sr, n_bands=64, fmin=200.0, fmax=8000.0, frame=FRAME, hop=HOP):
        self.sr = int(sr)
        self.frame = int(frame)     # window length in samples
        self.hop = int(hop)         # samples between windows
        self.hop_s = self.hop / float(self.sr)
        self.frame_s = self.frame / float(self.sr)
        self.n_fft = 1 << (self.frame - 1).bit_length()
        self.n_bands = int(n_bands)
        self.fmin = float(fmin)
        self.fmax = float(min(fmax, 0.45 * self.sr))
        self.edges = np.geomspace(self.fmin, self.fmax, self.n_bands + 1)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        lo = np.searchsorted(freqs, self.edges[:-1])
        hi = np.maximum(np.searchsorted(freqs, self.edges[1:]), lo + 1)
        self.lo = lo
        self.hi = np.minimum(hi, len(freqs))
        self.win = np.hanning(self.frame)

    def frame_feats(self, frame):
        """frame: int16 array of length self.frame -> (unit band vector, linear band magnitudes)."""
        x = frame.astype(np.float64) / 32768.0
        mag = np.abs(np.fft.rfft(x * self.win, self.n_fft))
        c = np.concatenate(([0.0], np.cumsum(mag)))
        bmag = c[self.hi] - c[self.lo]
        v = 20.0 * np.log10(bmag + 1e-9)                      # band magnitude in dB
        v = np.maximum(v, v.max() - DR_DB)                    # ignore the noise-floor bands
        v = v - v.mean()                                      # removes overall gain
        n = np.linalg.norm(v)
        return (v / n if n > 1e-6 else np.zeros_like(v)), bmag

    def signal_feats(self, x):
        """Whole signal -> (band vectors [frames x bands], band magnitudes [frames x bands])."""
        n = (len(x) - self.frame) // self.hop + 1
        vecs = np.empty((n, self.n_bands))
        bmags = np.empty((n, self.n_bands))
        for i in range(n):
            vecs[i], bmags[i] = self.frame_feats(x[i * self.hop: i * self.hop + self.frame])
        return vecs, bmags


def inband_db(bmag, mask):
    """Level (dB) of the energy inside the alarm's own bands: far more robust to
    motor/broadband noise than total loudness. Works on one frame or an array of frames."""
    return 20.0 * np.log10(np.asarray(bmag)[..., mask].sum(axis=-1) + 1e-9)


def find_runs(on, bridge, min_len):
    """Runs of True in `on` as (start, end) frame indices, end exclusive.
    Gaps of <= `bridge` frames are bridged; runs shorter than `min_len` frames are dropped."""
    d = np.diff(np.r_[0, np.asarray(on, dtype=np.int8), 0])
    merged = []
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if merged and s - merged[-1][1] <= bridge:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    return [(s, e) for s, e in merged if e - s >= min_len]


def smooth(s, n=SIM_SMOOTH):
    """Moving average (valid part only) of a 1-D similarity track."""
    s = np.asarray(s, dtype=float)
    return np.convolve(s, np.ones(n) / n, mode="valid") if len(s) >= n else s


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ------------------------------------------------------------------ detector
class Detector:
    """Streaming detector. feed() int16 samples; it returns a list of events:
         'heard'   - the alarm pattern was just recognised (once per alarm episode)
         'cleared' - the alarm has been gone for `rearm_s`; detector re-armed
    Stage 1 (per frame): energy in the alarm's bands is well above the noise floor AND the
             spectrum matches the template.
    Stage 2 (rolling window): >= n_required matched segments of about the template's buzz length.
    """

    def __init__(self, tpl, sim_thr=0.0, margin_db=0.0, window_s=0.0,
                 rearm_s=5.0, warmup_s=5.0, log=False):
        sr = int(tpl["sr"])
        frame = int(tpl["frame"]) if "frame" in tpl else int(round(sr * LEGACY_FRAME_S))
        hop = int(tpl["hop"]) if "hop" in tpl else int(round(sr * LEGACY_HOP_S))
        self.f = Features(sr, int(tpl["n_bands"]), float(tpl["fmin"]), float(tpl["fmax"]), frame, hop)
        self.ref = np.asarray(tpl["ref"], dtype=np.float64)
        self.mask = np.asarray(tpl["mask"], dtype=int)
        self.sim_thr = float(sim_thr or tpl["sim_thr"])
        self.margin_db = float(margin_db or tpl["margin_db"])
        self.window_s = float(window_s or tpl["window_s"])
        self.seg_len = float(tpl["seg_len_s"])
        self.seg_tol = float(tpl["seg_tol_s"])
        self.n_required = int(tpl["n_required"])
        hop_s = self.f.hop_s
        self.bridge = int(round(float(tpl["min_gap_s"]) / hop_s))
        self.min_len = int(round(float(tpl["min_len_s"]) / hop_s))
        self.rearm_s = float(rearm_s)
        self.warmup_s = float(warmup_s)
        self.buf = np.zeros(0, dtype=np.int16)
        self.db_hist = collections.deque(maxlen=int(NOISE_WINDOW_S / hop_s))
        self.on_hist = collections.deque(maxlen=int(round(self.window_s / hop_s)))
        self.sims = collections.deque(maxlen=SIM_SMOOTH)
        self.recent = collections.deque(maxlen=max(1, int(round(1.0 / hop_s))))   # last ~1 s of (sim, matched)
        self.n_ok = 0                                                              # qualifying buzzes in the window
        self.nframes = 0
        self.armed = True
        self.quiet = 0
        self.last = (0.0, 0.0, 0.0, False)   # in-band level dB, noise floor dB, similarity, matched
        self.log = [] if log else None

    @property
    def time_s(self):
        return (self.nframes - 1) * self.f.hop_s + self.f.frame_s

    def summary(self):
        """Last ~1 s at a glance (for debug logging): level, noise floor, best similarity,
        fraction of frames matched, and how many qualifying buzzes are in the window."""
        if not self.recent:
            return None
        sims = [a for a, _ in self.recent]
        return dict(level=self.last[0], noise=self.last[1], best_sim=max(sims),
                    matched=sum(1 for _, b in self.recent if b) / float(len(self.recent)),
                    buzzes=self.n_ok)

    def feed(self, x):
        self.buf = np.concatenate((self.buf, np.asarray(x, dtype=np.int16)))
        events = []
        i = 0
        f = self.f
        while len(self.buf) - i >= f.frame:
            ev = self._step(self.buf[i: i + f.frame])
            if ev:
                events.append(ev)
            i += f.hop
        self.buf = self.buf[i:]
        return events

    def _pattern(self):
        on = np.fromiter(self.on_hist, dtype=bool, count=len(self.on_hist))
        hop_s = self.f.hop_s
        n_ok = 0
        for s, e in find_runs(on, self.bridge, self.min_len):
            length = (e - s) * hop_s
            if e == len(on):                       # still sounding: long enough so far is good enough
                good = length >= self.seg_len - self.seg_tol
            else:
                good = abs(length - self.seg_len) <= self.seg_tol
            n_ok += int(good)
        self.n_ok = n_ok
        return n_ok >= self.n_required

    def _step(self, frame):
        v, bmag = self.f.frame_feats(frame)
        self.sims.append(float(v @ self.ref))
        sim = float(np.mean(self.sims))
        db = float(inband_db(bmag, self.mask))
        self.db_hist.append(db)
        self.nframes += 1
        noise = float(np.percentile(np.fromiter(self.db_hist, dtype=float, count=len(self.db_hist)), 10))
        loud = db > noise + self.margin_db
        on = bool(loud and sim >= self.sim_thr)
        self.on_hist.append(on)
        self.recent.append((sim, on))
        self.last = (db, noise, sim, on)
        warm = self.nframes * self.f.hop_s >= self.warmup_s
        if self.log is not None:
            self.log.append((self.time_s, db, noise, sim, on, bool(loud and warm)))
        if not warm:
            return None
        ok = self._pattern()
        if self.armed:
            if ok:
                self.armed = False
                self.quiet = 0
                return "heard"
        elif ok:
            self.quiet = 0
        else:
            self.quiet += 1
            if self.quiet * self.f.hop_s >= self.rearm_s:
                self.armed = True
                return "cleared"
        return None


# -------------------------------------------------------------------- replay
def replay(tpl_path, wav_path, chunk=1024, **kw):
    tpl = load_template(tpl_path)
    x, rate = read_wav(wav_path)
    if rate != int(tpl["sr"]):
        raise SystemExit("%s is %d Hz but the template is %d Hz; record/convert at the same rate."
                         % (wav_path, rate, int(tpl["sr"])))
    det = Detector(tpl, log=True, **kw)
    events = []
    for i in range(0, len(x), chunk):
        for ev in det.feed(x[i: i + chunk]):
            events.append((det.time_s, ev))
    loud_sims = np.array([r[3] for r in det.log if r[5]])
    print("%s  (%.1fs)" % (wav_path, len(x) / float(rate)))
    if events:
        print("  events: " + ", ".join("%s @ %.2fs" % (e, t) for t, e in events))
    else:
        print("  events: none")
    if len(loud_sims):
        print("  loud frames: %d   similarity p50 %.2f  p90 %.2f  max %.2f   (sim_thr %.2f)" % (
            len(loud_sims), np.percentile(loud_sims, 50), np.percentile(loud_sims, 90),
            loud_sims.max(), det.sim_thr))
    else:
        print("  no frames more than %.0f dB above the in-band noise floor" % det.margin_db)
    return events


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="run WAV file(s) through the detector")
    r.add_argument("template")
    r.add_argument("wavs", nargs="+")
    r.add_argument("--sim-thr", type=float, default=0.0, help="override the template's threshold")
    r.add_argument("--margin-db", type=float, default=0.0)
    a = p.parse_args()
    for w in a.wavs:
        replay(a.template, w, sim_thr=a.sim_thr, margin_db=a.margin_db)


if __name__ == "__main__":
    main()