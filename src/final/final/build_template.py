#!/usr/bin/env python3
"""Combine several recordings of the alarm into ONE averaged template.

    python3 build_template.py alarm_takes/alarm_0*_raw.wav -o alarm_template.npz

Why this does not average the waveforms: the takes start at different points in
the loop, so their phases never line up and averaging samples would cancel the
sound. Instead every buzz is found in every take, its (gain-invariant) spectrum is
computed, and the spectra are averaged: first within a take, then across takes, so
each take counts equally. Thresholds and timing are derived from the same data.

Use the *_raw.wav files (they include quiet before/after the alarm, which gives a
better noise-floor estimate than the trimmed ones).
"""
import argparse
import sys

import numpy as np

from alarm_dsp import FRAME, HOP, Detector, Features, find_runs, inband_db, normalize, read_wav, smooth

EDGE_S = 0.10   # ignore the first/last 100 ms of each buzz (on/off transients)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wavs", nargs="+", help="recordings of the alarm (same sample rate)")
    p.add_argument("-o", "--out", default="alarm_template.npz")
    p.add_argument("--bands", type=int, default=64)
    p.add_argument("--frame", type=int, default=FRAME,
                   help="STFT window in samples (default %d = 85.3 ms at 48 kHz; use a power of two)" % FRAME)
    p.add_argument("--hop", type=int, default=HOP, help="samples between windows (default %d)" % HOP)
    p.add_argument("--fmin", type=float, default=200.0)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--margin-db", type=float, default=10.0,
                   help="dB above the noise floor that counts as sound")
    p.add_argument("--min-gap", type=float, default=0.10,
                   help="silences shorter than this (s) do not split a buzz")
    p.add_argument("--min-len", type=float, default=0.20, help="shorter sounds are ignored (s)")
    p.add_argument("--n-required", type=int, default=0,
                   help="buzzes needed to trigger (default: 3 if buzzes are <1.6 s long, else 1)")
    a = p.parse_args()

    # ---- load
    audio = [(w,) + read_wav(w) for w in a.wavs]
    rates = {r for _, _, r in audio}
    if len(rates) != 1:
        sys.exit("All recordings must have the same sample rate, got %s" % sorted(rates))
    sr = rates.pop()
    F = Features(sr, a.bands, a.fmin, a.fmax, a.frame, a.hop)
    hop_s = F.hop_s
    bridge = int(round(a.min_gap / hop_s))
    min_len = int(round(a.min_len / hop_s))
    edge = int(round(EDGE_S / hop_s))

    # ---- which bands carry the alarm? Those whose level swings the most (loud when the
    #      alarm sounds, quiet in the pauses), averaged over all takes.
    feats = []
    scores = []
    for path, x, _ in audio:
        vecs, bmags = F.signal_feats(x)
        bdb = 20.0 * np.log10(bmags + 1e-9)
        scores.append(np.percentile(bdb, 90, axis=0) - np.percentile(bdb, 10, axis=0))
        feats.append((vecs, bmags))
    k = max(4, F.n_bands // 6)
    mask = np.sort(np.argsort(np.mean(scores, axis=0))[-k:])
    band_hz = np.sqrt(F.edges[:-1] * F.edges[1:])
    print("Alarm bands (Hz): " + ", ".join("%.0f" % f for f in band_hz[mask]) + "\n")

    # ---- find the buzzes in each take
    takes = []
    for (path, x, _), (vecs, bmags) in zip(audio, feats):
        db = inband_db(bmags, mask)
        noise = float(np.percentile(db, 10))
        runs = find_runs(db > noise + a.margin_db, bridge, min_len)
        core = [vecs[s + edge: e - edge] for s, e in runs if e - s > 2 * edge]   # one array per buzz
        if not core:
            sys.exit("%s: no alarm found (nothing %.0f dB above the noise floor). "
                     "Re-record, or lower --margin-db." % (path, a.margin_db))
        takes.append(dict(path=path, noise=noise, runs=runs, buzzes=core, core=np.vstack(core)))
        lens = [(e - s) * hop_s for s, e in runs]
        print("%s: %d sound segments, length median %.2fs"
              % (path, len(runs), np.median(lens)))

    # ---- average the spectra: within each take, then across takes
    means = [normalize(t["core"].mean(axis=0)) for t in takes]
    ref = normalize(np.mean(means, axis=0))

    # ---- how consistent are the takes? (leave-one-take-out)
    pooled = []
    print("\nLeave-one-take-out similarity (each take vs. the average of the others):")
    for i, t in enumerate(takes):
        others = [m for j, m in enumerate(means) if j != i]
        ref_i = normalize(np.mean(others, axis=0)) if others else ref
        s = np.concatenate([smooth(c @ ref_i) for c in t["buzzes"]])   # same smoothing as the detector
        pooled.append(s)
        print("  %s: median %.3f, 5th percentile %.3f" % (t["path"], np.median(s), np.percentile(s, 5)))
    if len(takes) == 1:
        print("  (only one take: this is optimistic. Use 3 or more.)")
    pooled = np.concatenate(pooled)
    p5 = float(np.percentile(pooled, 5))
    sim_thr = float(np.clip(p5 - 0.04, 0.75, 0.93))

    # ---- timing
    seg_lens = np.array([(e - s) * hop_s for t in takes for s, e in t["runs"]])
    gaps = [(t["runs"][k + 1][0] - t["runs"][k][1]) * hop_s
            for t in takes for k in range(len(t["runs"]) - 1)]
    seg_len = float(np.median(seg_lens))
    seg_tol = float(min(0.35, 0.4 * seg_len))
    gap_med = float(np.median(gaps)) if gaps else 1.0
    n_required = a.n_required or (3 if seg_len < 1.6 else 1)
    window_s = float(np.clip(n_required * (seg_len + gap_med) + 2.0, 4.0, 15.0))

    np.savez(a.out, ref=ref, mask=mask, sr=sr, frame=F.frame, hop=F.hop, n_bands=F.n_bands, fmin=F.fmin, fmax=F.fmax,
             sim_thr=sim_thr, margin_db=a.margin_db, seg_len_s=seg_len, seg_tol_s=seg_tol,
             n_required=n_required, window_s=window_s, min_gap_s=a.min_gap, min_len_s=a.min_len)

    print("\nTemplate from %d take(s), %d buzzes, sample rate %d Hz" % (
        len(takes), len(seg_lens), sr))
    print("  STFT window %d samples (%.1f ms, %.1f Hz bins), hop %d samples (%.1f ms)" % (
        F.frame, 1000.0 * F.frame_s, sr / float(F.n_fft), F.hop, 1000.0 * F.hop_s))
    print("  buzz length %.2fs (+/- %.2f), median gap between sounds %.2fs" % (seg_len, seg_tol, gap_med))
    print("  trigger: %d matched buzz(es) within a %.1fs window, similarity >= %.2f" % (
        n_required, window_s, sim_thr))
    if seg_len >= 1.6 and not a.n_required:
        print("  NOTE: sounds are long (%.1fs), so your 3 buzzes seem to run together; "
              "n_required was set to 1." % seg_len)
    print("  saved %s" % a.out)

    # ---- self-test: replay the training takes through the streaming detector
    print("\nSelf-test (replaying the same takes; optimistic, real tests come later):")
    tpl = dict(np.load(a.out))
    tpl = {k: (v.item() if v.ndim == 0 else v) for k, v in tpl.items()}
    for path, x, _ in audio:
        det = Detector(tpl)
        events = []
        for i in range(0, len(x), 1024):
            for ev in det.feed(x[i: i + 1024]):
                events.append("%s @ %.2fs" % (ev, det.time_s))
        print("  %s: %s" % (path, ", ".join(events) or "NOT DETECTED"))


if __name__ == "__main__":
    main()