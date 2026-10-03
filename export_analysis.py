#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
import numpy as np
from bisect import bisect_right
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Read GT CSV
# ----------------------------
def load_gt_csv(path):
    """
    Load GT trajectory from CSV.

    Expected columns:
        t_sec, x, y, z, qx, qy, qz, qw

    Returns:
        ts : (N,) timestamps
        ps : (N,3) positions
        qs : (N,4) quaternions in xyzw order
    """
    ts, ps, qs = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["t_sec"])
            p = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            q = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float64)  # xyzw
            ts.append(t); ps.append(p); qs.append(q)

    if len(ts) == 0:
        raise RuntimeError("Empty CSV: %s" % path)

    # Sort by time, just in case.
    order = np.argsort(np.array(ts))
    ts = np.array([ts[i] for i in order], dtype=np.float64)
    ps = np.vstack([ps[i] for i in order]).astype(np.float64)
    qs = np.vstack([qs[i] for i in order]).astype(np.float64)
    return ts, ps, qs


# ----------------------------
# Quaternion utils (xyzw)
# ----------------------------
def q_normalize(q):
    """
    Normalize quaternion to unit length.

    Valid rotation quaternions should satisfy:
        ||q|| = 1
    """
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def enforce_sign_continuity(qs_xyzw):
    """
    Make quaternion signs consistent across time.

    Important:
        q and -q represent the same rotation.

    But for interpolation, sudden sign flips are bad.
    So if dot(q_i, q_{i-1}) < 0, flip q_i.
    """
    qs = np.array([q_normalize(q) for q in qs_xyzw], dtype=np.float64)
    for i in range(1, qs.shape[0]):
        if float(np.dot(qs[i], qs[i-1])) < 0.0:
            qs[i] *= -1.0
    return qs


def slerp(q0, q1, a):
    """
    Quaternion SLERP.

    a in [0, 1]:
        a = 0 -> q0
        a = 1 -> q1

    Formula:
        theta = acos(q0 . q1)

        q(a) = sin((1-a)*theta)/sin(theta) * q0
             + sin(a*theta)/sin(theta)     * q1
    """
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)

    dot = float(np.dot(q0, q1))

    # q and -q are the same rotation.
    # Flip q1 if needed so interpolation uses the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Numerical safety for acos
    dot = max(-1.0, min(1.0, dot))

    # If q0 and q1 are very close, use linear interpolation as fallback.
    if dot > 0.9995:
        return q_normalize(q0 + a * (q1 - q0))

    theta = math.acos(dot)
    s0 = math.sin((1.0 - a) * theta) / math.sin(theta)
    s1 = math.sin(a * theta) / math.sin(theta)
    return q_normalize(s0 * q0 + s1 * q1)


# ----------------------------
# Cubic cumulative B-spline basis (degree=3)
# ----------------------------
def b3(u):
    """
    Cubic B-spline basis b3(u), support on [0, 4).

    Outside [0,4), the value is 0.
    """
    if u < 0.0 or u >= 4.0:
        return 0.0
    if u < 1.0:
        return (u**3) / 6.0
    if u < 2.0:
        x = u - 1.0
        return (-3.0*x**3 + 3.0*x**2 + 3.0*x + 1.0) / 6.0
    if u < 3.0:
        x = u - 2.0
        return (3.0*x**3 - 6.0*x**2 + 4.0) / 6.0
    x = 4.0 - u
    return (x**3) / 6.0


def c3(x):
    """
    Cumulative cubic basis used in the cumulative B-spline form.

    Formula:
        c3(x) = 0,                              x <= 0
        c3(x) = 1,                              x >= 3
        c3(x) = b3(x) + b3(x-1) + b3(x-2),     otherwise
    """
    if x <= 0.0:
        return 0.0
    if x >= 3.0:
        return 1.0
    return b3(x) + b3(x - 1.0) + b3(x - 2.0)


def pos_cumulative_bspline(t, ts, ps, dt_knot):
    """
    Position interpolation with uniform-knot cumulative cubic B-spline.

    Real time t is mapped to knot-space:
        u = (t - t0) / dt_knot

    Then position is built in cumulative difference form:
        p(u) = p_f + sum (p_i - p_{i-1}) * c3(u - i)
    """
    M = ps.shape[0]
    if M == 1:
        return ps[0].copy()
    if dt_knot <= 1e-9:
        return ps[0].copy()

    t0 = float(ts[0])
    u = (float(t) - t0) / float(dt_knot)

    k = int(math.floor(u))
    f = k - 4  # for degree 3 cumulative form: f = floor(u) - 4

    # Clamp to valid index range
    if f < 0: f = 0
    if k < 0: k = 0
    if k >= M: k = M - 1
    if f >= M: f = M - 1

    # Start from p_f, then accumulate weighted control-point differences.
    p = ps[f].copy()
    for i in range(f + 1, k + 1):
        if i >= M:
            break
        dp = ps[i] - ps[i - 1]
        w = c3(u - float(i))
        p += dp * w
    return p


class GTInterpolator(object):
    """
    Continuous GT interpolator.

    Position:
        - cumulative B-spline (default if enabled)
        - or linear interpolation

    Rotation:
        - quaternion SLERP between neighboring samples
    """
    def __init__(self, ts, ps, qs_xyzw, use_bspline_pos=True):
        self.ts = np.asarray(ts, dtype=np.float64)
        self.ps = np.asarray(ps, dtype=np.float64)
        self.qs = enforce_sign_continuity(qs_xyzw)
        self.use_bspline_pos = bool(use_bspline_pos)

        # Use median timestamp spacing as the uniform knot spacing.
        if self.ts.shape[0] > 1:
            d = np.diff(self.ts)
            d = d[np.isfinite(d) & (d > 0)]
            self.dt_knot = float(np.median(d)) if d.size > 0 else 0.0
        else:
            self.dt_knot = 0.0

    def query(self, t):
        """
        Query interpolated pose at time t.

        Returns:
            p : interpolated position
            q : interpolated quaternion (xyzw)
        """
        t = float(t)

        # Position interpolation
        if self.use_bspline_pos:
            p = pos_cumulative_bspline(t, self.ts, self.ps, self.dt_knot)
        else:
            p = self._pos_linear(t)

        # Rotation interpolation
        q = self._rot_slerp(t)
        return p, q

    def _pos_linear(self, t):
        """
        Linear position interpolation between neighboring timestamps.

        If t is between t_j and t_{j+1}, then:
            a = (t - t_j) / (t_{j+1} - t_j)
            p(t) = (1-a) p_j + a p_{j+1}
        """
        if t <= self.ts[0]:
            return self.ps[0].copy()
        if t >= self.ts[-1]:
            return self.ps[-1].copy()

        # Find the interval [t_j, t_{j+1}] containing t
        j = bisect_right(self.ts, t) - 1
        j = max(0, min(j, len(self.ts) - 2))

        t0, t1 = self.ts[j], self.ts[j + 1]
        a = (t - t0) / max(1e-12, (t1 - t0))
        return (1.0 - a) * self.ps[j] + a * self.ps[j + 1]

    def _rot_slerp(self, t):
        """
        Rotation interpolation with SLERP between neighboring quaternion samples.

        If t is between t_j and t_{j+1}, then:
            a = (t - t_j) / (t_{j+1} - t_j)
            q(t) = SLERP(q_j, q_{j+1}, a)
        """
        if t <= self.ts[0]:
            return self.qs[0].copy()
        if t >= self.ts[-1]:
            return self.qs[-1].copy()

        j = bisect_right(self.ts, t) - 1
        j = max(0, min(j, len(self.ts) - 2))
        t0, t1 = self.ts[j], self.ts[j + 1]
        a = (t - t0) / max(1e-12, (t1 - t0))
        return slerp(self.qs[j], self.qs[j + 1], a)

# todo 随便用哪个bag 来跑都行
def main():
    """
    Load GT CSV, build the interpolator, resample the trajectory, and plot it.

    Figure 1:
        XY trajectory

    Figure 2:
        x(t) and y(t)
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_csv", help="Path to gt.csv (t_sec,x,y,z,qx,qy,qz,qw)")
    ap.add_argument("--dt", type=float, default=0.02, help="Resample step in seconds for plotting (default: 0.02)")
    ap.add_argument("--use_bspline_pos", action="store_true", help="Use cumulative B-spline for position")
    ap.add_argument("--save_prefix", default="", help="If set, saves figures to <prefix>_xy.png and <prefix>_xy_t.png")
    args = ap.parse_args()

    ts, ps, qs = load_gt_csv(args.gt_csv)
    itp = GTInterpolator(ts, ps, qs, use_bspline_pos=args.use_bspline_pos)

    # ============ 打印 Z 维度统计 ============
    z_raw = ps[:, 2]  # 原始 GT 的 Z
    print("\n========== Z (height) statistics ==========")
    print("  samples:        %d" % len(z_raw))
    print("  Z mean:         %.4f m" % np.mean(z_raw))
    print("  Z std:          %.4f m  (= %.2f mm)" % (np.std(z_raw), np.std(z_raw) * 1000))
    print("  Z min:          %.4f m" % np.min(z_raw))
    print("  Z max:          %.4f m" % np.max(z_raw))
    print("  Z peak-to-peak: %.4f m  (= %.2f mm)" % (np.ptp(z_raw), np.ptp(z_raw) * 1000))
    print("  Z median:       %.4f m" % np.median(z_raw))

    # 看相邻帧间的 Z 跳变（识别噪声 vs 真实运动）
    dz = np.diff(z_raw)
    print("\n  Inter-frame |dZ|:")
    print("    mean:    %.3f mm" % (np.mean(np.abs(dz)) * 1000))
    print("    max:     %.3f mm" % (np.max(np.abs(dz)) * 1000))
    print("    p99:     %.3f mm" % (np.percentile(np.abs(dz), 99) * 1000))
    print("===========================================\n")



    t0, t1 = float(ts[0]), float(ts[-1])
    dt = float(args.dt)
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # Build a uniform query-time grid for plotting.
    t_plot = np.arange(t0, t1 + 1e-12, dt, dtype=np.float64)
    P = np.zeros((t_plot.shape[0], 3), dtype=np.float64)

    # Query interpolated position at each plotting time.
    for i, t in enumerate(t_plot):
        p, _ = itp.query(t)
        P[i, :] = p


    # ---- Figure 1: XY trajectory ----
    # Path shape in the ground plane.
    plt.figure()
    plt.plot(P[:, 0], P[:, 1])
    plt.axis('equal')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("GT trajectory (XY)")

    if args.save_prefix:
        plt.savefig(args.save_prefix + "_xy.png", dpi=200)

    # ---- Figure 2: x/y vs time ----
    # How x and y change over time.
    plt.figure()
    t_rel = t_plot - t0
    plt.plot(t_rel, P[:, 0], label="x")
    plt.plot(t_rel, P[:, 1], label="y")
    plt.plot(t_rel, P[:, 2], label="z")
    plt.xlabel("t (s) from start")
    plt.ylabel("position (m)")
    plt.title("GT position vs time")
    plt.legend()

    # ---- Figure 3: Z vs time (zoomed) ----
    plt.figure()
    z_centered = (P[:, 2] - np.mean(P[:, 2])) * 1000.0  # 减去均值，转 mm
    plt.plot(t_rel, z_centered)
    plt.xlabel("t (s) from start")
    plt.ylabel("Z deviation from mean (mm)")
    plt.title("GT Z motion (zoomed, mean-subtracted)")
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    if args.save_prefix:
        plt.savefig(args.save_prefix + "_z.png", dpi=200)

    # NOTE:
    # These two lines add an XY scatter to the current figure.
    # So figure 2 will contain both x/y-vs-time curves and XY scatter points.
    plt.scatter(P[:,0], P[:,1], s=2)
    plt.axis('equal')

    if args.save_prefix:
        plt.savefig(args.save_prefix + "_xy_t.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()