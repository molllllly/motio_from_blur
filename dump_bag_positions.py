#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import argparse
import numpy as np


def read_csv(path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([
                float(row["t"]),
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["yaw_deg"])
            ])
    return np.array(rows, dtype=np.float64)


def yaw_diff_deg(a, b):
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", required=True)
    ap.add_argument("--slow", required=True)
    ap.add_argument("--drop_last_sec", type=float, default=0.0)
    ap.add_argument("--dist_thresh", type=float, default=0.03)  # 3 cm
    ap.add_argument("--yaw_thresh", type=float, default=5.0)    # 5 deg
    args = ap.parse_args()

    fast = read_csv(args.fast)
    slow = read_csv(args.slow)

    if args.drop_last_sec > 0:
        fast_end = fast[:, 0].max()
        keep = fast[:, 0] <= fast_end - args.drop_last_sec
        fast = fast[keep]

    slow_xyz = slow[:, 1:4]
    slow_xy = slow[:, 1:3]
    slow_yaw = slow[:, 4]

    best_ids = []
    best_d_xyz = []
    best_d_xy = []
    best_yaw = []

    for i in range(len(fast)):
        p_xyz = fast[i, 1:4]
        p_xy = fast[i, 1:3]

        d_xy_all = np.linalg.norm(slow_xy - p_xy, axis=1)
        k = int(np.argmin(d_xy_all))

        d_xyz = np.linalg.norm(slow_xyz[k] - p_xyz)
        d_xy = d_xy_all[k]
        dyaw = abs(yaw_diff_deg(fast[i, 4], slow_yaw[k]))

        best_ids.append(k)
        best_d_xyz.append(d_xyz)
        best_d_xy.append(d_xy)
        best_yaw.append(dyaw)

    best_ids = np.array(best_ids)
    best_d_xyz = np.array(best_d_xyz)
    best_d_xy = np.array(best_d_xy)
    best_yaw = np.array(best_yaw)

    good = (best_d_xy <= args.dist_thresh) & (best_yaw <= args.yaw_thresh)

    print("Fast poses checked:", len(fast))
    print("Slow poses:", len(slow))
    print("")
    print("XY nearest distance:")
    print("  mean = %.3f m" % np.mean(best_d_xy))
    print("  p50  = %.3f m" % np.percentile(best_d_xy, 50))
    print("  p90  = %.3f m" % np.percentile(best_d_xy, 90))
    print("  p95  = %.3f m" % np.percentile(best_d_xy, 95))
    print("  max  = %.3f m" % np.max(best_d_xy))
    print("")
    print("Yaw error:")
    print("  mean = %.2f deg" % np.mean(best_yaw))
    print("  p90  = %.2f deg" % np.percentile(best_yaw, 90))
    print("  p95  = %.2f deg" % np.percentile(best_yaw, 95))
    print("  max  = %.2f deg" % np.max(best_yaw))
    print("")
    print("Good matches:")
    print("  good = %d / %d = %.1f%%" % (
        int(good.sum()), len(good), 100.0 * good.sum() / max(len(good), 1)
    ))
    print("  bad  = %d / %d = %.1f%%" % (
        int((~good).sum()), len(good), 100.0 * (~good).sum() / max(len(good), 1)
    ))

    print("\nWorst 20 fast poses:")
    worst = np.argsort(best_d_xy)[-20:][::-1]
    for idx in worst:
        k = best_ids[idx]
        print(
            "fast_i=%4d t=%.6f fast=(%.3f, %.3f, %.3f yaw=%.1f)  "
            "slow_k=%4d slow=(%.3f, %.3f, %.3f yaw=%.1f)  "
            "d_xy=%.3f d_xyz=%.3f dyaw=%.1f"
            % (
                idx,
                fast[idx, 0],
                fast[idx, 1], fast[idx, 2], fast[idx, 3], fast[idx, 4],
                k,
                slow[k, 1], slow[k, 2], slow[k, 3], slow[k, 4],
                best_d_xy[idx], best_d_xyz[idx], best_yaw[idx]
            )
        )


if __name__ == "__main__":
    main()