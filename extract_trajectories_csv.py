#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_trajectories_csv.py

Read one or more rosbags, extract hamster (x, y, yaw) from TF,
write each to its own CSV. No matplotlib dependency.

Usage:
  python2.7 extract_trajectories_csv.py bag1.bag bag2.bag bag3.bag
"""
#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
extract_trajectories_csv.py

Read one or more rosbags, extract hamster pose from TF,
write each to its own CSV.

Output columns:
    t_sec, x, y, z, qx, qy, qz, qw, yaw_rad

Usage:
    python2.7 extract_trajectories_csv.py bag1.bag bag2.bag
"""

import argparse
import os
import csv
import math

import rosbag
import tf.transformations as tft


def extract(bag_path, parent="mocap", child="hamster"):
    raw_rows = []

    try:
        bag = rosbag.Bag(bag_path)
    except Exception as e:
        print("Cannot open %s: %s" % (bag_path, e))
        return []

    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for tr in msg.transforms:
            if tr.header.frame_id == parent and tr.child_frame_id == child:
                stamp = tr.header.stamp.to_sec()

                trans = tr.transform.translation
                rot = tr.transform.rotation

                qx = rot.x
                qy = rot.y
                qz = rot.z
                qw = rot.w

                _, _, yaw = tft.euler_from_quaternion([qx, qy, qz, qw])

                raw_rows.append((
                    stamp,
                    trans.x, trans.y, trans.z,
                    qx, qy, qz, qw,
                    yaw
                ))

    bag.close()

    if not raw_rows:
        return []

    raw_rows.sort(key=lambda r: r[0])
    t0 = raw_rows[0][0]

    rows = []
    for r in raw_rows:
        rows.append((
            r[0] - t0,   # t_sec
            r[1], r[2], r[3],   # x y z
            r[4], r[5], r[6], r[7],   # qx qy qz qw
            r[8]   # yaw_rad
        ))

    return rows


def write_csv(rows, out_path):
    with open(out_path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['t_sec', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw_rad'])
        for r in rows:
            w.writerow([
                "%.6f" % r[0],
                "%.6f" % r[1],
                "%.6f" % r[2],
                "%.6f" % r[3],
                "%.6f" % r[4],
                "%.6f" % r[5],
                "%.6f" % r[6],
                "%.6f" % r[7],
                "%.6f" % r[8],
            ])


def path_length(rows):
    if len(rows) < 2:
        return 0.0

    d = 0.0
    for i in range(1, len(rows)):
        dx = rows[i][1] - rows[i - 1][1]
        dy = rows[i][2] - rows[i - 1][2]
        d += math.sqrt(dx * dx + dy * dy)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--parent", default="mocap")
    ap.add_argument("--child", default="hamster")
    args = ap.parse_args()

    for bag_path in args.bags:
        base = os.path.splitext(os.path.basename(bag_path))[0]
        out = base + "_traj.csv"

        rows = extract(bag_path, args.parent, args.child)
        if not rows:
            print("%s: no TF found" % bag_path)
            continue

        write_csv(rows, out)

        dur = rows[-1][0]
        dist = path_length(rows)
        avg_speed = dist / dur if dur > 0 else 0.0

        print("%s: N=%d  duration=%.2fs  path=%.3fm  avg=%.3f m/s  -> %s"
              % (bag_path, len(rows), dur, dist, avg_speed, out))


if __name__ == "__main__":
    main()