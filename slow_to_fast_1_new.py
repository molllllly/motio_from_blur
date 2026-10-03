#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rosbag
import numpy as np
import rospy
import cv2
import pickle
import argparse
import signal
import sys
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
import tf.transformations as tft


# ============================================================
# Helper functions
# ============================================================

def tfmsg_to_T(msg):
    t = msg.transform.translation
    q = msg.transform.rotation

    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float64)
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def K_from_info(info):
    return np.array(info.K, dtype=np.float64).reshape(3, 3)


def make_T(R, Tvec):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = Tvec
    return T


def slerp_R(R0, R1, tau):
    M0 = np.eye(4)
    M1 = np.eye(4)
    M0[:3, :3] = R0
    M1[:3, :3] = R1

    q0 = tft.quaternion_from_matrix(M0)
    q1 = tft.quaternion_from_matrix(M1)

    if np.dot(q0, q1) < 0:
        q1 = -q1

    q = tft.quaternion_slerp(q0, q1, float(tau))
    return tft.quaternion_matrix(q)[:3, :3]


def lerp_T(T0, T1, tau):
    return T0 + (T1 - T0) * float(tau)


# ============================================================
# STEP 1: Build slow DB using raw depth
# ============================================================

class SlowBuilder(object):
    def __init__(self, out_path):
        self.out = out_path
        self.bridge = CvBridge()
        self.records = []

        self.world_frame = rospy.get_param("~world_frame", "mocap")
        self.base_frame = rospy.get_param("~base_frame", "hamster")
        self.depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")
        self.color_frame = rospy.get_param("~color_frame", "camera_color_optical_frame")

        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 10.0))
        self.exclude_floor = bool(rospy.get_param("~exclude_floor", False))

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        sub_d = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        sub_di = message_filters.Subscriber("/camera/depth/camera_info", CameraInfo)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_d, sub_di],
            queue_size=20,
            slop=0.03
        )
        self.sync.registerCallback(self.cb)

        self.pub_depth_raw = rospy.Publisher(
            "/slow_debug/depth_raw", Image, queue_size=1)

        self.pub_depth_valid = rospy.Publisher(
            "/slow_debug/depth_valid", Image, queue_size=1)

        self.pub_floor_vis = rospy.Publisher(
            "/slow_debug/floor_vis", Image, queue_size=1)

        signal.signal(signal.SIGINT, self._save_and_exit)

    def _save_and_exit(self, *args):
        with open(self.out, "wb") as f:
            pickle.dump(self.records, f, protocol=2)

        rospy.loginfo("Saved %d slow raw-depth frames to %s",
                      len(self.records), self.out)
        sys.exit(0)

    def _publish_depth32(self, pub, z_m, stamp, frame_id):
        z_pub = np.where(np.isfinite(z_m), z_m, 0.0).astype(np.float32)
        msg = self.bridge.cv2_to_imgmsg(z_pub, encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

    def _make_floor_visualization(self, z, valid_basic, floor_mask):
        non_floor = valid_basic & (~floor_mask)

        if non_floor.sum() > 100:
            vmin = float(np.percentile(z[non_floor], 5))
            vmax = float(np.percentile(z[non_floor], 95))
        else:
            vmin = self.min_depth
            vmax = min(self.max_depth, 5.0)

        z_safe = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
        z_norm = np.clip((z_safe - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
        z_u8 = (z_norm * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(z_u8, cv2.COLORMAP_JET)

        vis[~valid_basic] = (0, 0, 0)
        vis[floor_mask] = (0, 0, 255)

        cv2.putText(
            vis,
            "Red=floor  Black=invalid  JET=valid depth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        return vis

    def _lookup_T_bd(self):
        """
        Return T_bd: base <- depth.
        """
        try:
            return tfmsg_to_T(self.tfbuf.lookup_transform(
                self.base_frame,
                self.depth_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            ))
        except Exception as e1:
            rospy.logwarn_throttle(
                2.0,
                "[SLOW] direct T_bd lookup failed, trying via color: %s" % str(e1)
            )

        T_cb = tfmsg_to_T(self.tfbuf.lookup_transform(
            self.color_frame,
            self.base_frame,
            rospy.Time(0),
            rospy.Duration(0.1)
        ))  # color <- base

        T_cd = tfmsg_to_T(self.tfbuf.lookup_transform(
            self.color_frame,
            self.depth_frame,
            rospy.Time(0),
            rospy.Duration(0.1)
        ))  # color <- depth

        T_bd = np.linalg.inv(T_cb).dot(T_cd)  # base <- depth
        return T_bd

    def cb(self, dmsg, di):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                dmsg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[SLOW] depth decode failed: %s" % str(e))
            return

        if dmsg.encoding == "16UC1":
            z = depth.astype(np.float32) * 0.001
        elif dmsg.encoding == "32FC1":
            z = depth.astype(np.float32)
        else:
            rospy.logwarn_throttle(
                2.0, "[SLOW] unsupported depth encoding: %s" % dmsg.encoding)
            return

        K_d = K_from_info(di)
        H, W = z.shape

        try:
            T_wb = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame,
                self.base_frame,
                dmsg.header.stamp,
                rospy.Duration(0.1)
            ))

            T_bd = self._lookup_T_bd()

        except Exception as e:
            rospy.logwarn_throttle(2.0, "[SLOW] TF failed: %s" % str(e))
            return

        valid_basic = (
            np.isfinite(z) &
            (z > self.min_depth) &
            (z < self.max_depth)
        )

        fx_d, fy_d = K_d[0, 0], K_d[1, 1]
        cx_d, cy_d = K_d[0, 2], K_d[1, 2]

        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        z_safe = np.where(valid_basic, z, 0.0)

        X_d = (uu - cx_d) * z_safe / fx_d
        Y_d = (vv - cy_d) * z_safe / fy_d
        Z_d = z_safe

        P_d = np.stack(
            [X_d, Y_d, Z_d, np.ones_like(z_safe)],
            axis=-1
        )

        T_wd = T_wb.dot(T_bd)  # world <- depth
        P_w = np.einsum("ij,hwj->hwi", T_wd, P_d)

        height_world = P_w[..., 2]

        floor_mask = (
            valid_basic &
            (height_world > -0.01) &
            (height_world < 0.01)
        )

        if floor_mask.sum() > 0:
            rospy.loginfo(
                "[SLOW FLOOR] pixels=%d valid_pct=%.1f%%",
                int(floor_mask.sum()),
                100.0 * floor_mask.sum() / max(valid_basic.sum(), 1)
            )
        else:
            rospy.loginfo_throttle(2.0, "[SLOW FLOOR] no floor pixels found")

        self._publish_depth32(
            self.pub_depth_raw,
            z,
            dmsg.header.stamp,
            dmsg.header.frame_id
        )

        z_valid = np.where(valid_basic, z, 0.0)
        self._publish_depth32(
            self.pub_depth_valid,
            z_valid,
            dmsg.header.stamp,
            dmsg.header.frame_id
        )

        floor_vis = self._make_floor_visualization(z, valid_basic, floor_mask)
        msg = self.bridge.cv2_to_imgmsg(floor_vis, encoding="bgr8")
        msg.header.stamp = dmsg.header.stamp
        msg.header.frame_id = dmsg.header.frame_id
        self.pub_floor_vis.publish(msg)

        depth_to_save = z.copy()
        if self.exclude_floor:
            depth_to_save[floor_mask] = 0.0

        self.records.append({
            "t": dmsg.header.stamp.to_sec(),

            "R": T_wb[:3, :3].copy(),
            "T": T_wb[:3, 3].copy(),

            "depth": depth_to_save.astype(np.float32),

            "K_depth": K_d.copy(),

            "T_bd": T_bd.copy(),

            "depth_frame": self.depth_frame,
            "base_frame": self.base_frame,
            "world_frame": self.world_frame,
        })

        rospy.loginfo(
            "[SLOW RAW] stored=%d valid_depth=%.1f%%",
            len(self.records),
            100.0 * valid_basic.sum() / max(valid_basic.size, 1)
        )


# ============================================================
# STEP 2: Project slow raw depth to fast rolling-shutter image
# ============================================================

class Projector(object):
    def __init__(self, db_path, subscribe=True):
        self.bridge = CvBridge()

        with open(db_path, "rb") as f:
            self.db = pickle.load(f)

        rospy.loginfo("Loaded %d slow raw-depth frames", len(self.db))

        self.world_frame = rospy.get_param("~world_frame", "mocap")
        self.base_frame = rospy.get_param("~base_frame", "hamster")
        self.color_frame = rospy.get_param("~color_frame", "camera_color_optical_frame")

        self.readout = float(rospy.get_param("~readout_sec", 0.030))
        self.exposure = float(rospy.get_param("~exposure_sec", 0.033))

        self.stride = int(rospy.get_param("~stride", 6))
        self.num_slow_neighbors = int(rospy.get_param("~num_slow_neighbors", 10))

        self.slow_select_mode = rospy.get_param("~slow_select_mode", "topk")
        self.slow_window = int(rospy.get_param("~slow_window", 20))
        self.slow_window_step = int(rospy.get_param("~slow_window_step", 2))
        self.slow_window_step = max(1, self.slow_window_step)

        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 10.0))

        self.coverage_cell = int(rospy.get_param("~coverage_cell", 32))
        self.coverage_min_points = int(rospy.get_param("~coverage_min_points", 5))

        self.process_every = int(rospy.get_param("~process_every", 1))
        self.smear_stride = int(rospy.get_param("~smear_stride", 1))
        self.save_every = int(rospy.get_param("~save_every", 5))

        self.debug_max_examples = int(rospy.get_param("~debug_max_examples", 20))
        self.debug_margin = int(rospy.get_param("~debug_margin", 100))

        self.fast_cb_count = 0
        self.fast_done_count = 0
        self.yaw_rates = []

        self.pub_color = rospy.Publisher("/blur_color", Image, queue_size=1)
        self.pub_depth_vis = rospy.Publisher("/sparse_depth_vis", Image, queue_size=1)
        self.pub_smear = rospy.Publisher("/smear_path_vis", Image, queue_size=1)
        self.pub_coverage = rospy.Publisher("/coverage_vis", Image, queue_size=1)

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(3600.0))

        if subscribe:
            self.tfl = tf2_ros.TransformListener(self.tfbuf)

            sub_c = message_filters.Subscriber(
                "/camera/color/image_raw", Image, queue_size=100)
            sub_ci = message_filters.Subscriber(
                "/camera/color/camera_info", CameraInfo, queue_size=100)

            self.sync = message_filters.ApproximateTimeSynchronizer(
                [sub_c, sub_ci],
                queue_size=100,
                slop=0.03
            )
            self.sync.registerCallback(self.cb)
        else:
            self.tfl = None

    # ------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------

    def _inside_label(self, u, v, W, H, margin=0):
        if u is None or v is None:
            return "none"

        if (not np.isfinite(u)) or (not np.isfinite(v)):
            return "nan_uv"

        if 0 <= u < W and 0 <= v < H:
            return "inside"

        if -margin <= u < W + margin and -margin <= v < H + margin:
            return "near_outside"

        if u < -margin:
            return "far_left"
        if u >= W + margin:
            return "far_right"
        if v < -margin:
            return "far_top"
        if v >= H + margin:
            return "far_bottom"

        return "outside"

    def _fmt_counts(self, dic):
        if not dic:
            return "none"
        items = sorted(dic.items(), key=lambda kv: -kv[1])
        return " | ".join(["%s=%d" % (k, v) for k, v in items])

    def _inc(self, dic, key):
        dic[key] = dic.get(key, 0) + 1

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------

    def _yaw_from_R(self, R):
        return np.arctan2(R[1, 0], R[0, 0])

    def _yaw_err_deg(self, Ra, Rb):
        R_rel = Ra.T.dot(Rb)
        dy = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        return abs(np.degrees(dy))

    def _pose_distance(self, slow_R, slow_T, ref_R, ref_T, w_yaw=0.5):
        d_pos = np.linalg.norm(slow_T - ref_T)

        R_rel = slow_R.T.dot(ref_R)
        yaw_diff = abs(np.arctan2(R_rel[1, 0], R_rel[0, 0]))

        return d_pos + w_yaw * yaw_diff

    def _project_tau(self, X_w, R0, T0, R1, T1, tau, K, T_cb):
        ok, u, v, Zc, reason = self._project_tau_full(
            X_w, R0, T0, R1, T1, tau, K, T_cb
        )
        if not ok:
            return None
        return u, v, Zc

    def _project_tau_full(self, X_w, R0, T0, R1, T1, tau, K, T_cb):
        R_b = slerp_R(R0, R1, tau)
        T_b = lerp_T(T0, T1, tau)

        R_bw = R_b.T
        T_bw = -R_bw.dot(T_b)

        T_base_world = np.eye(4, dtype=np.float64)
        T_base_world[:3, :3] = R_bw
        T_base_world[:3, 3] = T_bw

        T_color_world = T_cb.dot(T_base_world)

        X_c = T_color_world[:3, :3].dot(X_w) + T_color_world[:3, 3]
        Zc = X_c[2]

        if not np.isfinite(Zc):
            return False, np.nan, np.nan, Zc, "nan_Z"

        if Zc <= 1e-3:
            return False, np.nan, np.nan, Zc, "behind"

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = fx * X_c[0] / Zc + cx
        v = fy * X_c[1] / Zc + cy

        if (not np.isfinite(u)) or (not np.isfinite(v)):
            return False, u, v, Zc, "nan_uv"

        return True, u, v, Zc, "ok"

    def _solve_tau_debug(self, X_w, R0, T0, R1, T1, K, T_cb, W, H):
        dbg = {
            "mid_u": np.nan,
            "mid_v": np.nan,
            "mid_Z": np.nan,
            "mid_state": "unknown",
            "r0": np.nan,
            "r05": np.nan,
            "r1": np.nan,
            "last_tau": np.nan,
            "last_u": np.nan,
            "last_v": np.nan,
            "last_Z": np.nan,
            "last_r": np.nan,
            "iters": 0,
        }

        valid_probe_count = 0

        for tau_probe, key in [(0.0, "0"), (0.5, "05"), (1.0, "1")]:
            ok, u, v, Zc, reason = self._project_tau_full(
                X_w, R0, T0, R1, T1, tau_probe, K, T_cb
            )

            if ok:
                valid_probe_count += 1
                r = v / float(H - 1) - tau_probe
                dbg["r" + key] = r

                if key == "05":
                    dbg["mid_u"] = u
                    dbg["mid_v"] = v
                    dbg["mid_Z"] = Zc
                    dbg["mid_state"] = self._inside_label(
                        u, v, W, H, margin=self.debug_margin
                    )
            else:
                if key == "05":
                    dbg["mid_Z"] = Zc
                    dbg["mid_state"] = reason

        if valid_probe_count == 0:
            return False, None, "behind_or_invalid_all_tau", dbg

        if np.isfinite(dbg["mid_v"]):
            tau = dbg["mid_v"] / float(H - 1)
            tau = max(0.0, min(1.0, tau))
        else:
            tau = 0.5

        max_iter = 12
        eps = 1e-3
        dt = 1e-3

        for it in range(max_iter):
            ok, u, v, Zc, reason = self._project_tau_full(
                X_w, R0, T0, R1, T1, tau, K, T_cb
            )

            dbg["iters"] = it + 1
            dbg["last_tau"] = tau

            if not ok:
                return False, None, "project_fail_iter_" + reason, dbg

            r = v / float(H - 1) - tau

            dbg["last_u"] = u
            dbg["last_v"] = v
            dbg["last_Z"] = Zc
            dbg["last_r"] = r

            if abs(r) < eps:
                return True, (u, v, Zc, tau), "ok", dbg

            if tau <= 1.0 - dt:
                tau2 = tau + dt
            else:
                tau2 = tau - dt

            ok2, u2, v2, Zc2, reason2 = self._project_tau_full(
                X_w, R0, T0, R1, T1, tau2, K, T_cb
            )

            if not ok2:
                return False, None, "project_fail_fd_" + reason2, dbg

            r2 = v2 / float(H - 1) - tau2
            denom = tau2 - tau

            if abs(denom) < 1e-12:
                return False, None, "zero_fd_denom", dbg

            dr = (r2 - r) / denom

            if (not np.isfinite(dr)) or abs(dr) < 1e-8:
                return False, None, "small_derivative", dbg

            tau_new = tau - r / dr

            if not np.isfinite(tau_new):
                return False, None, "nan_tau_update", dbg

            tau = max(0.0, min(1.0, tau_new))

        r0 = dbg["r0"]
        r1 = dbg["r1"]

        if np.isfinite(r0) and np.isfinite(r1) and r0 * r1 > 0:
            return False, None, "no_root_in_readout", dbg

        return False, None, "newton_no_converge", dbg

    # ------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------

    def _compute_grid_coverage(self, proj_mask, cell=32, min_points=5):
        H, W = proj_mask.shape
        covered = 0
        total = 0

        for y0 in range(0, H, cell):
            for x0 in range(0, W, cell):
                y1 = min(y0 + cell, H)
                x1 = min(x0 + cell, W)

                patch = proj_mask[y0:y1, x0:x1]
                n = np.count_nonzero(patch)

                total += 1
                if n >= min_points:
                    covered += 1

        ratio = covered / float(max(total, 1))
        return ratio, covered, total

    def _compute_region_coverage(self, proj_mask, x0, x1, cell=32, min_points=5):
        H, W = proj_mask.shape
        covered = 0
        total = 0

        x0 = max(0, int(x0))
        x1 = min(W, int(x1))

        for y0 in range(0, H, cell):
            for xx0 in range(x0, x1, cell):
                y1 = min(y0 + cell, H)
                xx1 = min(xx0 + cell, x1)

                patch = proj_mask[y0:y1, xx0:xx1]
                n = np.count_nonzero(patch)

                total += 1
                if n >= min_points:
                    covered += 1

        ratio = covered / float(max(total, 1))
        return ratio, covered, total

    # ------------------------------------------------------------
    # Offline functions
    # ------------------------------------------------------------

    def _load_tf_from_bag(self, bag_path):
        rospy.loginfo("[OFFLINE] Loading TF from bag: %s", bag_path)

        n_tf = 0
        n_static = 0

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=["/tf_static"]):
                for tr in msg.transforms:
                    try:
                        self.tfbuf.set_transform_static(tr, "offline")
                        n_static += 1
                    except Exception as e:
                        rospy.logwarn_throttle(
                            2.0,
                            "[OFFLINE] set static TF failed: %s" % str(e)
                        )

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=["/tf"]):
                for tr in msg.transforms:
                    try:
                        self.tfbuf.set_transform(tr, "offline")
                        n_tf += 1
                    except Exception as e:
                        rospy.logwarn_throttle(
                            2.0,
                            "[OFFLINE] set dynamic TF failed: %s" % str(e)
                        )

        rospy.loginfo(
            "[OFFLINE] Loaded TF: dynamic=%d static=%d",
            n_tf,
            n_static
        )

    def _read_first_camera_info(self, bag_path, topic="/camera/color/camera_info"):
        with rosbag.Bag(bag_path, "r") as bag:
            for tp, msg, t in bag.read_messages(topics=[topic]):
                rospy.loginfo("[OFFLINE] Loaded CameraInfo from %s", topic)
                return msg

        raise RuntimeError("No CameraInfo found on topic %s" % topic)

    def run_offline(self,
                    bag_path,
                    color_topic="/camera/color/image_raw",
                    camera_info_topic="/camera/color/camera_info",
                    max_frames=0,
                    start_frame=1,
                    end_frame=0,
                    frame_step=1):

        self._load_tf_from_bag(bag_path)

        ci = self._read_first_camera_info(
            bag_path,
            topic=camera_info_topic
        )

        rospy.loginfo("[OFFLINE] Start processing fast color frames")

        n_seen = 0
        n_processed = 0

        frame_step = max(1, int(frame_step))

        rospy.logwarn(
            "[DEBUG_ARGS] start_frame=%d end_frame=%d frame_step=%d max_frames=%d",
            start_frame,
            end_frame,
            frame_step,
            max_frames
        )

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[color_topic]):

                if rospy.is_shutdown():
                    rospy.logwarn("[OFFLINE] shutdown requested, stop")
                    break

                n_seen += 1

                if n_seen < start_frame:
                    continue

                if end_frame > 0 and n_seen > end_frame:
                    break

                if ((n_seen - start_frame) % frame_step) != 0:
                    continue

                if max_frames > 0 and n_processed >= max_frames:
                    break

                rospy.loginfo(
                    "[OFFLINE] Processing fast frame %d stamp=%.6f",
                    n_seen,
                    msg.header.stamp.to_sec()
                )

                ok = self.process_fast_frame(msg, ci)

                if ok:
                    n_processed += 1

        rospy.loginfo(
            "[OFFLINE] Finished. seen=%d processed=%d start_frame=%d end_frame=%d frame_step=%d",
            n_seen,
            n_processed,
            start_frame,
            end_frame,
            frame_step
        )

    def cb(self, cmsg, ci):
        self.process_fast_frame(cmsg, ci)

    # ------------------------------------------------------------
    # Main frame processing
    # ------------------------------------------------------------

    def process_fast_frame(self, cmsg, ci):
        self.fast_cb_count += 1

        if self.process_every > 1:
            if (self.fast_cb_count % self.process_every) != 0:
                return False

        try:
            color = self.bridge.imgmsg_to_cv2(cmsg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[FAST] color decode failed: %s" % str(e))
            return False

        H, W = color.shape[:2]
        K = K_from_info(ci)

        rospy.loginfo(
            "[CAMERA K] fx=%.2f fy=%.2f cx=%.2f cy=%.2f image=%dx%d",
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], W, H
        )

        t_ref = cmsg.header.stamp
        t0 = t_ref - rospy.Duration.from_sec(0.5 * self.readout)
        t1 = t_ref + rospy.Duration.from_sec(0.5 * self.readout)

        try:
            T0 = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame,
                self.base_frame,
                t0,
                rospy.Duration(0.1)
            ))

            T1 = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame,
                self.base_frame,
                t1,
                rospy.Duration(0.1)
            ))

            T_ref = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame,
                self.base_frame,
                t_ref,
                rospy.Duration(0.1)
            ))

            T_cb = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.color_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            ))

        except Exception as e:
            rospy.logwarn_throttle(2.0, "[FAST] TF failed: %s" % str(e))
            return False

        R0, Tv0 = T0[:3, :3], T0[:3, 3]
        R1, Tv1 = T1[:3, :3], T1[:3, 3]

        yaw0 = self._yaw_from_R(R0)
        yaw1 = self._yaw_from_R(R1)
        dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))

        yaw_rate_rad = dyaw / max(self.readout, 1e-6)
        yaw_rate_deg = np.degrees(yaw_rate_rad)

        R_rel = R0.T.dot(R1)
        trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        total_rot_deg = np.degrees(np.arccos(trace))

        rospy.loginfo(
            "[YAW] yaw_rate=%+7.2f deg/s | total_rot=%.3f deg over readout=%.3fs",
            yaw_rate_deg,
            total_rot_deg,
            self.readout
        )

        T_target = T_ref[:3, 3]
        R_target = T_ref[:3, :3]

        pose_dists = np.array([
            self._pose_distance(r["R"], r["T"], R_target, T_target, w_yaw=0.5)
            for r in self.db
        ])

        best_k = int(np.argmin(pose_dists))

        best_R = self.db[best_k]["R"]
        best_T = self.db[best_k]["T"]

        best_d_xy = np.linalg.norm(best_T[:2] - T_target[:2])
        best_d_z = abs(best_T[2] - T_target[2])
        best_d_xyz = np.linalg.norm(best_T - T_target)
        best_d_yaw = self._yaw_err_deg(best_R, R_target)

        rospy.loginfo(
            "[POSE_MATCH_BEST] best_k=%d score=%.4f d_xy=%.4fm d_z=%.4fm d_xyz=%.4fm dyaw=%.2fdeg",
            best_k,
            float(pose_dists[best_k]),
            float(best_d_xy),
            float(best_d_z),
            float(best_d_xyz),
            float(best_d_yaw)
        )

        if self.slow_select_mode == "topk":
            K_neighbors = min(self.num_slow_neighbors, len(self.db))
            neighbor_idx = np.argsort(pose_dists)[:K_neighbors]

        elif self.slow_select_mode == "window":
            lo = max(0, best_k - self.slow_window)
            hi = min(len(self.db), best_k + self.slow_window + 1)

            neighbor_idx = np.arange(
                lo,
                hi,
                self.slow_window_step,
                dtype=np.int32
            )

            K_neighbors = len(neighbor_idx)

        else:
            rospy.logwarn(
                "[MULTI_SLOW_SELECT] unknown slow_select_mode=%s, fallback to topk",
                str(self.slow_select_mode)
            )

            K_neighbors = min(self.num_slow_neighbors, len(self.db))
            neighbor_idx = np.argsort(pose_dists)[:K_neighbors]

        rospy.loginfo(
            "[MULTI_SLOW_SELECT] mode=%s best_k=%d using=%d slow frames, dists=[%s]",
            str(self.slow_select_mode),
            best_k,
            K_neighbors,
            ", ".join("%.3f" % pose_dists[i] for i in neighbor_idx)
        )

        for debug_sid in neighbor_idx[:min(len(neighbor_idx), 10)]:
            debug_sid = int(debug_sid)
            rr = self.db[debug_sid]["R"]
            tt = self.db[debug_sid]["T"]

            d_xy = np.linalg.norm(tt[:2] - T_target[:2])
            d_z = abs(tt[2] - T_target[2])
            d_xyz = np.linalg.norm(tt - T_target)
            d_yaw = self._yaw_err_deg(rr, R_target)

            rospy.loginfo(
                "[NEIGHBOR_POSE] id=%d score=%.4f d_xy=%.4fm d_z=%.4fm d_xyz=%.4fm dyaw=%.2fdeg",
                debug_sid,
                float(pose_dists[debug_sid]),
                float(d_xy),
                float(d_z),
                float(d_xyz),
                float(d_yaw)
            )

        # ============================================================
        # Build world points from selected slow raw-depth frames
        # ============================================================

        all_pts_world = []
        all_slow_ids = []

        for slow_id in neighbor_idx:
            slow_id = int(slow_id)
            slow = self.db[slow_id]

            Z = slow["depth"]
            K_d = slow["K_depth"]
            T_bd = slow["T_bd"]

            R_s = slow["R"]
            T_s = slow["T"]

            Hs, Ws = Z.shape

            fx_d, fy_d = K_d[0, 0], K_d[1, 1]
            cx_d, cy_d = K_d[0, 2], K_d[1, 2]

            T_wb_s = make_T(R_s, T_s)
            T_wd_s = T_wb_s.dot(T_bd)

            uu_s, vv_s = np.meshgrid(
                np.arange(0, Ws, self.stride),
                np.arange(0, Hs, self.stride)
            )

            uu_s = uu_s.flatten()
            vv_s = vv_s.flatten()

            zs_s = Z[vv_s, uu_s]

            ok_s = (
                np.isfinite(zs_s) &
                (zs_s > self.min_depth) &
                (zs_s < self.max_depth)
            )

            n_sampled = len(zs_s)
            n_valid = int(ok_s.sum())

            rospy.loginfo(
                "[MULTI_SLOW_SAMPLE] slow_id=%d dist=%.3f sampled=%d valid=%d (%.1f%%)",
                slow_id,
                float(pose_dists[slow_id]),
                n_sampled,
                n_valid,
                100.0 * n_valid / max(n_sampled, 1)
            )

            if n_valid == 0:
                continue

            uu_s = uu_s[ok_s]
            vv_s = vv_s[ok_s]
            zs_s = zs_s[ok_s]

            X_d = (uu_s - cx_d) * zs_s / fx_d
            Y_d = (vv_s - cy_d) * zs_s / fy_d

            pts_depth = np.stack(
                [X_d, Y_d, zs_s, np.ones_like(zs_s)],
                axis=0
            )

            pts_w = T_wd_s.dot(pts_depth)[:3, :]

            all_pts_world.append(pts_w)
            all_slow_ids.append(
                np.full(pts_w.shape[1], slow_id, dtype=np.int32)
            )

        if not all_pts_world:
            rospy.logwarn("[MULTI_SLOW] no valid world points from selected slow frames")
            return False

        pts_world = np.concatenate(all_pts_world, axis=1)
        slow_ids = np.concatenate(all_slow_ids)

        rospy.loginfo(
            "[MULTI_SLOW] total world points = %d",
            pts_world.shape[1]
        )

        # ============================================================
        # Project world points into fast image
        # ============================================================

        sparse = np.zeros((H, W), dtype=np.float32)

        smear_vis = color.copy()
        proj_all_vis = np.zeros((H, W), dtype=np.uint8)
        drawn_vis = np.zeros((H, W), dtype=np.uint8)

        e_frac = 0.5 * self.exposure / max(self.readout, 1e-6)

        n_total = pts_world.shape[1]
        n_solve_fail = 0
        n_outside = 0
        n_behind = 0
        n_conv = 0
        n_smear_endfail = 0
        n_smear_end_outside = 0
        n_drawn = 0

        fail_reasons = {}
        final_outside_reasons = {}
        end_outside_reasons = {}
        bad_examples = []

        tau_ok_list = []
        u_ok_list = []
        v_ok_list = []
        z_ok_list = []

        smear_lengths = []
        contrib = {}

        def add_bad_example(reason, idx, sid, dbg):
            if len(bad_examples) >= self.debug_max_examples:
                return

            bad_examples.append({
                "reason": reason,
                "idx": idx,
                "sid": sid,
                "mid_state": dbg.get("mid_state", "unknown"),
                "mid_u": dbg.get("mid_u", np.nan),
                "mid_v": dbg.get("mid_v", np.nan),
                "mid_Z": dbg.get("mid_Z", np.nan),
                "r0": dbg.get("r0", np.nan),
                "r05": dbg.get("r05", np.nan),
                "r1": dbg.get("r1", np.nan),
                "last_tau": dbg.get("last_tau", np.nan),
                "last_u": dbg.get("last_u", np.nan),
                "last_v": dbg.get("last_v", np.nan),
                "last_Z": dbg.get("last_Z", np.nan),
                "last_r": dbg.get("last_r", np.nan),
                "iters": dbg.get("iters", 0),
            })

        for i in range(n_total):
            if i > 0 and (i % 5000) == 0:
                rospy.loginfo(
                    "[PROJECT_PROGRESS] i=%d/%d conv=%d solve_fail=%d outside=%d drawn=%d",
                    i,
                    n_total,
                    n_conv,
                    n_solve_fail,
                    n_outside,
                    n_drawn
                )

            X_w = pts_world[:, i]
            sid = int(slow_ids[i])

            ok_solve, res, reason, dbg = self._solve_tau_debug(
                X_w, R0, Tv0, R1, Tv1, K, T_cb, W, H
            )

            if not ok_solve:
                n_solve_fail += 1
                self._inc(fail_reasons, reason)
                add_bad_example(reason, i, sid, dbg)
                continue

            u, v, Zc, tau = res

            tau_ok_list.append(tau)
            u_ok_list.append(u)
            v_ok_list.append(v)
            z_ok_list.append(Zc)

            if not (0 <= u < W and 0 <= v < H):
                n_outside += 1
                label = self._inside_label(u, v, W, H, margin=self.debug_margin)
                self._inc(final_outside_reasons, label)
                continue

            if Zc <= 0:
                n_behind += 1
                continue

            n_conv += 1
            contrib[sid] = contrib.get(sid, 0) + 1

            ui = int(np.floor(u))
            vi = int(np.floor(v))

            proj_all_vis[vi, ui] = 255

            cur = sparse[vi, ui]
            if cur == 0 or Zc < cur:
                sparse[vi, ui] = Zc

            if self.smear_stride > 1:
                if (i % self.smear_stride) != 0:
                    continue

            tau_a = max(0.0, tau - e_frac)
            tau_b = min(1.0, tau + e_frac)

            pa = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_a, K, T_cb)
            pb = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_b, K, T_cb)

            if pa is None or pb is None:
                n_smear_endfail += 1
                if pa is None:
                    self._inc(end_outside_reasons, "pa_none")
                if pb is None:
                    self._inc(end_outside_reasons, "pb_none")
                continue

            pa_inside = (0 <= pa[0] < W and 0 <= pa[1] < H)
            pb_inside = (0 <= pb[0] < W and 0 <= pb[1] < H)

            if not (pa_inside and pb_inside):
                n_smear_end_outside += 1

                if not pa_inside:
                    self._inc(
                        end_outside_reasons,
                        "pa_" + self._inside_label(pa[0], pa[1], W, H, margin=self.debug_margin)
                    )

                if not pb_inside:
                    self._inc(
                        end_outside_reasons,
                        "pb_" + self._inside_label(pb[0], pb[1], W, H, margin=self.debug_margin)
                    )

                continue

            n_drawn += 1
            drawn_vis[vi, ui] = 255

            smear_length = np.hypot(pb[0] - pa[0], pb[1] - pa[1])
            smear_lengths.append(smear_length)

            cv2.arrowedLine(
                smear_vis,
                (int(round(pa[0])), int(round(pa[1]))),
                (int(round(pb[0])), int(round(pb[1]))),
                (0, 255, 255),
                1,
                tipLength=0.3
            )

        # ============================================================
        # Logs
        # ============================================================

        for sid in sorted(contrib.keys()):
            rospy.loginfo(
                "[MULTI_SLOW_CONTRIB] slow_id=%d projected=%d dist=%.3f",
                sid,
                contrib[sid],
                float(pose_dists[sid])
            )

        grid_ratio, covered_cells, total_cells = self._compute_grid_coverage(
            proj_all_vis,
            cell=self.coverage_cell,
            min_points=self.coverage_min_points
        )

        rospy.loginfo(
            "[COVERAGE_GRID] covered_cells=%d/%d ratio=%.1f%% cell=%d min_points=%d",
            covered_cells,
            total_cells,
            100.0 * grid_ratio,
            self.coverage_cell,
            self.coverage_min_points
        )

        W3 = W // 3

        left_cov, left_c, left_t = self._compute_region_coverage(
            proj_all_vis, 0, W3,
            cell=self.coverage_cell,
            min_points=self.coverage_min_points
        )

        mid_cov, mid_c, mid_t = self._compute_region_coverage(
            proj_all_vis, W3, 2 * W3,
            cell=self.coverage_cell,
            min_points=self.coverage_min_points
        )

        right_cov, right_c, right_t = self._compute_region_coverage(
            proj_all_vis, 2 * W3, W,
            cell=self.coverage_cell,
            min_points=self.coverage_min_points
        )

        rospy.loginfo(
            "[COVERAGE_REGION] left=%.1f%% (%d/%d) mid=%.1f%% (%d/%d) right=%.1f%% (%d/%d)",
            100.0 * left_cov, left_c, left_t,
            100.0 * mid_cov, mid_c, mid_t,
            100.0 * right_cov, right_c, right_t
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cover_mask = cv2.dilate(proj_all_vis, kernel, iterations=1)

        pixel_coverage = np.count_nonzero(cover_mask) / float(H * W)

        rospy.loginfo(
            "[COVERAGE_PIXEL_DILATED] ratio=%.1f%%",
            100.0 * pixel_coverage
        )

        rospy.loginfo(
            "[PROJECT_DEBUG_COUNTS] total=%d solve_ok=%d solve_fail=%d final_inside=%d final_outside=%d drawn=%d",
            n_total,
            n_conv + n_outside,
            n_solve_fail,
            n_conv,
            n_outside,
            n_drawn
        )

        rospy.loginfo(
            "[PROJECT_FAIL_REASONS] %s",
            self._fmt_counts(fail_reasons)
        )

        rospy.loginfo(
            "[PROJECT_FINAL_OUTSIDE_REASONS] %s",
            self._fmt_counts(final_outside_reasons)
        )

        rospy.loginfo(
            "[SMEAR_ENDPOINT_REASONS] %s",
            self._fmt_counts(end_outside_reasons)
        )

        if tau_ok_list:
            tau_arr = np.array(tau_ok_list)
            u_arr = np.array(u_ok_list)
            v_arr = np.array(v_ok_list)
            z_arr = np.array(z_ok_list)

            rospy.loginfo(
                "[SOLVE_OK_STATS] tau[min=%.3f p50=%.3f max=%.3f] "
                "u[min=%.1f max=%.1f] v[min=%.1f max=%.1f] "
                "Z[min=%.3f p50=%.3f max=%.3f]",
                float(np.min(tau_arr)),
                float(np.percentile(tau_arr, 50)),
                float(np.max(tau_arr)),
                float(np.min(u_arr)),
                float(np.max(u_arr)),
                float(np.min(v_arr)),
                float(np.max(v_arr)),
                float(np.min(z_arr)),
                float(np.percentile(z_arr, 50)),
                float(np.max(z_arr))
            )
        else:
            rospy.logwarn("[SOLVE_OK_STATS] no successful tau solutions")

        for ex in bad_examples:
            rospy.loginfo(
                "[BAD_EXAMPLE] reason=%s idx=%d slow_id=%d mid_state=%s "
                "mid=(u=%.1f v=%.1f Z=%.3f) "
                "r0=%.3f r05=%.3f r1=%.3f "
                "last=(tau=%.3f u=%.1f v=%.1f Z=%.3f r=%.3f iters=%d)",
                ex["reason"],
                ex["idx"],
                ex["sid"],
                ex["mid_state"],
                ex["mid_u"],
                ex["mid_v"],
                ex["mid_Z"],
                ex["r0"],
                ex["r05"],
                ex["r1"],
                ex["last_tau"],
                ex["last_u"],
                ex["last_v"],
                ex["last_Z"],
                ex["last_r"],
                ex["iters"]
            )

        rospy.loginfo(
            "[SMEAR DIAG DEBUG] total=%d conv=%d (%.1f%%) drawn=%d (%.1f%%) | "
            "solve_fail=%d outside=%d behind=%d end_fail=%d end_outside=%d",
            n_total,
            n_conv,
            100.0 * n_conv / max(n_total, 1),
            n_drawn,
            100.0 * n_drawn / max(n_total, 1),
            n_solve_fail,
            n_outside,
            n_behind,
            n_smear_endfail,
            n_smear_end_outside
        )

        rospy.loginfo(
            "[CORE METRIC] conv%%=%.1f%% solve_fail%%=%.1f%% outside%%=%.1f%% grid_cov=%.1f%%",
            100.0 * n_conv / max(n_total, 1),
            100.0 * n_solve_fail / max(n_total, 1),
            100.0 * n_outside / max(n_total, 1),
            100.0 * grid_ratio
        )

        if smear_lengths:
            arr = np.array(smear_lengths)
            rospy.loginfo(
                "[SMEAR LEN] mean=%.2f median=%.2f p95=%.2f max=%.2f px",
                float(np.mean(arr)),
                float(np.median(arr)),
                float(np.percentile(arr, 95)),
                float(np.max(arr))
            )

        # ============================================================
        # Save diagnostic images
        # ============================================================

        self.fast_done_count += 1

        if self.save_every > 0 and (self.fast_done_count % self.save_every) == 0:
            cv2.imwrite("/proj_all_debug_%06d.png" % self.fast_done_count, proj_all_vis)
            cv2.imwrite("/drawn_points_debug_%06d.png" % self.fast_done_count, drawn_vis)

            compare_vis = np.zeros((H, W, 3), dtype=np.uint8)
            compare_vis[proj_all_vis > 0] = (0, 0, 255)
            compare_vis[drawn_vis > 0] = (0, 255, 0)
            cv2.imwrite("/proj_vs_drawn_debug_%06d.png" % self.fast_done_count, compare_vis)

            coverage_vis = color.copy()
            coverage_vis[cover_mask > 0] = (
                0.5 * coverage_vis[cover_mask > 0] +
                0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)
            cv2.imwrite("/coverage_debug_%06d.png" % self.fast_done_count, coverage_vis)

        # ============================================================
        # Publish outputs
        # ============================================================

        self.pub_color.publish(self.bridge.cv2_to_imgmsg(color, "bgr8"))

        m = sparse.copy()
        m[m <= 0] = np.nan

        try:
            lo = np.nanpercentile(m, 5)
            hi = np.nanpercentile(m, 95)
        except Exception:
            lo, hi = 0.3, 5.0

        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = 0.3, 5.0

        norm = (np.clip(sparse, lo, hi) - lo) / (hi - lo + 1e-9)
        u8 = (norm * 255).astype(np.uint8)
        u8[sparse <= 0] = 0

        cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        cmap[sparse <= 0] = 0

        self.pub_depth_vis.publish(self.bridge.cv2_to_imgmsg(cmap, "bgr8"))
        self.pub_smear.publish(self.bridge.cv2_to_imgmsg(smear_vis, "bgr8"))

        coverage_vis_pub = color.copy()
        coverage_vis_pub[cover_mask > 0] = (
            0.5 * coverage_vis_pub[cover_mask > 0] +
            0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        self.pub_coverage.publish(self.bridge.cv2_to_imgmsg(coverage_vis_pub, "bgr8"))

        rospy.loginfo(
            "[FAST_DONE_DEBUG] cb=%d done=%d neighbors=%d best_k=%d best_dist=%.3f grid_cov=%.1f%%",
            self.fast_cb_count,
            self.fast_done_count,
            K_neighbors,
            best_k,
            float(pose_dists[best_k]),
            100.0 * grid_ratio
        )

        self.yaw_rates.append(yaw_rate_deg)

        if len(self.yaw_rates) % 50 == 0:
            arr = np.array(self.yaw_rates)
            rospy.loginfo(
                "[YAW SUMMARY] frames=%d mean=%+.2f std=%.2f p95_abs=%.2f max_abs=%.2f deg/s",
                len(arr),
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.percentile(np.abs(arr), 95)),
                float(np.max(np.abs(arr)))
            )

        return True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", required=True, choices=["build", "project"])
    ap.add_argument("--out", default="slow_raw.pkl")
    ap.add_argument("--db", default="slow_raw.pkl")

    ap.add_argument("--fast_bag", default=None)
    ap.add_argument("--color_topic", default="/camera/color/image_raw")
    ap.add_argument("--camera_info_topic", default="/camera/color/camera_info")

    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--start_frame", type=int, default=1)
    ap.add_argument("--end_frame", type=int, default=0)
    ap.add_argument("--frame_step", type=int, default=1)

    args, _ = ap.parse_known_args()

    if args.mode == "build":
        rospy.init_node("slow_raw_builder_debug", anonymous=True)
        SlowBuilder(args.out)
        rospy.spin()

    else:
        rospy.init_node("slow_raw_to_fast_projector_debug", anonymous=True)

        if args.fast_bag is None:
            projector = Projector(args.db, subscribe=True)
            rospy.spin()

        else:
            projector = Projector(args.db, subscribe=False)

            try:
                projector.run_offline(
                    args.fast_bag,
                    color_topic=args.color_topic,
                    camera_info_topic=args.camera_info_topic,
                    max_frames=args.max_frames,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    frame_step=args.frame_step
                )
            except KeyboardInterrupt:
                rospy.logwarn("[MAIN] KeyboardInterrupt, exiting")