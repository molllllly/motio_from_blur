#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
slow_to_fast_raw_direct.py

Core idea:
    Use raw slow depth directly.

Pipeline:
    slow raw depth pixel
    -> back-project to slow depth camera 3D
    -> transform to world
    -> project to fast rolling-shutter RGB image

No RGB aligned depth.
No RGB-to-depth reverse lookup.
"""

from __future__ import print_function

import argparse
import pickle
import signal
import sys

import numpy as np
import cv2
import rospy
import rosbag
import message_filters

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import tf2_ros
import tf.transformations as tft
from util_smear.pfm import writePFM

from export_analysis import load_gt_csv, GTInterpolator
import os

# ============================================================
# Basic helper functions
# ============================================================

def tfmsg_to_T(msg):
    """
    Convert TransformStamped to 4x4 matrix.

    Meaning:
        target <- source

    If lookup_transform(A, B, t), then returned transform is:
        A <- B
    """
    tr = msg.transform.translation
    q = msg.transform.rotation

    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float64)
    T[0, 3] = tr.x
    T[1, 3] = tr.y
    T[2, 3] = tr.z
    return T


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def K_from_info(info):
    return np.array(info.K, dtype=np.float64).reshape(3, 3)


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


def lerp_t(t0, t1, tau):
    return t0 + (t1 - t0) * float(tau)


# ============================================================
# STEP 1: build slow raw-depth database
# ============================================================

class SlowRawBuilder(object):
    """
    Save raw slow depth frames.

    Each record contains:
        t        : slow depth timestamp
        R, T     : world <- base
        depth    : raw slow depth in depth image plane, meters
        K_depth  : depth camera intrinsics
        T_bd     : base <- depth optical frame
    """

    def __init__(self, out_path):
        self.out_path = out_path
        self.bridge = CvBridge()
        self.records = []

        self.world_frame = rospy.get_param("~world_frame", "mocap")
        self.base_frame = rospy.get_param("~base_frame", "hamster")
        self.depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")

        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 40.0))

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        self.grad_ksize = int(rospy.get_param("~grad_ksize", 3))
        self.grad_percentile = float(rospy.get_param("~grad_percentile", 98.0))
        self._grad_cache = {}

        sub_d = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        sub_di = message_filters.Subscriber("/camera/depth/camera_info", CameraInfo)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_d, sub_di],
            queue_size=20,
            slop=0.03
        )

        self.sync.registerCallback(self.cb)

        self.pub_depth_valid = rospy.Publisher(
            "/slow_debug/raw_depth_valid", Image, queue_size=1
        )
        self.T_bd_fixed = None   # 固定外参,首帧计算一次
        signal.signal(signal.SIGINT, self.save_and_exit)


    def save_and_exit(self, *args):
        with open(self.out_path, "wb") as f:
            pickle.dump(self.records, f, protocol=2)

        rospy.loginfo("Saved %d slow raw-depth frames to %s",
                      len(self.records), self.out_path)
        sys.exit(0)

    def _compute_fixed_T_bd(self):
        """
        计算固定外参 T_bd (hamster <- depth),只走静态链,只算一次。

        拆成两段:
          T_bd = T(hamster <- base_link) x T(base_link <- depth)

        第二段 base_link <- depth 是 URDF 静态外参,真常数。
        第一段 hamster <- base_link 假设 mocap 刚体原点与 base_link
        近似重合(初始化时已对齐),取单位阵。
        如果你有实测的 hamster<-base_link 偏移,把 T_hb 换成实测值。
        """
        # 纯静态段:base_link <- camera_depth_optical_frame
        T_base_depth = tfmsg_to_T(self.tfbuf.lookup_transform(
            "base_link",
            self.depth_frame,
            rospy.Time(0),
            rospy.Duration(5.0)
        ))

        # 固定段:hamster <- base_link(近似单位阵;有实测值就替换)
        T_hb = np.eye(4, dtype=np.float64)

        return T_hb.dot(T_base_depth)

    def cb(self, dmsg, di):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                dmsg, desired_encoding="passthrough"
            )
        except Exception as e:
            rospy.logwarn("[SLOW] depth decode failed: %s", str(e))
            return

        idx = len(self.records)

        #cv2.imwrite("raw_depth__%06d.png" % idx, depth)
        
        if dmsg.encoding == "16UC1":
            z = depth.astype(np.float32) * 0.001
        elif dmsg.encoding == "32FC1":
            z = depth.astype(np.float32)
        else:
            rospy.logwarn("[SLOW] unsupported depth encoding: %s", dmsg.encoding)
            return

        K_d = K_from_info(di)

        try:
            # world <- base at slow depth timestamp
            T_wb = tfmsg_to_T(
                self.tfbuf.lookup_transform(
                    self.world_frame,
                    self.base_frame,
                    dmsg.header.stamp,
                    rospy.Duration(0.1)
                )
            )

            # base <- depth optical frame
            # T_bd = tfmsg_to_T(
            #     self.tfbuf.lookup_transform(
            #         self.base_frame,
            #         self.depth_frame,
            #         rospy.Time(0),
            #         rospy.Duration(0.1)
            #     )
            # )
            if self.T_bd_fixed is None:
                self.T_bd_fixed = self._compute_fixed_T_bd()
                rospy.loginfo("[SLOW] fixed T_bd computed once:\n%s",
                              str(self.T_bd_fixed))

            T_bd = self.T_bd_fixed
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[SLOW] TF failed: %s", str(e))
            return


        valid = (
            np.isfinite(z)

        )

        valid_ratio = 100.0 * valid.sum() / max(valid.size, 1)
        #
        # z_valid = np.where(valid, z, 0.0).astype(np.float32)
        # msg = self.bridge.cv2_to_imgmsg(z_valid, encoding="32FC1")
        # msg.header.stamp = dmsg.header.stamp
        # msg.header.frame_id = dmsg.header.frame_id
        # self.pub_depth_valid.publish(msg)
        #
        # z_valid_mm = (z_valid * 1000.0).astype(np.uint16)
        #
        # cv2.imwrite("depth_valid_%06d.png" % idx, z_valid_mm)

        base_valid = np.isfinite(z) & (z > 0)
        Zf = np.where(base_valid, z, 0.0).astype(np.float32)

        k = self.grad_ksize if self.grad_ksize in [1, 3, 5, 7] else 3
        gx = cv2.Sobel(Zf, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(Zf, cv2.CV_32F, 0, 1, ksize=k)
        G = cv2.magnitude(gx, gy)

        vals = G[base_valid]
        if vals.size > 0:
            Tgrad = float(np.percentile(vals, self.grad_percentile))
            grad_bad = (G > Tgrad) & base_valid
        else:
            grad_bad = np.zeros_like(base_valid, dtype=bool)
        z[grad_bad] = 0.0

        self.records.append({
            "t": dmsg.header.stamp.to_sec(),

            # world <- base
            "R": T_wb[:3, :3].copy(),
            "T": T_wb[:3, 3].copy(),

            # raw depth in depth image plane
            "depth": z.copy().astype(np.float32),

            # depth intrinsics
            "K_depth": K_d.copy(),

            # base <- depth
            "T_bd": T_bd.copy(),
        })

        # rospy.loginfo(
        #     "[SLOW RAW] stored=%d valid_depth=%.1f%%",
        #     len(self.records),
        #     valid_ratio
        # )


# ============================================================
# STEP 2: project slow raw depth to fast RGB
# ============================================================

class RawDepthProjector(object):

    def __init__(self, db_path, subscribe=True):
        self.bridge = CvBridge()

        with open(db_path, "rb") as f:
            self.db = pickle.load(f)

        rospy.loginfo("Loaded %d slow raw-depth frames", len(self.db))

        self.world_frame = rospy.get_param("~world_frame", "mocap")
        self.base_frame = rospy.get_param("~base_frame", "hamster")
        self.color_frame = rospy.get_param("~color_frame", "camera_color_optical_frame")

        self.readout = float(rospy.get_param("~readout_sec", 0.030))
        self.exposure = float(rospy.get_param("~exposure_sec", 0.0166))

        self.stride = int(rospy.get_param("~stride", 1))
        self.num_slow_neighbors = int(rospy.get_param("~num_slow_neighbors", 1))

        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 15.0))

        self.save_every = int(rospy.get_param("~save_every", 5))
        self.smear_stride = int(rospy.get_param("~smear_stride", 1))
        self.T_cb_fixed = None
        self.fast_cb_count = 0
        self.fast_done_count = 0

        self.pub_color = rospy.Publisher("/blur_color", Image, queue_size=1)
        self.pub_depth_vis = rospy.Publisher("/sparse_depth_vis", Image, queue_size=1)
        self.pub_smear = rospy.Publisher("/smear_path_vis", Image, queue_size=1)
        self.pub_proj = rospy.Publisher("/projected_points_vis", Image, queue_size=1)


        self.pose_out_dir = rospy.get_param("~pose_out_dir", "./out/poses/")  # 新增



        self.label_out_dir = rospy.get_param("~label_out_dir", "./out/double_ang_reps/")
        self.image_out_dir = rospy.get_param("~image_out_dir", "./out/images/")


        for d in (self.label_out_dir, self.image_out_dir, self.pose_out_dir):
            if not os.path.isdir(d):
                os.makedirs(d)  # 兼容 py2, 不用 exist_ok

        # GT 插值器: 给了 csv 才建, 没给就只存 tf 位姿
        gt_csv = rospy.get_param("~gt_csv", "fast_positions.csv")
        if gt_csv:
            self.itp = GTInterpolator(*load_gt_csv(gt_csv), use_bspline_pos=True)
            rospy.loginfo("[INIT] GT interpolator loaded from %s", gt_csv)
        else:
            self.itp = None
            rospy.logwarn("[INIT] no ~gt_csv given, will store tf pose only")

        self.frame_counter = 0

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(3600.0))


        if subscribe:
            self.tfl = tf2_ros.TransformListener(self.tfbuf)

            sub_c = message_filters.Subscriber("/camera/color/image_raw", Image)
            sub_ci = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)

            self.sync = message_filters.ApproximateTimeSynchronizer(
                [sub_c, sub_ci],
                queue_size=100,
                slop=0.03
            )
            self.sync.registerCallback(self.cb)
        else:
            self.tfl = None

    # ------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------

    def yaw_from_R(self, R):
        return np.arctan2(R[1, 0], R[0, 0])


    def _compute_fixed_T_cb(self):
        """
        固定外参 T_cb (color <- hamster),只算一次。
        T_cb = T(color <- base_link) x T(base_link <- hamster)
        """
        # 纯静态段:color <- base_link
        T_color_base = tfmsg_to_T(self.tfbuf.lookup_transform(
            self.color_frame,
            "base_link",
            rospy.Time(0),
            rospy.Duration(5.0)
        ))

        # 固定段:base_link <- hamster(builder 里 T_hb 的逆;单位阵的逆还是单位阵)
        T_bh = np.eye(4, dtype=np.float64)

        return T_color_base.dot(T_bh)
    
    def pose_distance(self, slow_R, slow_T, ref_R, ref_T, w_yaw=0.5):
        d_xyz = np.linalg.norm(slow_T - ref_T)

        yaw_s = self.yaw_from_R(slow_R)
        yaw_f = self.yaw_from_R(ref_R)

        dyaw = np.arctan2(
            np.sin(yaw_s - yaw_f),
            np.cos(yaw_s - yaw_f)
        )

        return d_xyz + w_yaw * abs(dyaw)

    # ------------------------------------------------------------
    # Projection functions
    # ------------------------------------------------------------

    def _project_tau(self, X_w, R0, T0, R1, T1, tau, K, T_cb):
        """
        Project world point into fast RGB image at row-time tau.

        R0,T0:
            world <- base at readout start

        R1,T1:
            world <- base at readout end

        T_cb:
            color <- base
        """
        R_b = slerp_R(R0, R1, tau)
        T_b = lerp_t(T0, T1, tau)

        # base <- world
        R_bw = R_b.T
        T_bw = -R_bw.dot(T_b)

        T_bw_4 = np.eye(4, dtype=np.float64)
        T_bw_4[:3, :3] = R_bw
        T_bw_4[:3, 3] = T_bw

        # color <- world
        T_cw = T_cb.dot(T_bw_4)

        X_c = T_cw[:3, :3].dot(X_w) + T_cw[:3, 3]
        Zc = X_c[2]

        if Zc <= 1e-6:
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = fx * X_c[0] / Zc + cx
        v = fy * X_c[1] / Zc + cy

        return u, v, Zc

    def solve_tau(self, X_w, R0, T0, R1, T1, K, T_cb, H):
        """
        Solve for tau with Newton-Raphson.

        We want the projected row to match the row-time used:

            r(tau) = v(tau)/(H - 1) - tau = 0

        This is needed because:
            tau determines the pose,
            but the projected row v also depends on that pose.
        """
        tau = 0.0
        u = v = Zc = 0.0
        for _ in range(8): #todo increase value
            p = self._project_tau(X_w, R0, T0, R1, T1, tau, K, T_cb)
            if p is None:
                return None
            u, v, Zc = p

            # Residual:
            # if tau is correct, normalized row index should equal tau.
            r = v / float(H - 1) - tau
            if abs(r) < 1e-3:
                return u, v, Zc, tau

            # Finite-difference estimate of r'(tau):
            #   r'(tau) ≈ ( r(tau + dt) - r(tau) ) / dt
            # This avoids deriving the full analytic derivative.
            dt = 1e-3
            tau2 = min(1.0, tau + dt)
            p2 = self._project_tau(X_w, R0, T0, R1, T1, tau2, K, T_cb)
            if p2 is None:
                return None
            _, v2, _ = p2
            #Approximate derivative
            dr = ((v2 / float(H - 1) - tau2) - r) / dt
            if abs(dr) < 1e-8:
                return None

            # Newton update:
            #   tau <- tau - r / r'
            # Clamp to [0, 1] because tau is a normalized row-time.
            tau = max(0.0, min(1.0, tau - r / dr))
        return None

    # ------------------------------------------------------------
    # Offline functions
    # ------------------------------------------------------------

    def load_tf_from_bag(self, bag_path):
        rospy.loginfo("[OFFLINE] Loading TF from bag: %s", bag_path)

        n_static = 0
        n_dynamic = 0

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=["/tf_static"]):
                for tr in msg.transforms:
                    try:
                        self.tfbuf.set_transform_static(tr, "offline")
                        n_static += 1
                    except Exception as e:
                        rospy.logwarn("[OFFLINE] static TF failed: %s", str(e))

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=["/tf"]):
                for tr in msg.transforms:
                    try:
                        self.tfbuf.set_transform(tr, "offline")
                        n_dynamic += 1
                    except Exception as e:
                        rospy.logwarn("[OFFLINE] dynamic TF failed: %s", str(e))

        rospy.loginfo(
            "[OFFLINE] Loaded TF: dynamic=%d static=%d",
            n_dynamic,
            n_static
        )

    def read_first_camera_info(self, bag_path, topic="/camera/color/camera_info"):
        with rosbag.Bag(bag_path, "r") as bag:
            for tp, msg, t in bag.read_messages(topics=[topic]):
                return msg

        raise RuntimeError("No CameraInfo found on topic %s" % topic)

    def run_offline(self, bag_path,
                    color_topic="/camera/color/image_raw",
                    camera_info_topic="/camera/color/camera_info",
                    start_frame=1,
                    end_frame=0,
                    frame_step=1,
                    max_frames=0):

        self.load_tf_from_bag(bag_path)
        ci = self.read_first_camera_info(bag_path, camera_info_topic)

        n_seen = 0
        n_processed = 0
        frame_step = max(1, int(frame_step))

        rospy.loginfo("[OFFLINE] Start processing fast bag")

        with rosbag.Bag(bag_path, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[color_topic]):
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
            "[OFFLINE] Finished. seen=%d processed=%d",
            n_seen,
            n_processed
        )

    # ------------------------------------------------------------
    # Online callback
    # ------------------------------------------------------------

    def cb(self, cmsg, ci):
        self.process_fast_frame(cmsg, ci)

    # ------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------

    def process_fast_frame(self, cmsg, ci):
        """
        生成一帧的 smear 标注。

        返回 True 表示成功写出了一对 (png, pfm)。
        """
        self.fast_cb_count += 1

        # --------------------------------------------------------
        # 解码 RGB
        # --------------------------------------------------------
        try:
            color = self.bridge.imgmsg_to_cv2(cmsg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("[FAST] color decode failed: %s", str(e))
            return False

        H, W = color.shape[:2]
        K_color = K_from_info(ci)

        # --------------------------------------------------------
        # 取曝光起止 / 参考时刻的位姿
        # --------------------------------------------------------
        t_ref = cmsg.header.stamp
        t0 = t_ref - rospy.Duration.from_sec(0.5 * self.readout)
        t1 = t_ref + rospy.Duration.from_sec(0.5 * self.readout)

        try:
            T0 = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t0, rospy.Duration(0.1)))
            T1 = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t1, rospy.Duration(0.1)))
            T_ref = tfmsg_to_T(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_ref, rospy.Duration(0.1)))

            if self.T_cb_fixed is None:
                self.T_cb_fixed = self._compute_fixed_T_cb()
                rospy.loginfo("[FAST] fixed T_cb computed once:\n%s",
                              str(self.T_cb_fixed))
            T_cb = self.T_cb_fixed

        except Exception as e:
            rospy.logwarn("[FAST] TF failed: %s", str(e))
            return False

        R0, Tv0 = T0[:3, :3], T0[:3, 3]
        R1, Tv1 = T1[:3, :3], T1[:3, 3]

        # --------------------------------------------------------
        # 按位姿挑最近的慢速深度帧
        # --------------------------------------------------------
        R_target = T_ref[:3, :3]
        T_target = T_ref[:3, 3]

        pose_dists = np.array([
            self.pose_distance(r["R"], r["T"], R_target, T_target, w_yaw=0.5)
            for r in self.db
        ])

        K_neighbors = min(self.num_slow_neighbors, len(self.db))
        neighbor_idx = np.argsort(pose_dists)[:K_neighbors]
        best_k = int(neighbor_idx[0])

        # --------------------------------------------------------
        # 从原始深度反投影出世界点
        # --------------------------------------------------------
        all_pts_world = []

        for sid in neighbor_idx:
            sid = int(sid)
            slow = self.db[sid]

            Z = slow["depth"]
            K_d = slow["K_depth"]
            T_bd = slow["T_bd"]
            R_s = slow["R"]
            T_s = slow["T"]

            Hs, Ws = Z.shape
            fx_d, fy_d = K_d[0, 0], K_d[1, 1]
            cx_d, cy_d = K_d[0, 2], K_d[1, 2]

            uu, vv = np.meshgrid(
                np.arange(0, Ws, self.stride),
                np.arange(0, Hs, self.stride)
            )
            uu = uu.flatten()
            vv = vv.flatten()
            zs = Z[vv, uu]

            valid = (
                    np.isfinite(zs) &
                    (zs > self.min_depth) &
                    (zs < self.max_depth)
            )

            uu, vv, zs = uu[valid], vv[valid], zs[valid]

            X_d = (uu - cx_d) * zs / fx_d
            Y_d = (vv - cy_d) * zs / fy_d

            pts_d = np.stack([X_d, Y_d, zs, np.ones_like(zs)], axis=0)  # 4 x N

            T_wb_s = make_T(R_s, T_s)
            T_wd_s = T_wb_s.dot(T_bd)

            all_pts_world.append(T_wd_s.dot(pts_d)[:3, :])

        if not all_pts_world:
            rospy.logwarn("[LABEL] no valid slow raw-depth points")
            return False

        pts_world = np.concatenate(all_pts_world, axis=1)
        n_total = pts_world.shape[1]

        # --------------------------------------------------------
        # 投影到卷帘快门 RGB, 累积 smear 标注
        # --------------------------------------------------------
        uv_label = np.zeros((H, W, 2), dtype=np.float32)  # 局部位移 (u, v)
        m_label = np.zeros((H, W), dtype=np.float32)  # 有效性 {0,1}
        zbuf = np.full((H, W), np.inf, dtype=np.float32)  # z-buffer, 用于遮挡

        e_frac = 0.5 * self.exposure / max(self.readout, 1e-6)

        n_solve_fail = 0
        n_outside = 0
        n_written = 0
        n_occluded = 0
        smear_lengths = []

        for i in range(n_total):
            X_w = pts_world[:, i]

            # Newton-Raphson 解卷帘行时刻 tau (只保证投影自洽, 不保证可见)
            res = self.solve_tau(X_w, R0, Tv0, R1, Tv1, K_color, T_cb, H)
            if res is None:
                n_solve_fail += 1
                continue

            u_c, v_c, Zc, tau = res

            if not (0 <= u_c < W and 0 <= v_c < H and Zc > 0):
                n_outside += 1
                continue

            ui = int(np.floor(u_c))
            vi = int(np.floor(v_c))

            # ---- z-buffer: 同一像素只留最近的点, 远的判为被遮挡 ----
            if Zc >= zbuf[vi, ui]:
                n_occluded += 1
                continue

            # 曝光起止两个时刻的投影位置
            tau_a = max(0.0, tau - e_frac)
            tau_b = min(1.0, tau + e_frac)

            pa = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_a, K_color, T_cb)
            pb = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_b, K_color, T_cb)

            if pa is None or pb is None:
                continue
            if not (0 <= pa[0] < W and 0 <= pa[1] < H):
                continue
            if not (0 <= pb[0] < W and 0 <= pb[1] < H):
                continue

            # ---- 这就是 (u, v): 曝光期间该点在图像上滑动的位移 ----
            du = pb[0] - pa[0]
            dv = pb[1] - pa[1]

            zbuf[vi, ui] = Zc
            uv_label[vi, ui, 0] = du
            uv_label[vi, ui, 1] = dv
            m_label[vi, ui] = 1.0

            n_written += 1
            smear_lengths.append(np.hypot(du, dv))

        if n_written == 0:
            rospy.logwarn("[LABEL] frame produced 0 valid pixels, skipped")
            return False

        # --------------------------------------------------------
        # 局部 (u,v) -> double-angle (u', v')
        #   d     = sqrt(u^2 + v^2)
        #   alpha = atan2(v, u)
        #   u' = d*cos(2a),  v' = d*sin(2a)
        # --------------------------------------------------------
        d = np.sqrt(uv_label[..., 0] ** 2 + uv_label[..., 1] ** 2)
        alpha = np.arctan2(uv_label[..., 1], uv_label[..., 0])

        u_prime = d * np.cos(2.0 * alpha)
        v_prime = d * np.sin(2.0 * alpha)

        label = np.stack([u_prime, v_prime, m_label], axis=-1).astype(np.float32)

        # dataloader 里有同样的断言, 这里先拦一道
        assert np.all((label[..., 2] == 0) | (label[..., 2] == 1)), \
            "m must be binary {0,1}"
        assert label.shape == (H, W, 3), "label must be (H, W, 3)"

        # --------------------------------------------------------
        # 写盘: png 和 pfm 必须同名同号, 且编号连续
        # --------------------------------------------------------
        frame_id = self.frame_counter
        self.frame_counter += 1

        img_path = os.path.join(self.image_out_dir, "%06d.png" % frame_id)
        lbl_path = os.path.join(self.label_out_dir, "%06d.pfm" % frame_id)

        cv2.imwrite(img_path, color)
        writePFM(lbl_path, label)

        self.fast_done_count += 1

        valid_pct = 100.0 * m_label.sum() / m_label.size
        arr = np.array(smear_lengths)


        # 查找real position
        t_ref_s = t_ref.to_sec()
        t0_s, t1_s = t0.to_sec(), t1.to_sec()

        depth_label = np.where(np.isfinite(zbuf), zbuf, 0.0).astype(np.float32)
        # 三个关键时刻的 base 位姿（插值后的实际位置）
        p_ref, q_ref = self.itp.query(t_ref_s)
        p_0, q_0 = self.itp.query(t0_s)
        p_1, q_1 = self.itp.query(t1_s)

        # readout 窗口内密集采样，供后续评估用
        taus = np.linspace(0.0, 1.0, 11)
        dense_p, dense_q = [], []
        for tau in taus:
            pk, qk = self.itp.query(t0_s + tau * self.readout)
            dense_p.append(pk);
            dense_q.append(qk)

        np.savez(os.path.join(self.pose_out_dir, "%06d.npz" % frame_id),
                 t_ref=t_ref_s, t0=t0_s, t1=t1_s,
                 readout=self.readout, exposure=self.exposure,
                 p_ref=p_ref, q_ref=q_ref,
                 p_0=p_0, q_0=q_0, p_1=p_1, q_1=q_1,
                 taus=taus,
                 dense_p=np.array(dense_p), dense_q=np.array(dense_q),
                 T_cb=T_cb, K_color=K_color,
                 slow_id=best_k, slow_dist=float(pose_dists[best_k]), depth = depth_label)

        #
        rospy.loginfo(
            "[LABEL] %06d valid=%.2f%% (%d px)  occluded=%d solve_fail=%d outside=%d",
            frame_id, valid_pct, n_written, n_occluded, n_solve_fail, n_outside
        )
        rospy.loginfo(
            "[SMEAR_LEN] mean=%.2f median=%.2f p95=%.2f max=%.2f px",
            float(np.mean(arr)), float(np.median(arr)),
            float(np.percentile(arr, 95)), float(np.max(arr))
        )

        return True

    def process_fast_frame_repaced(self, cmsg, ci):
        self.fast_cb_count += 1

        try:
            color = self.bridge.imgmsg_to_cv2(cmsg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("[FAST] color decode failed: %s", str(e))
            return False

        H, W = color.shape[:2]
        K_color = K_from_info(ci)

        fx = K_color[0, 0]
        fy = K_color[1, 1]
        cx = K_color[0, 2]
        cy = K_color[1, 2]

        rospy.loginfo(
            "[CAMERA K] fx=%.2f fy=%.2f cx=%.2f cy=%.2f image=%dx%d",
            fx, fy, cx, cy, W, H
        )

        t_ref = cmsg.header.stamp
        t0 = t_ref - rospy.Duration.from_sec(0.5 * self.readout)
        t1 = t_ref + rospy.Duration.from_sec(0.5 * self.readout)

        try:
            # world <- base
            T0 = tfmsg_to_T(
                self.tfbuf.lookup_transform(
                    self.world_frame,
                    self.base_frame,
                    t0,
                    rospy.Duration(0.1)
                )
            )

            T1 = tfmsg_to_T(
                self.tfbuf.lookup_transform(
                    self.world_frame,
                    self.base_frame,
                    t1,
                    rospy.Duration(0.1)
                )
            )

            T_ref = tfmsg_to_T(
                self.tfbuf.lookup_transform(
                    self.world_frame,
                    self.base_frame,
                    t_ref,
                    rospy.Duration(0.1)
                )
            )

            # color <- base
            # T_cb = tfmsg_to_T(
            #     self.tfbuf.lookup_transform(
            #         self.color_frame,
            #         self.base_frame,
            #         rospy.Time(0),
            #         rospy.Duration(0.1)
            #     )
            # )

            if self.T_cb_fixed is None:
                self.T_cb_fixed = self._compute_fixed_T_cb()
                rospy.loginfo("[FAST] fixed T_cb computed once:\n%s",
                              str(self.T_cb_fixed))

            T_cb = self.T_cb_fixed
            
        except Exception as e:
            rospy.logwarn("[FAST] TF failed: %s", str(e))
            return False

        R0, Tv0 = T0[:3, :3], T0[:3, 3]
        R1, Tv1 = T1[:3, :3], T1[:3, 3]

        yaw0 = self.yaw_from_R(R0)
        yaw1 = self.yaw_from_R(R1)

        dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        yaw_rate_deg = np.degrees(dyaw / max(self.readout, 1e-6))

        rospy.loginfo(
            "[YAW] yaw_rate=%+.2f deg/s over readout=%.3fs",
            yaw_rate_deg,
            self.readout
        )

        # --------------------------------------------------------
        # Choose nearest slow raw-depth frame by pose
        # --------------------------------------------------------

        R_target = T_ref[:3, :3]
        T_target = T_ref[:3, 3]

        pose_dists = np.array([
            self.pose_distance(r["R"], r["T"], R_target, T_target, w_yaw=0.5)
            for r in self.db
        ])

        K_neighbors = min(self.num_slow_neighbors, len(self.db))
        neighbor_idx = np.argsort(pose_dists)[:K_neighbors]

        best_k = int(neighbor_idx[0])

        rospy.loginfo(
            "[SLOW_SELECT] using=%d best_k=%d best_dist=%.4f dists=[%s]",
            K_neighbors,
            best_k,
            float(pose_dists[best_k]),
            ", ".join("%.4f" % pose_dists[i] for i in neighbor_idx)
        )

        # --------------------------------------------------------
        # Build world points from raw slow depth
        # --------------------------------------------------------

        all_pts_world = []

        for sid in neighbor_idx:
            sid = int(sid)
            slow = self.db[sid]

            Z = slow["depth"]
            K_d = slow["K_depth"]
            T_bd = slow["T_bd"]       # base <- depth
            R_s = slow["R"]           # world <- base
            T_s = slow["T"]

            Hs, Ws = Z.shape

            fx_d = K_d[0, 0]
            fy_d = K_d[1, 1]
            cx_d = K_d[0, 2]
            cy_d = K_d[1, 2]

            uu, vv = np.meshgrid(
                np.arange(0, Ws, self.stride),
                np.arange(0, Hs, self.stride)
            )

            uu = uu.flatten()
            vv = vv.flatten()
            zs = Z[vv, uu]

            valid = (
                np.isfinite(zs) &
                (zs > self.min_depth) &
                (zs < self.max_depth)
            )



            rospy.loginfo(
                "[RAW_DEPTH_SAMPLE] slow_id=%d sampled=%d valid=%d %.1f%%",
                sid,
                len(zs),
                int(valid.sum()),
                100.0 * valid.sum() / max(len(zs), 1)
            )

            if valid.sum() == 0:
                continue

            uu = uu[valid]
            vv = vv[valid]
            zs = zs[valid]

            # Raw depth pixel -> depth camera 3D
            X_d = (uu - cx_d) * zs / fx_d
            Y_d = (vv - cy_d) * zs / fy_d

            pts_d = np.stack(
                [X_d, Y_d, zs, np.ones_like(zs)],
                axis=0
            )  # 4 x N

            # depth -> world:
            # world <- depth = world <- base  *  base <- depth
            T_wb_s = make_T(R_s, T_s)
            T_wd_s = T_wb_s.dot(T_bd)

            pts_w = T_wd_s.dot(pts_d)[:3, :]
            all_pts_world.append(pts_w)

        if not all_pts_world:
            rospy.logwarn("[PROJECT] no valid slow raw-depth points")
            return False

        pts_world = np.concatenate(all_pts_world, axis=1)
        n_total = pts_world.shape[1]

        rospy.loginfo("[PROJECT] total world points=%d", n_total)

        # --------------------------------------------------------
        # Project world points to fast rolling-shutter RGB
        # --------------------------------------------------------

        sparse = np.zeros((H, W), dtype=np.float32)
        proj_vis = np.zeros((H, W), dtype=np.uint8)
        smear_vis = color.copy()

        e_frac = 0.5 * self.exposure / max(self.readout, 1e-6)

        n_solve_fail = 0
        n_inside = 0
        n_outside = 0
        n_drawn = 0

        smear_lengths = []

        for i in range(n_total):
            X_w = pts_world[:, i]

            res = self.solve_tau(X_w, R0, Tv0, R1, Tv1, K_color, T_cb, H)

            if res is None:
                n_solve_fail += 1
                continue

            u, v, Zc, tau = res

            if not (0 <= u < W and 0 <= v < H and Zc > 0):
                n_outside += 1
                continue

            n_inside += 1

            ui = int(np.floor(u))
            vi = int(np.floor(v))

            proj_vis[vi, ui] = 255

            cur = sparse[vi, ui]
            if cur == 0 or Zc < cur:
                sparse[vi, ui] = Zc

            if self.smear_stride > 1 and (i % self.smear_stride) != 0:
                continue

            tau_a = max(0.0, tau - e_frac)
            tau_b = min(1.0, tau + e_frac)

            pa = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_a, K_color, T_cb)
            pb = self._project_tau(X_w, R0, Tv0, R1, Tv1, tau_b, K_color, T_cb)

            if pa is None or pb is None:
                continue

            if not (0 <= pa[0] < W and 0 <= pa[1] < H):
                continue

            if not (0 <= pb[0] < W and 0 <= pb[1] < H):
                continue

            n_drawn += 1

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

        rospy.loginfo(
            "[PROJECT_RESULT] total=%d inside=%d %.1f%% solve_fail=%d %.1f%% outside=%d %.1f%% drawn=%d",
            n_total,
            n_inside,
            100.0 * n_inside / max(n_total, 1),
            n_solve_fail,
            100.0 * n_solve_fail / max(n_total, 1),
            n_outside,
            100.0 * n_outside / max(n_total, 1),
            n_drawn
        )

        if smear_lengths:
            arr = np.array(smear_lengths)
            rospy.loginfo(
                "[SMEAR_LEN] mean=%.2f median=%.2f p95=%.2f max=%.2f px",
                float(np.mean(arr)),
                float(np.median(arr)),
                float(np.percentile(arr, 95)),
                float(np.max(arr))
            )

        # --------------------------------------------------------
        # Publish visualization topics
        # --------------------------------------------------------

        self.pub_color.publish(self.bridge.cv2_to_imgmsg(color, "bgr8"))
        self.pub_proj.publish(self.bridge.cv2_to_imgmsg(proj_vis, "mono8"))
        self.pub_smear.publish(self.bridge.cv2_to_imgmsg(smear_vis, "bgr8"))

        m = sparse.copy()
        m[m <= 0] = np.nan

        try:
            lo = np.nanpercentile(m, 5)
            hi = np.nanpercentile(m, 95)
        except Exception:
            lo, hi = 0.3, 5.0

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.3, 5.0

        norm = (np.clip(sparse, lo, hi) - lo) / (hi - lo + 1e-9)
        u8 = (norm * 255).astype(np.uint8)
        u8[sparse <= 0] = 0

        depth_vis = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        depth_vis[sparse <= 0] = 0

        self.pub_depth_vis.publish(self.bridge.cv2_to_imgmsg(depth_vis, "bgr8"))

        self.fast_done_count += 1


        if self.save_every > 0 and self.fast_done_count % self.save_every == 0:
            cv2.imwrite("/proj_raw_%06d.png" % self.fast_done_count, proj_vis)
            cv2.imwrite("/smear_raw_%06d.png" % self.fast_done_count, smear_vis)
            cv2.imwrite("/sparse_raw_%06d.png" % self.fast_done_count, depth_vis)

        rospy.loginfo(
            "[FAST_DONE_RAW_DIRECT] cb=%d done=%d best_k=%d best_dist=%.4f",
            self.fast_cb_count,
            self.fast_done_count,
            best_k,
            float(pose_dists[best_k])
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

    ap.add_argument("--start_frame", type=int, default=1)
    ap.add_argument("--end_frame", type=int, default=0)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0)

    args, _ = ap.parse_known_args()

    if args.mode == "build":
        rospy.init_node("slow_raw_builder", anonymous=True)
        SlowRawBuilder(args.out)
        rospy.spin()

    else:
        rospy.init_node("raw_depth_to_fast_projector", anonymous=True)

        if args.fast_bag is None:
            projector = RawDepthProjector(args.db, subscribe=True)
            rospy.spin()
        else:
            projector = RawDepthProjector(args.db, subscribe=False)
            projector.run_offline(
                args.fast_bag,
                color_topic=args.color_topic,
                camera_info_topic=args.camera_info_topic,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                frame_step=args.frame_step,
                max_frames=args.max_frames
            )