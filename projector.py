#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
slow_to_fast_projector.py  (ROS1 / Python2.7, OpenCV 3.2)

Pipeline:
  - Listens to TWO bags (played in parallel or sequentially):
      * slow bag: clean depth + color + Qualisys poses (short exposure, slow motion)
      * fast bag: blurry color + Qualisys poses (long exposure, fast motion)
  - For each blurry frame in fast bag:
      1. Find the closest slow frame by pose distance
      2. Take that slow depth, lift to 3D world points
      3. For each world point, solve for tau (0..1) such that
         y/H == tau, i.e. the RS capture time of that row matches the
         pose used to project it. Newton-Raphson on tau.
      4. If converged and valid, assign depth to (x,y) in blurry image.
      5. Generate smear path: project same point at tau - e/2 and tau + e/2.

Publishes:
  - /blur_color           : the blurry image (republished, for sync)
  - /sparse_depth         : sparse depth GT (same size as blurry image)
  - /sparse_depth_vis     : colormap of sparse depth for rqt_image_view
  - /smear_path_vis       : visualization with arrows for each point
  - /smear_path           : custom topic with (x,y,dx,dy,depth) arrays

Usage (simplest):
  - Load slow bag frames to disk first (or keep in memory)
  - Run this node, play the FAST bag
  - Node will query closest slow frame from its in-memory DB
"""

from __future__ import print_function
import numpy as np
import rospy
import cv2
import message_filters
import os
import pickle

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

import tf2_ros
import tf.transformations as tft

# ============================================================
# Math helpers
# ============================================================

def tfmsg_to_RT(transform):
    """Extract R (3x3) and T (3,) from geometry_msgs/TransformStamped."""
    t = transform.transform.translation
    q = transform.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    R = T[:3, :3].astype(np.float64)
    Tr = np.array([t.x, t.y, t.z], dtype=np.float64)
    return R, Tr


def pose_to_RT(T44):
    return T44[:3, :3].copy(), T44[:3, 3].copy()


def RT_to_pose(R, T):
    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = R
    P[:3, 3] = T
    return P


def slerp_R(R1, R2, tau):
    """SLERP between rotations using matrix log/exp (small step safe)."""
    R_rel = R1.T.dot(R2)
    # logm via axis-angle
    # Convert to quaternion, slerp, back to R (more numerically stable)
    q1 = tft.quaternion_from_matrix(RT_to_pose(R1, np.zeros(3)))
    q2 = tft.quaternion_from_matrix(RT_to_pose(R2, np.zeros(3)))
    # Ensure shortest path
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    q = tft.quaternion_slerp(q1, q2, float(tau))
    return tft.quaternion_matrix(q)[:3, :3]


def lerp_T(T1, T2, tau):
    return T1 + (T2 - T1) * float(tau)


def K_from_info(info):
    K = np.array(info.K, dtype=np.float64).reshape(3, 3)
    return K


# ============================================================
# Slow-frame database
# ============================================================

class SlowFrameDB(object):
    """
    Holds N slow-frame records in memory, each:
      { 'R': 3x3, 'T': 3,
        'depth': HxW float32 (in meters),
        'K_d': 3x3, 'K_c': 3x3,
        'T_c_d': 4x4 (color <- depth static),
        'color_frame': str, 'depth_frame': str }
    """
    def __init__(self):
        self.frames = []

    def add(self, record):
        self.frames.append(record)

    def __len__(self):
        return len(self.frames)

    def find_closest(self, R_target, T_target, w_trans=1.0, w_rot=0.3):
        """Return the record with smallest weighted pose distance."""
        if not self.frames:
            return None, -1, None
        best_i = -1
        best_d = float('inf')
        for i, f in enumerate(self.frames):
            dT = np.linalg.norm(f['T'] - T_target)
            # rotation distance via axis-angle magnitude
            R_rel = f['R'].T.dot(R_target)
            tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
            dR = np.arccos(tr)
            d = w_trans * dT + w_rot * dR
            if d < best_d:
                best_d = d
                best_i = i
        return self.frames[best_i], best_i, best_d


# ============================================================
# Core projector
# ============================================================

class Projector(object):
    def __init__(self):
        self.bridge = CvBridge()

        # --- topics ---
        self.fast_color_topic = rospy.get_param("~fast_color", "/camera/color/image_raw")
        self.fast_color_info_topic = rospy.get_param("~fast_color_info", "/camera/color/camera_info")

        self.pub_color = rospy.Publisher("/blur_color", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("/sparse_depth", Image, queue_size=1)
        self.pub_depth_vis = rospy.Publisher("/sparse_depth_vis", Image, queue_size=1)
        self.pub_smear_vis = rospy.Publisher("/smear_path_vis", Image, queue_size=1)
        self.pub_smear_array = rospy.Publisher("/smear_path", Float32MultiArray, queue_size=1)

        # --- frames ---
        self.world_frame = rospy.get_param("~world_frame", "mocap")
        self.base_frame = rospy.get_param("~base_frame", "hamster")
        self.color_frame = rospy.get_param("~color_frame", "camera_color_optical_frame")
        self.depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")

        # --- timing ---
        self.readout_sec = float(rospy.get_param("~readout_sec", 0.030))
        self.exposure_sec = float(rospy.get_param("~exposure_sec", 0.0166))

        # --- depth sampling ---
        self.depth_stride = int(rospy.get_param("~depth_stride", 4))
        self.min_depth = float(rospy.get_param("~min_depth", 0.3))
        self.max_depth = float(rospy.get_param("~max_depth", 10.0))

        # --- Newton-Raphson ---
        self.nr_max_iter = int(rospy.get_param("~nr_max_iter", 8))
        self.nr_tol = float(rospy.get_param("~nr_tol", 1e-3))

        # --- slow DB ---
        self.slow_db_path = rospy.get_param("~slow_db_path", "")  # path to pickle
        self.db = SlowFrameDB()
        if self.slow_db_path and os.path.exists(self.slow_db_path):
            with open(self.slow_db_path, 'rb') as f:
                self.db.frames = pickle.load(f)
            rospy.loginfo("Loaded %d slow frames from %s", len(self.db), self.slow_db_path)
        else:
            rospy.logwarn("No slow DB loaded. Set ~slow_db_path.")

        # --- TF ---
        self.tfbuf = tf2_ros.Buffer(rospy.Duration(60.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # --- sync ---
        sub_c = message_filters.Subscriber(self.fast_color_topic, Image)
        sub_ci = message_filters.Subscriber(self.fast_color_info_topic, CameraInfo)
        slop = float(rospy.get_param("~slop", 0.03))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_c, sub_ci], queue_size=10, slop=slop)
        self.sync.registerCallback(self.cb_fast)

        self.frame_count = 0
        rospy.loginfo("slow_to_fast_projector ready.")

    # ------------------------------------------------------------
    # Given world point X_w (3,) and a pose track
    #   R(tau) = SLERP(R0, R1, tau),  T(tau) = LERP(T0, T1, tau)
    # find tau such that the projected y equals tau * H.
    # Returns (x, y, z_cam, tau, converged).
    # ------------------------------------------------------------
    def project_with_tau(self, X_w, R0, T0, R1, T1, K, H, W,
                         tau_init=0.0, T_c_b=None):
        """
        X_w      : (3,) point in world
        R0,T0    : world pose at tau=0 (start of readout)
        R1,T1    : world pose at tau=1 (end of readout)
        K        : (3,3) color intrinsics
        H, W     : image size
        T_c_b    : static 4x4 color<-base extrinsic

        Note: R*, T* are "world <- base" transforms. To project, we need
        the base-frame pose at time tau, invert it to get world->base,
        then apply T_c_b to get world->color.
        """
        tau = float(tau_init)
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]

        for it in range(self.nr_max_iter):
            # base pose at tau
            R_b = slerp_R(R0, R1, tau)
            T_b = lerp_T(T0, T1, tau)
            # world -> base
            R_bw = R_b.T
            T_bw = -R_bw.dot(T_b)
            # world -> color = T_c_b * T_b_w
            P_bw = RT_to_pose(R_bw, T_bw)
            P_cw = T_c_b.dot(P_bw)
            R_cw, T_cw = pose_to_RT(P_cw)

            X_c = R_cw.dot(X_w) + T_cw  # point in color frame
            Zc = X_c[2]
            if Zc <= 1e-3:
                return 0, 0, 0.0, tau, False

            u = fx * X_c[0] / Zc + cx
            v = fy * X_c[1] / Zc + cy

            # residual: we want v/H == tau
            r = v / float(H - 1) - tau
            if abs(r) < self.nr_tol:
                return u, v, Zc, tau, True

            # Numerical derivative of r w.r.t. tau
            dtau = 1e-4
            tau2 = np.clip(tau + dtau, 0.0, 1.0)
            R_b2 = slerp_R(R0, R1, tau2)
            T_b2 = lerp_T(T0, T1, tau2)
            R_bw2 = R_b2.T
            T_bw2 = -R_bw2.dot(T_b2)
            P_bw2 = RT_to_pose(R_bw2, T_bw2)
            P_cw2 = T_c_b.dot(P_bw2)
            R_cw2, T_cw2 = pose_to_RT(P_cw2)
            X_c2 = R_cw2.dot(X_w) + T_cw2
            Zc2 = X_c2[2]
            if Zc2 <= 1e-3:
                return 0, 0, 0.0, tau, False
            v2 = fy * X_c2[1] / Zc2 + cy

            dr = ((v2 / float(H - 1) - tau2) - r) / dtau
            if abs(dr) < 1e-8:
                return u, v, Zc, tau, False

            tau_new = tau - r / dr
            tau_new = max(0.0, min(1.0, tau_new))
            tau = tau_new

        return u, v, Zc, tau, False

    # ------------------------------------------------------------
    # Fast frame callback
    # ------------------------------------------------------------
    def cb_fast(self, color_msg, color_info):
        if len(self.db) == 0:
            rospy.logwarn_throttle(5.0, "Slow DB empty; skip fast frame.")
            return

        self.frame_count += 1

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        H, W = color.shape[:2]
        K_f = K_from_info(color_info)

        t_ref = color_msg.header.stamp
        t_start = t_ref + rospy.Duration.from_sec(-0.5 * self.readout_sec)
        t_end = t_ref + rospy.Duration.from_sec(0.5 * self.readout_sec)

        # Look up base poses at readout start/end
        try:
            R0, T0 = tfmsg_to_RT(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_start, rospy.Duration(0.1)))
            R1, T1 = tfmsg_to_RT(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_end, rospy.Duration(0.1)))
            R_ref, T_ref = tfmsg_to_RT(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_ref, rospy.Duration(0.1)))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Pose lookup fail: %s" % str(e))
            return

        # Static color <- base
        try:
            tf_cb = self.tfbuf.lookup_transform(
                self.color_frame, self.base_frame,
                rospy.Time(0), rospy.Duration(0.1))
            R_cb, T_cb = tfmsg_to_RT(tf_cb)
            T_c_b = RT_to_pose(R_cb, T_cb)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Static TF color<-base fail: %s" % str(e))
            return

        # Poses for exposure window (for smear path): +/- e/2 around tau
        # Represented as additional SLERP/LERP but over a larger window
        t_exp_start = t_ref + rospy.Duration.from_sec(-0.5 * self.exposure_sec)
        t_exp_end = t_ref + rospy.Duration.from_sec(0.5 * self.exposure_sec)
        try:
            R_exp0, T_exp0 = tfmsg_to_RT(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_exp_start, rospy.Duration(0.05)))
            R_exp1, T_exp1 = tfmsg_to_RT(self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_exp_end, rospy.Duration(0.05)))
            exposure_ok = True
        except Exception:
            exposure_ok = False

        # Find closest slow frame by pose
        slow, idx, dist = self.db.find_closest(R_ref, T_ref)
        if slow is None:
            return

        # Lift slow depth to world points (in color frame coordinates)
        # The slow frame already has depth aligned to color (we assume it's
        # the "mapped" depth from publish1.py output).
        Z = slow['depth']
        Hs, Ws = Z.shape
        K_c_slow = slow['K_c']
        R_s, T_s = slow['R'], slow['T']
        T_c_b_slow = slow['T_c_b']  # color <- base (static, same as ours)

        # Build base->world of slow frame
        P_bw_slow = RT_to_pose(R_s, T_s)  # world <- base
        # color<-world for slow = T_c_b * world<-base^-1
        R_bw_s = R_s.T
        T_bw_s = -R_bw_s.dot(T_s)
        P_wc_slow_inv = RT_to_pose(R_bw_s, T_bw_s)  # base <- world
        P_cw_slow = T_c_b_slow.dot(P_wc_slow_inv)   # color <- world
        # we need world <- color, so invert
        P_wc_slow = np.linalg.inv(P_cw_slow)
        R_wc_s, T_wc_s = pose_to_RT(P_wc_slow)

        # Sample depth points
        stride = self.depth_stride
        us = np.arange(0, Ws, stride)
        vs = np.arange(0, Hs, stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.flatten()
        vv = vv.flatten()
        zs = Z[vv, uu]
        valid = (zs > self.min_depth) & (zs < self.max_depth) & np.isfinite(zs)
        uu = uu[valid]; vv = vv[valid]; zs = zs[valid]
        N = zs.size
        if N == 0:
            return

        # Lift to 3D in slow-color frame
        fx_s = K_c_slow[0, 0]; fy_s = K_c_slow[1, 1]
        cx_s = K_c_slow[0, 2]; cy_s = K_c_slow[1, 2]
        X_sc = (uu - cx_s) * zs / fx_s
        Y_sc = (vv - cy_s) * zs / fy_s
        pts_cam = np.stack([X_sc, Y_sc, zs, np.ones(N)], axis=0)  # (4, N)

        # To world: X_w = P_wc_slow * X_cam
        pts_world = P_wc_slow.dot(pts_cam)[:3, :]  # (3, N)

        # Output buffers
        sparse_depth = np.zeros((H, W), dtype=np.float32)
        smear_vis = color.copy()
        smear_records = []

        n_conv = 0
        n_inside = 0

        # Project each point with NR
        for i in range(N):
            X_w = pts_world[:, i]
            u, v, Zc, tau, ok = self.project_with_tau(
                X_w, R0, T0, R1, T1, K_f, H, W, tau_init=0.0, T_c_b=T_c_b)
            if not ok:
                continue
            n_conv += 1
            if (u < 0) or (u >= W) or (v < 0) or (v >= H) or (Zc <= 0):
                continue
            n_inside += 1

            ui = int(round(u)); vi = int(round(v))
            ui = max(0, min(W - 1, ui))
            vi = max(0, min(H - 1, vi))

            # Z-buffer: keep nearest
            cur = sparse_depth[vi, ui]
            if (cur == 0.0) or (Zc < cur):
                sparse_depth[vi, ui] = Zc

            # Smear path: project at tau - e/2 and tau + e/2 using the same
            # readout window but offset by exposure (convert exposure to readout
            # fraction). exposure_frac = exposure_sec / readout_sec
            if exposure_ok:
                e_frac = 0.5 * self.exposure_sec / max(self.readout_sec, 1e-6)
                tau_a = max(0.0, tau - e_frac)
                tau_b = min(1.0, tau + e_frac)

                u_a, v_a = self._project_at(X_w, R0, T0, R1, T1, tau_a, K_f, T_c_b)
                u_b, v_b = self._project_at(X_w, R0, T0, R1, T1, tau_b, K_f, T_c_b)
                if (u_a is not None) and (u_b is not None):
                    smear_records.append((ui, vi, u_a, v_a, u_b, v_b, Zc))
                    cv2.arrowedLine(
                        smear_vis,
                        (int(round(u_a)), int(round(v_a))),
                        (int(round(u_b)), int(round(v_b))),
                        (0, 255, 255), 1, tipLength=0.3)
                    cv2.circle(smear_vis, (ui, vi), 1, (0, 0, 255), -1)

        # Publish
        self._publish_color(color, color_msg.header)
        self._publish_sparse_depth(sparse_depth, color_msg.header)
        self._publish_smear(smear_vis, smear_records, color_msg.header)

        rospy.loginfo(
            "[S2F] frame=%d  slow_idx=%d  dist=%.3f  pts: N=%d conv=%d inside=%d",
            self.frame_count, idx, dist, N, n_conv, n_inside)

    def _project_at(self, X_w, R0, T0, R1, T1, tau, K, T_c_b):
        """One-shot projection at given tau, no NR."""
        R_b = slerp_R(R0, R1, tau)
        T_b = lerp_T(T0, T1, tau)
        R_bw = R_b.T
        T_bw = -R_bw.dot(T_b)
        P_bw = RT_to_pose(R_bw, T_bw)
        P_cw = T_c_b.dot(P_bw)
        R_cw, T_cw = pose_to_RT(P_cw)
        X_c = R_cw.dot(X_w) + T_cw
        Zc = X_c[2]
        if Zc <= 1e-3:
            return None, None
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        u = fx * X_c[0] / Zc + cx
        v = fy * X_c[1] / Zc + cy
        return u, v

    # ------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------
    def _publish_color(self, color, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
            msg.header = header
            self.pub_color.publish(msg)
        except Exception as e:
            rospy.logwarn("pub color fail: %s" % str(e))

    def _publish_sparse_depth(self, depth, header):
        try:
            msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            msg.header = header
            self.pub_depth.publish(msg)
        except Exception as e:
            rospy.logwarn("pub depth fail: %s" % str(e))

        # Colormap visualization
        m = depth.copy()
        m[m <= 0] = np.nan
        try:
            dmin = np.nanpercentile(m, 5)
            dmax = np.nanpercentile(m, 95)
        except Exception:
            dmin, dmax = 0.3, 5.0
        if (not np.isfinite(dmin)) or (not np.isfinite(dmax)) or (dmax <= dmin):
            dmin, dmax = 0.3, 5.0
        d_norm = (np.clip(depth, dmin, dmax) - dmin) / (dmax - dmin + 1e-9)
        d_u8 = (d_norm * 255.0).astype(np.uint8)
        d_u8[depth <= 0] = 0
        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        d_color[depth <= 0] = 0
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(d_color, encoding="bgr8")
            vis_msg.header = header
            self.pub_depth_vis.publish(vis_msg)
        except Exception as e:
            rospy.logwarn("pub depth_vis fail: %s" % str(e))

    def _publish_smear(self, vis, records, header):
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            vis_msg.header = header
            self.pub_smear_vis.publish(vis_msg)
        except Exception as e:
            rospy.logwarn("pub smear_vis fail: %s" % str(e))

        # Flatten records: (N, 7) as float32
        if records:
            arr = np.array(records, dtype=np.float32)
            msg = Float32MultiArray()
            msg.layout.dim.append(MultiArrayDimension(
                label="point", size=arr.shape[0], stride=arr.size))
            msg.layout.dim.append(MultiArrayDimension(
                label="field", size=arr.shape[1], stride=arr.shape[1]))
            msg.data = arr.flatten().tolist()
            self.pub_smear_array.publish(msg)


if __name__ == "__main__":
    rospy.init_node("slow_to_fast_projector", anonymous=True)
    Projector()
    rospy.spin()
