#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
publish1.py  (ROS1 / Python2.7, OpenCV 3.2)

Depth->Color mapping + overlay image + colored point cloud, with:
- robust TF frames (use params, but can fallback to message header frames)
- correct (v,u) indexing everywhere
- safe integer pixel indexing (clip to image bounds)
- two-stage modes:
    * overlay_mode: 'round' or 'bilinear_splat' (for visualization only)
    * keep_mode:    'round' or 'floor'          (for selecting points for cloud)
- keep_eps separated for overlay/cloud
- optional filters:
    * grad filter (Sobel/Scharr), threshold: percentile/fixed
    * fly filter (local median outlier removal) on ORIGINAL depth (scheme-3)
    * publish mapped depth colormap as a separate topic

NOTE:
- fly filter is applied on the original depth image BEFORE projection to avoid
  mapped-depth sparsity / zero contamination issues.
"""

import numpy as np
import rospy
import cv2
import message_filters

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

import tf2_ros
import tf.transformations as tft


def K_from_info(info):
    K = np.array(info.K, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return fx, fy, cx, cy


def tfmsg_to_T(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float64)
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def _fast_median_filter_float(Z, k):
    """
    Fast median for 2D float32 using numpy stride tricks.
    - Works for OpenCV builds where cv2.medianBlur doesn't accept float.
    - O(H*W*k*k) but with vectorized median; k<=7 is usually OK at 480x640.
    """
    if k <= 1:
        return Z
    if k % 2 != 1:
        k = k + 1
    r = k // 2
    Zp = np.pad(Z, ((r, r), (r, r)), mode='edge')
    H, W = Z.shape
    s0, s1 = Zp.strides
    windows = np.lib.stride_tricks.as_strided(
        Zp,
        shape=(H, W, k, k),
        strides=(s0, s1, s0, s1)
    )
    return np.median(windows, axis=(2, 3)).astype(np.float32)


def _remove_small_cc(mask_bool, min_area):
    if min_area <= 0:
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    if num <= 1:
        return mask_bool
    out = np.zeros_like(mask_bool, dtype=bool)
    for lab in range(1, num):
        area = int(np.sum(labels == lab))
        if area >= int(min_area):
            out |= (labels == lab)
    return out


class Mapper(object):
    def __init__(self):
        self.bridge = CvBridge()

        # input topics
        self.depth_topic = rospy.get_param("~depth", "/camera/depth/image_rect_raw")
        self.color_topic = rospy.get_param("~color", "/camera/color/image_raw")
        self.depth_info_topic = rospy.get_param("~depth_info", "/camera/depth/camera_info")
        self.color_info_topic = rospy.get_param("~color_info", "/camera/color/camera_info")

        # frames
        self.depth_frame_param = rospy.get_param("~depth_frame", "camera_depth_optical_frame")
        self.color_frame_param = rospy.get_param("~color_frame", "camera_color_optical_frame")
        self.use_msg_frames = bool(rospy.get_param("~use_msg_frames", False))

        self.color_rect_topic = rospy.get_param("~color_rect_topic", "/color_rectified")

        # outputs
        self.overlay_topic = rospy.get_param("~overlay_topic", "/overlay_image")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/colored_cloud")
        self.depth_vis_topic = rospy.get_param("~depth_vis_topic", "/mapped_depth_color")

        self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=1)
        self.pub_cloud = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1)
        self.pub_depth_vis = rospy.Publisher(self.depth_vis_topic, Image, queue_size=1)
        self.pub_color_rect = rospy.Publisher(self.color_rect_topic, Image, queue_size=1)

        # visualization blend
        self.alpha = float(rospy.get_param("~alpha", 0.35))

        # depth range
        self.min_depth = float(rospy.get_param("~min_depth", 0.1))
        self.max_depth = float(rospy.get_param("~max_depth", 10.0))

        # decimation
        self.publish_every = int(rospy.get_param("~publish_every", 1))

        # rolling shutter rectification
        self.rs_enable = bool(rospy.get_param("~rs_enable", True))
        self.rs_readout_sec = float(rospy.get_param("~rs_readout_sec", 0.001))
        self.rs_strip = int(rospy.get_param("~rs_strip", 16))
        self.rs_dir = float(rospy.get_param("~rs_dir", 1.0))
        self.rs_ref_row = float(rospy.get_param("~rs_ref_row", 0.5))

        # modes
        self.overlay_mode = rospy.get_param("~overlay_mode", "round").lower()
        self.keep_mode = rospy.get_param("~keep_mode", "round").lower()

        # world/base frames for motion compensation
        self.world_frame = rospy.get_param("~world_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.rs_z0 = float(rospy.get_param("~rs_z0", 2.0))
        # eps
        self.keep_eps_overlay = float(rospy.get_param("~keep_eps_overlay", 0.002))
        self.keep_eps_cloud = float(rospy.get_param("~keep_eps_cloud", 0.002))
        if rospy.has_param("~keep_eps"):
            eps = float(rospy.get_param("~keep_eps"))
            self.keep_eps_overlay = eps
            self.keep_eps_cloud = eps

        # ---- Grad filter ----
        self.grad_enable = bool(rospy.get_param("~grad_enable", True))
        self.grad_op = rospy.get_param("~grad_op", "sobel").lower()
        self.grad_ksize = int(rospy.get_param("~grad_ksize", 3))
        self.grad_mode = rospy.get_param("~grad_mode", "percentile").lower()
        self.grad_percentile = float(rospy.get_param("~grad_percentile", 99.0))
        self.grad_fixed_thresh = float(rospy.get_param("~grad_fixed_thresh", 2.0))
        self.grad_mask_dilate = int(rospy.get_param("~grad_mask_dilate", 0))

        self.median_ksize = int(rospy.get_param("~median_ksize", 0))
        self.mask_small_cc = bool(rospy.get_param("~mask_small_cc", False))
        self.mask_min_area = int(rospy.get_param("~mask_min_area", 400))

        # ---- Fly filter (scheme-3: on original depth) ----
        self.fly_enable = bool(rospy.get_param("~fly_enable", False))
        self.fly_mode = rospy.get_param("~fly_mode", "far").lower()
        self.fly_ksize = int(rospy.get_param("~fly_ksize", 5))
        self.fly_abs_th = float(rospy.get_param("~fly_abs_th", 0.04))
        self.fly_rel_th = float(rospy.get_param("~fly_rel_th", 0.02))

        # prints
        self.stats_every = int(rospy.get_param("~stats_every", 0))
        self._last_cfg_printed = False
        self.cb_count = 0

        # TF
        self.tfbuf = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        # sync
        sub_d = message_filters.Subscriber(self.depth_topic, Image)
        sub_c = message_filters.Subscriber(self.color_topic, Image)
        sub_di = message_filters.Subscriber(self.depth_info_topic, CameraInfo)
        sub_ci = message_filters.Subscriber(self.color_info_topic, CameraInfo)

        slop = float(rospy.get_param("~slop", 0.03))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_d, sub_c, sub_di, sub_ci],
            queue_size=10,
            slop=slop
        )
        self.sync.registerCallback(self.cb)

        rospy.loginfo("Publishing:\n  overlay_topic=%s\n  depth_vis_topic=%s\n  cloud_topic=%s",
                      self.overlay_topic, self.depth_vis_topic, self.cloud_topic)
        rospy.loginfo("Inputs:\n  depth=%s\n  color=%s\n  depth_info=%s\n  color_info=%s",
                      self.depth_topic, self.color_topic, self.depth_info_topic, self.color_info_topic)
        rospy.logwarn(
            "RECT CFG | world_frame=%s base_frame=%s rs_enable=%s rs_readout_sec=%.6f rs_strip=%d rs_dir=%.1f rs_ref_row=%.3f slop=%.4f",
            self.world_frame, self.base_frame,
            str(self.rs_enable), self.rs_readout_sec,
            self.rs_strip, self.rs_dir, self.rs_ref_row, slop)

    def _print_cfg_once(self):
        if self._last_cfg_printed:
            return
        self._last_cfg_printed = True
        rospy.loginfo(
            "CFG: overlay_mode=%s keep_mode=%s keep_eps_overlay=%.6f keep_eps_cloud=%.6f depth_range=[%.2f, %.2f] | "
            "grad_enable=%s op=%s mode=%s p=%.1f fixed=%.3f ksize=%d median=%d dilate=%d cc=%s min_area=%d | "
            "fly_enable=%s(mode=%s) ksize=%d abs=%.3f rel=%.3f (scheme-3 on depth) | "
            "stats_every=%d use_msg_frames=%s",
            self.overlay_mode, self.keep_mode, self.keep_eps_overlay, self.keep_eps_cloud,
            self.min_depth, self.max_depth,
            str(self.grad_enable), self.grad_op, self.grad_mode, self.grad_percentile, self.grad_fixed_thresh,
            self.grad_ksize, self.median_ksize, self.grad_mask_dilate,
            str(self.mask_small_cc), self.mask_min_area,
            str(self.fly_enable), self.fly_mode, self.fly_ksize, self.fly_abs_th, self.fly_rel_th,
            self.stats_every, str(self.use_msg_frames)
        )

    def _compute_grad_bad(self, mapped):
        Zf = mapped.astype(np.float32)
        base_valid = (Zf > 0)

        if self.median_ksize and self.median_ksize >= 3 and (self.median_ksize % 2 == 1):
            Zf = _fast_median_filter_float(Zf, self.median_ksize)

        if self.grad_op == "scharr":
            gx = cv2.Scharr(Zf, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(Zf, cv2.CV_32F, 0, 1)
        else:
            k = self.grad_ksize if self.grad_ksize in [1, 3, 5, 7] else 3
            gx = cv2.Sobel(Zf, cv2.CV_32F, 1, 0, ksize=k)
            gy = cv2.Sobel(Zf, cv2.CV_32F, 0, 1, ksize=k)

        G = cv2.magnitude(gx, gy)
        vals = G[base_valid]
        if vals.size == 0:
            return np.zeros_like(base_valid, dtype=bool), 0.0, {"base": 0, "bad": 0}

        if self.grad_mode == "fixed":
            Tgrad = float(self.grad_fixed_thresh)
        else:
            Tgrad = float(np.percentile(vals, self.grad_percentile))

        grad_bad = (G > Tgrad) & base_valid

        if self.grad_mask_dilate > 0:
            gb = (grad_bad.astype(np.uint8) * 255)
            gb = cv2.dilate(gb, np.ones((3, 3), np.uint8), iterations=int(self.grad_mask_dilate))
            grad_bad = gb.astype(bool)

        if self.mask_small_cc:
            grad_bad = _remove_small_cc(grad_bad, self.mask_min_area)

        stats = {"base": int(base_valid.sum()), "bad": int(grad_bad.sum())}
        return grad_bad, Tgrad, stats

    def _apply_fly_filter_depth(self, z):
        Z = z.astype(np.float32)
        base = np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)
        base_n = int(base.sum())
        if base_n == 0:
            return np.zeros_like(base, dtype=bool), 0, 0

        k = int(self.fly_ksize)
        if k < 3:
            k = 3
        if (k % 2) != 1:
            k += 1

        base_u8 = base.astype(np.uint8)
        valid_cnt = cv2.boxFilter(base_u8, ddepth=-1, ksize=(k, k), normalize=False)
        enough = valid_cnt >= int(0.3 * k * k)

        Z_fill = Z.copy()
        Z_fill[~base] = 0.0
        Zmed = _fast_median_filter_float(Z_fill, k)

        eval_mask = base & enough

        th = np.maximum(float(self.fly_abs_th),
                        float(self.fly_rel_th) * np.maximum(Zmed, 1e-6))

        fly_bad = np.zeros_like(base, dtype=bool)
        if self.fly_mode == "near":
            fly_bad[eval_mask] = (Z[eval_mask] < (Zmed[eval_mask] - th[eval_mask]))
        else:
            fly_bad[eval_mask] = (Z[eval_mask] > (Zmed[eval_mask] + th[eval_mask]))

        return fly_bad, base_n, int(fly_bad.sum())

    def _get_compensated_T(self, t_d, t_c, depth_frame, color_frame):
        try:
            tf_cd = self.tfbuf.lookup_transform(color_frame, depth_frame, rospy.Time(0), rospy.Duration(0.1))
            T_cd_static = tfmsg_to_T(tf_cd)

            tf_star = self.tfbuf.lookup_transform(self.base_frame, depth_frame, rospy.Time(0), rospy.Duration(0.1))
            T_star = tfmsg_to_T(tf_star)
            T_star_inv = np.linalg.inv(T_star)

            tf_td = self.tfbuf.lookup_transform(self.world_frame, self.base_frame, t_d, rospy.Duration(0.1))
            tf_tc = self.tfbuf.lookup_transform(self.world_frame, self.base_frame, t_c, rospy.Duration(0.1))
            T_m_td = tfmsg_to_T(tf_td)
            T_m_tc = tfmsg_to_T(tf_tc)

            Delta_T = np.linalg.inv(T_m_tc).dot(T_m_td)

            trans = tft.translation_from_matrix(Delta_T)
            dist_mm = np.linalg.norm(trans) * 1000.0
            dt_ms = (t_c - t_d).to_sec() * 1000.0

            rospy.loginfo_throttle(
                1.0,
                ">>> [COMP ACTIVE] Delay: %.3fms | Move: %.3fmm | Status: OK" % (dt_ms, dist_mm)
            )

            T_compensated = T_star_inv.dot(Delta_T).dot(T_star)
            return T_cd_static.dot(T_compensated)

        except Exception as e:
            rospy.logwarn_throttle(2.0, "运动补偿失败: %s" % str(e))
            return None

    def row_time(self, t_ref, v, H):
        v_ref = self.rs_ref_row * float(H - 1)
        alpha = (float(v) - v_ref) / max(float(H - 1), 1.0)
        dt = self.rs_dir * alpha * self.rs_readout_sec
        return t_ref + rospy.Duration.from_sec(dt)

    def _summarize_T(self, T):
        trans = T[:3, 3]
        dist_mm = np.linalg.norm(trans) * 1000.0
        yaw_deg = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        return {
            "tx": float(trans[0]),
            "ty": float(trans[1]),
            "tz": float(trans[2]),
            "dist_mm": float(dist_mm),
            "yaw_deg": float(yaw_deg),
        }


    def _rectify_color_stripwise_remap(self, color, fx_c, fy_c, cx_c, cy_c, t_ref, Hc, Wc):
        """
        Whole-image strip-wise color rectification using cv2.remap.

        Updated version:
        - vectorized per strip (no per-row Python loop)
        - strip stats are computed over the whole strip, not only the last row
        - top/mid/bottom strips are printed together roughly once per second
        - map_x/map_y are clipped before remap to avoid heavy BORDER_REPLICATE artifacts
        - if one strip pose lookup fails, reuse previous valid strip warp
        """
        map_x = np.tile(np.arange(Wc, dtype=np.float32)[None, :], (Hc, 1))
        map_y = np.tile(np.arange(Hc, dtype=np.float32)[:, None], (1, Wc))

        K = np.array([[fx_c, 0.0, cx_c],
                      [0.0, fy_c, cy_c],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        Kinv = np.linalg.inv(K)

        try:
            tf_ref = self.tfbuf.lookup_transform(
                self.world_frame, self.base_frame, t_ref, rospy.Duration(0.1)
            )
            T_ref = tfmsg_to_T(tf_ref)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Rectify ref pose failed: %s" % str(e))
            return color

        strip = max(1, int(self.rs_strip))
        num_strips = int(np.ceil(float(Hc) / float(strip)))
        mid_sid = num_strips // 2
        debug_sids = set([0, mid_sid, num_strips - 1])

        # print all debug strips together about once per second at ~15 fps
        do_debug = ((self.cb_count % 15) == 0)

        if do_debug:
            ref_info = self._summarize_T(T_ref)
            rospy.loginfo(
                "RECT REF | t_ref=%.6f | pose tx=%.3f ty=%.3f tz=%.3f yaw=%.2fdeg",
                t_ref.to_sec(),
                ref_info["tx"], ref_info["ty"], ref_info["tz"], ref_info["yaw_deg"]
            )

        prev_H = None
        prev_have_valid = False

        cols = np.arange(Wc, dtype=np.float64)

        for sid in range(num_strips):
            v0 = sid * strip
            v1 = min((sid + 1) * strip, Hc)
            if v1 <= v0:
                continue

            rows = np.arange(v0, v1, dtype=np.float64)
            v_center = 0.5 * (v0 + v1 - 1)
            t_row = self.row_time(t_ref, v_center, Hc)
            dt_ms = (t_row - t_ref).to_sec() * 1000.0

            fallback_mode = "none"
            delta_info = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "dist_mm": 0.0, "yaw_deg": 0.0}
            row_info = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "dist_mm": 0.0, "yaw_deg": 0.0}

            try:
                tf_row = self.tfbuf.lookup_transform(
                    self.world_frame, self.base_frame, t_row, rospy.Duration(0.1)
                )
                T_row = tfmsg_to_T(tf_row)

                # Delta = np.linalg.inv(T_ref).dot(T_row)
                # R = Delta[:3, :3]
                #
                # H_src_from_dst = np.dot(K, np.dot(R, Kinv))
                # if abs(H_src_from_dst[2, 2]) < 1e-12:
                #     raise ValueError("degenerate homography denominator")
                #
                # H_src_from_dst = H_src_from_dst / H_src_from_dst[2, 2]
                #pingyi
                Delta = np.linalg.inv(T_ref).dot(T_row)
                R = Delta[:3, :3]
                t = Delta[:3, 3]

                # 1) rotation-only homography
                H_rot = np.dot(K, np.dot(R, Kinv))
                if abs(H_rot[2, 2]) < 1e-12:
                    raise ValueError("degenerate homography denominator")
                H_rot = H_rot / H_rot[2, 2]

                # 2) simple image-plane translation induced by 3D translation
                #    assume a reference scene depth Z0
                Z0 = max(float(self.rs_z0), 1e-3)

                du = fx_c * (t[0] / Z0)
                dv = fy_c * (t[1] / Z0)

                T_img = np.array([
                    [1.0, 0.0, du],
                    [0.0, 1.0, dv],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)

                # final strip warp: translation + rotation
                H_src_from_dst = np.dot(T_img, H_rot)
                if abs(H_src_from_dst[2, 2]) < 1e-12:
                    raise ValueError("degenerate final homography denominator")
                H_src_from_dst = H_src_from_dst / H_src_from_dst[2, 2]


                prev_H = H_src_from_dst.copy()
                prev_have_valid = True

                delta_info = self._summarize_T(Delta)
                row_info = self._summarize_T(T_row)

            except Exception as e:
                if prev_have_valid:
                    H_src_from_dst = prev_H
                    fallback_mode = "prev"
                    rospy.logwarn_throttle(
                        1.0,
                        "Rectify row pose failed on sid=%d, reuse previous strip warp: %s" % (sid, str(e))
                    )
                else:
                    H_src_from_dst = np.eye(3, dtype=np.float64)
                    fallback_mode = "identity"
                    rospy.logwarn_throttle(
                        1.0,
                        "Rectify row pose failed on sid=%d, use identity warp: %s" % (sid, str(e))
                    )

            # vectorized mapping for the whole strip
            uu_grid, vv_grid = np.meshgrid(cols, rows)  # shape: (strip_h, Wc)
            n_pix = uu_grid.size

            pix = np.vstack((
                uu_grid.reshape(1, n_pix),
                vv_grid.reshape(1, n_pix),
                np.ones((1, n_pix), dtype=np.float64)
            ))

            src = np.dot(H_src_from_dst, pix)
            w = src[2, :]
            good = np.abs(w) > 1e-12

            # raw source coords before clipping (used for debug stats)
            x_raw = uu_grid.reshape(-1).copy()
            y_raw = vv_grid.reshape(-1).copy()
            x_raw[good] = src[0, good] / w[good]
            y_raw[good] = src[1, good] / w[good]

            x_raw_2d = x_raw.reshape(v1 - v0, Wc).astype(np.float32)
            y_raw_2d = y_raw.reshape(v1 - v0, Wc).astype(np.float32)

            # strip-level stats BEFORE clipping
            out_x = np.mean((x_raw_2d < 0.0) | (x_raw_2d > (Wc - 1)))
            out_y = np.mean((y_raw_2d < 0.0) | (y_raw_2d > (Hc - 1)))

            # clip BEFORE remap, so we do not rely on border replicate too much
            x_clip = np.clip(x_raw_2d, 0.0, float(Wc - 1))
            y_clip = np.clip(y_raw_2d, 0.0, float(Hc - 1))

            map_x[v0:v1, :] = x_clip
            map_y[v0:v1, :] = y_clip

            if do_debug and sid in debug_sids:
                rospy.loginfo(
                    "RECT STRIP sid=%d/%d rows=[%d,%d) dt=%.3fms fallback=%s | "
                    "DELTA dist=%.3fmm yaw=%.3fdeg dtx=%.6f dty=%.6f dtz=%.6f | "
                    "ROW pose tx=%.3f ty=%.3f tz=%.3f yaw=%.2fdeg",
                    sid, num_strips - 1, v0, v1, dt_ms, fallback_mode,
                    delta_info["dist_mm"], delta_info["yaw_deg"],
                    delta_info["tx"], delta_info["ty"], delta_info["tz"],
                    row_info["tx"], row_info["ty"], row_info["tz"], row_info["yaw_deg"]
                )

                rospy.loginfo(
                    "RECT MAP sid=%d | x[min,max]=[%.1f, %.1f] y[min,max]=[%.1f, %.1f] out_x=%.2f%% out_y=%.2f%%",
                    sid,
                    float(np.min(x_raw_2d)), float(np.max(x_raw_2d)),
                    float(np.min(y_raw_2d)), float(np.max(y_raw_2d)),
                    100.0 * float(out_x), 100.0 * float(out_y)
                )

        rectified = cv2.remap(
            color,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rectified

    def cb(self, depth_msg, color_msg, depth_info, color_info):
        self.cb_count += 1
        if self.publish_every > 1 and (self.cb_count % self.publish_every) != 0:
            return

        self._print_cfg_once()

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge decode failed: %s", str(e))
            return

        if depth_msg.encoding == "16UC1":
            z = depth.astype(np.float32) * 0.001
        elif depth_msg.encoding == "32FC1":
            z = depth.astype(np.float32)
        else:
            rospy.logwarn("Unsupported depth encoding: %s", depth_msg.encoding)
            return

        Hd, Wd = z.shape[:2]
        Hc, Wc = color.shape[:2]

        fx_d, fy_d, cx_d, cy_d = K_from_info(depth_info)
        fx_c, fy_c, cx_c, cy_c = K_from_info(color_info)

        color_rect = color
        if self.rs_enable and self.rs_readout_sec > 0.0:
            try:
                color_rect = self._rectify_color_stripwise_remap(
                    color, fx_c, fy_c, cx_c, cy_c,
                    color_msg.header.stamp, Hc, Wc
                )
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Color rectification failed: %s" % str(e))
                color_rect = color

        if self.use_msg_frames:
            depth_frame = depth_msg.header.frame_id
            color_frame = color_msg.header.frame_id
        else:
            depth_frame = self.depth_frame_param
            color_frame = self.color_frame_param

        t_d = depth_msg.header.stamp
        t_c = color_msg.header.stamp

        T_final = self._get_compensated_T(t_d, t_c, depth_frame, color_frame)
        if T_final is not None:
            rospy.loginfo_throttle(5.0, ">>> STATUS: Using COMPENSATED T_final (Qualisys OK)")
        else:
            rospy.logwarn_throttle(5.0, ">>> STATUS: Using STATIC T_cd (Mocap/TF missing)")

        if T_final is None:
            if not self.tfbuf.can_transform(color_frame, depth_frame, rospy.Time(0), rospy.Duration(0.5)):
                rospy.logwarn_throttle(1.0, "TF not ready: %s <- %s", color_frame, depth_frame)
                return
            tf_cd = self.tfbuf.lookup_transform(color_frame, depth_frame, rospy.Time(0), rospy.Duration(0.1))
            T_final = tfmsg_to_T(tf_cd)

        us = np.arange(Wd, dtype=np.float32)
        vs = np.arange(Hd, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        valid = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)

        fly_bad_ratio = 0.0
        if self.fly_enable:
            fly_bad_d, fly_base, fly_bad_n = self._apply_fly_filter_depth(z)
            if fly_base > 0:
                fly_bad_ratio = 100.0 * float(fly_bad_n) / float(fly_base)
            valid = valid & (~fly_bad_d)

        n_valid = int(valid.sum())
        if n_valid < 1000:
            rospy.logwarn("Too few valid depth points: %d", n_valid)
            return

        uu_v = uu[valid]
        vv_v = vv[valid]
        z_v = z[valid]

        Xd = (uu_v - cx_d) * z_v / fx_d
        Yd = (vv_v - cy_d) * z_v / fy_d
        Zd = z_v
        Pd = np.stack([Xd, Yd, Zd, np.ones_like(Zd)], axis=0)

        Pc = np.dot(T_final, Pd)
        Xc = Pc[0, :]
        Yc = Pc[1, :]
        Zc = Pc[2, :]

        uc = (fx_c * (Xc / Zc) + cx_c)
        vc = (fy_c * (Yc / Zc) + cy_c)

        front = (Zc > 0)
        inside = front & (uc >= 0.0) & (uc <= (Wc - 1)) & (vc >= 0.0) & (vc <= (Hc - 1))
        inside_n0 = int(np.count_nonzero(inside))
        if inside_n0 < 1000:
            rospy.logwarn("Too few projected points inside color image: %d", inside_n0)
            return

        uc_in = uc[inside]
        vc_in = vc[inside]
        Xc_in = Xc[inside]
        Yc_in = Yc[inside]
        Zc_in = Zc[inside]

        if self.keep_mode == "floor":
            uc_i = np.floor(uc_in).astype(np.int32)
            vc_i = np.floor(vc_in).astype(np.int32)
        else:
            uc_i = np.round(uc_in).astype(np.int32)
            vc_i = np.round(vc_in).astype(np.int32)

        uc_i = np.clip(uc_i, 0, Wc - 1)
        vc_i = np.clip(vc_i, 0, Hc - 1)

        inside_u = uc_i
        inside_v = vc_i
        insideZ = Zc_in
        inside_n = int(insideZ.shape[0])

        minZ_keep = np.full((Hc, Wc), np.inf, dtype=np.float32)
        np.minimum.at(minZ_keep, (inside_v, inside_u), insideZ)
        keep = insideZ <= (minZ_keep[inside_v, inside_u] + float(self.keep_eps_cloud))

        mapped = np.zeros((Hc, Wc), dtype=np.float32)
        good = np.isfinite(minZ_keep)
        mapped[good] = minZ_keep[good]

        grad_bad_ratio = 0.0
        if self.grad_enable:
            grad_bad, Tgrad, gstats = self._compute_grad_bad(mapped)
            if gstats["base"] > 0:
                grad_bad_ratio = 100.0 * float(gstats["bad"]) / float(gstats["base"])
            mapped[grad_bad] = 0.0
            keep = keep & (~grad_bad[inside_v, inside_u])

        m = mapped.copy()
        m[m <= 0] = np.nan
        try:
            dmin = np.nanpercentile(m, 5)
            dmax = np.nanpercentile(m, 95)
        except Exception:
            dmin, dmax = 0.1, 5.0
        if (not np.isfinite(dmin)) or (not np.isfinite(dmax)) or (dmax <= dmin):
            dmin, dmax = 0.1, 5.0

        d_norm = (np.clip(mapped, dmin, dmax) - dmin) / (dmax - dmin + 1e-9)
        d_u8 = (d_norm * 255.0).astype(np.uint8)
        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)

        try:
            depth_vis_msg = self.bridge.cv2_to_imgmsg(d_color, encoding="bgr8")
            depth_vis_msg.header = color_msg.header
            self.pub_depth_vis.publish(depth_vis_msg)
        except Exception as e:
            rospy.logwarn("Publish depth_vis failed: %s", str(e))

        overlay = cv2.addWeighted(color_rect, 1.0, d_color, self.alpha, 0.0)
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = color_msg.header
            self.pub_overlay.publish(overlay_msg)
        except Exception as e:
            rospy.logwarn("Publish overlay failed: %s", str(e))

        if keep is None or keep.size == 0:
            rospy.logwarn("keep mask empty")
            return

        try:
            color_rect_msg = self.bridge.cv2_to_imgmsg(color_rect, encoding="bgr8")
            color_rect_msg.header = color_msg.header
            self.pub_color_rect.publish(color_rect_msg)
        except Exception as e:
            rospy.logwarn("Publish color_rect failed: %s", str(e))

        uc_k = inside_u[keep]
        vc_k = inside_v[keep]
        Xk = Xc_in[keep]
        Yk = Yc_in[keep]
        Zk = Zc_in[keep]

        if Xk.size == 0:
            rospy.logwarn("All points filtered; check thresholds (eps/grad/fly).")
            return

        rgb = color_rect[vc_k, uc_k, :][:, ::-1].astype(np.uint8)
        xyz = np.stack([Xk, Yk, Zk], axis=1).astype(np.float32)

        r = rgb[:, 0].astype(np.uint32)
        g = rgb[:, 1].astype(np.uint32)
        b = rgb[:, 2].astype(np.uint32)
        rgb_u32 = (r << 16) | (g << 8) | b
        rgb_f32 = rgb_u32.view(np.float32)

        pts = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        pts[:, 0:3] = xyz
        pts[:, 3] = rgb_f32

        header = Header()
        header.stamp = color_msg.header.stamp
        header.frame_id = color_frame

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.FLOAT32, 1),
        ]

        try:
            cloud_msg = pc2.create_cloud(header, fields, list(map(tuple, pts)))
            self.pub_cloud.publish(cloud_msg)
        except Exception as e:
            rospy.logwarn("Publish cloud failed: %s", str(e))

        rospy.loginfo("pub: overlay + depth_vis + cloud | inside=%d -> kept=%d | fly_bad(depth)=%.2f%% grad_bad=%.2f%%",
                      inside_n, int(xyz.shape[0]), fly_bad_ratio, grad_bad_ratio)


if __name__ == "__main__":
    rospy.init_node("publish1_map_depth_to_color", anonymous=True)
    Mapper()
    rospy.spin()