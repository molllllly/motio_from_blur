# Motion from Blur

Code from my master's thesis at Linköping University, investigating camera motion estimation from motion blur.

This repository contains tools for RGB-D processing, trajectory analysis, and motion-blur ground-truth generation using RealSense recordings and motion-capture data.

## Main Features

- Project depth from slow recordings into fast-motion RGB frames.
- Account for rolling-shutter timing during projection.
- Generate smear-path labels using depth and camera motion.
- Extract, compare, and visualize motion-capture trajectories.

## Main Files

- `slow_to_fast_.py`  — depth projection and smear-label generation.
- `publish2.py` — RGB-D alignment and visualization.
- `extract_trajectories_csv.py` — extract trajectories from ROS bags.
- `export_analysis.py` — trajectory interpolation and plotting.
- `util_smear/` — PFM file utilities.

## Requirements

Python, ROS1, NumPy, OpenCV, Matplotlib, and ROS packages including `rosbag`, `cv_bridge`, and `tf2_ros`.

The original ROS bag recordings are not included. File paths, ROS topics, camera calibration, and timing parameters need to be configured for your recordings.

**Scope:** This repository focuses on data preparation and geometric processing for the thesis.
