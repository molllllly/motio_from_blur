import matplotlib.pyplot as plt
import numpy as np
import csv

def read_csv(path):
    data = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            data.append([float(row['x']), float(row['y']), float(row['yaw_deg'])])
    return np.array(data)

slow = read_csv("slow_positions.csv")
fast = read_csv("fast_positions.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：XY 轨迹
axes[0].plot(slow[:, 0], slow[:, 1], 'b.-', alpha=0.3, markersize=1, label='slow')
axes[0].plot(fast[:, 0], fast[:, 1], 'r.-', alpha=0.5, markersize=2, label='fast')
axes[0].axis('equal')
axes[0].legend()
axes[0].set_title("XY trajectory")
axes[0].grid()

# 右图：yaw vs time（或 yaw vs index）
axes[1].plot(np.arange(len(slow)), slow[:, 2], 'b-', alpha=0.5, label='slow')
axes[1].plot(np.arange(len(fast)), fast[:, 2], 'r-', alpha=0.5, label='fast')
axes[1].legend()
axes[1].set_title("Yaw over index")
axes[1].set_ylabel("Yaw (deg)")
axes[1].grid()

plt.tight_layout()
plt.savefig("trajectory_compare.png", dpi=120)
print("Saved trajectory_compare.png")