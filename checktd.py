import pickle, numpy as np
db = pickle.load(open("slow_raw.pkl", "rb"))
Ts = np.array([r["T_bd"] for r in db]).reshape(len(db), -1)
print("T_bd max variation:", np.ptp(Ts, axis=0).max())
# 顺手看它随时间的样子:
yaws = [np.degrees(np.arctan2(r["T_bd"][1,0], r["T_bd"][0,0])) for r in db]
print("T_bd yaw range (deg):", min(yaws), max(yaws))