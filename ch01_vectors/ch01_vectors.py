"""
第 1 章：向量基礎（Python 實作）

本腳本示範：
1. 純量與向量的建立（行向量／列向量）
2. 向量加法、純量乘法（含幾何圖示）
3. 向量的範數（L1、L2、L∞）
4. 內積、夾角、正交性判斷
5. 單位向量（normalize）
6. 向量投影（projection）

執行方式：
    python ch01_vectors.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # 無顯示器環境，僅存檔不顯示

import matplotlib.pyplot as plt
import numpy as np

# 設定中文字型，避免圖片中的中文顯示為方框（找不到字型時則沿用預設字型）
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang TC", "PingFang SC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 圖片輸出資料夾（本章資料夾）
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. 純量與向量
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 純量與向量")
print("=" * 60)

scalar = 3.0
print("純量 s =", scalar)

# 行向量（column vector）：在 numpy 中常用形狀 (n, 1) 表示
col_vec = np.array([[2], [3]])
print("行向量 v (column vector), shape =", col_vec.shape)
print(col_vec)

# 列向量（row vector）：在 numpy 中常用形狀 (1, n) 或直接一維陣列表示
row_vec = np.array([2, 3])
print("列向量 v (row vector), shape =", row_vec.shape)
print(row_vec)


# ---------------------------------------------------------------------------
# 2. 向量加法與純量乘法
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 向量加法與純量乘法")
print("=" * 60)

a = np.array([2, 1])
b = np.array([1, 3])

a_plus_b = a + b
print("向量 a =", a)
print("向量 b =", b)
print("向量 a + b =", a_plus_b)

k = 2
k_times_a = k * a
print(f"純量乘法 {k} * a =", k_times_a)


# ---------------------------------------------------------------------------
# 3. 向量的長度／範數（L1、L2、L∞）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 向量的長度／範數")
print("=" * 60)

v = np.array([3, -4])
print("向量 v =", v)

l1_norm = np.linalg.norm(v, ord=1)
l2_norm = np.linalg.norm(v, ord=2)
linf_norm = np.linalg.norm(v, ord=np.inf)

print("L1 範數 ||v||_1 =", l1_norm)
print("L2 範數 ||v||_2 =", l2_norm)
print("L∞ 範數 ||v||_inf =", linf_norm)


# ---------------------------------------------------------------------------
# 4. 內積、夾角與正交性
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 內積、夾角與正交性")
print("=" * 60)

u = np.array([1, 2])
w = np.array([3, -1])

dot_uw = np.dot(u, w)
print("向量 u =", u)
print("向量 w =", w)
print("內積 u . w =", dot_uw)

norm_u = np.linalg.norm(u)
norm_w = np.linalg.norm(w)
cos_theta = dot_uw / (norm_u * norm_w)
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)

print("|u| =", norm_u)
print("|w| =", norm_w)
print("cos(theta) =", cos_theta)
print("theta (degrees) =", theta_deg)

# 正交性判斷：內積是否為 0
p = np.array([1, 0])
q = np.array([0, 1])
dot_pq = np.dot(p, q)
print()
print("向量 p =", p, "、向量 q =", q)
print("p . q =", dot_pq, "-> p 與 q 正交" if np.isclose(dot_pq, 0) else "-> p 與 q 不正交")


# ---------------------------------------------------------------------------
# 5. 單位向量（normalize）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. 單位向量（normalize）")
print("=" * 60)

x = np.array([3, 4])
x_norm = np.linalg.norm(x)
x_unit = x / x_norm

print("向量 x =", x)
print("|x| =", x_norm)
print("單位向量 x_hat = x / |x| =", x_unit)
print("驗證 |x_hat| =", np.linalg.norm(x_unit))


# ---------------------------------------------------------------------------
# 6. 向量投影（projection of a onto b）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. 向量投影（projection of a onto b）")
print("=" * 60)

a2 = np.array([3, 2])
b2 = np.array([4, 0])

scalar_proj = np.dot(a2, b2) / np.linalg.norm(b2)
proj_vec = (np.dot(a2, b2) / np.dot(b2, b2)) * b2

print("向量 a =", a2)
print("向量 b =", b2)
print("純量投影 (scalar projection) =", scalar_proj)
print("向量投影 proj_b(a) =", proj_vec)


# ---------------------------------------------------------------------------
# 7. 幾何圖示：向量加法
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("7. 繪製向量加法幾何圖")
print("=" * 60)

fig, ax = plt.subplots(figsize=(6, 6))

origin = np.array([0, 0])

# 畫 a、b、a+b 三個向量
ax.quiver(*origin, *a, angles="xy", scale_units="xy", scale=1,
          color="tab:blue", label="a = (2, 1)")
ax.quiver(*a, *b, angles="xy", scale_units="xy", scale=1,
          color="tab:orange", label="b（從 a 端點畫出，平移後的 b）")
ax.quiver(*origin, *a_plus_b, angles="xy", scale_units="xy", scale=1,
          color="tab:green", label="a + b = (3, 4)")

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("向量加法：a + b（平行四邊形法則）")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left")

fig_path = os.path.join(OUT_DIR, "vector_addition.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存向量加法示意圖至:", fig_path)


# ---------------------------------------------------------------------------
# 8. 幾何圖示：向量投影
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

ax.quiver(*origin, *b2, angles="xy", scale_units="xy", scale=1,
          color="tab:orange", label="b = (4, 0)")
ax.quiver(*origin, *a2, angles="xy", scale_units="xy", scale=1,
          color="tab:blue", label="a = (3, 2)")
ax.quiver(*origin, *proj_vec, angles="xy", scale_units="xy", scale=1,
          color="tab:green", label="proj_b(a)")

# 畫出 a 到投影點的垂直虛線
ax.plot([a2[0], proj_vec[0]], [a2[1], proj_vec[1]],
        linestyle="--", color="gray", label="垂直於 b 的分量")

ax.set_xlim(-1, 5)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("向量投影：a 投影到 b 上")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left")

fig_path2 = os.path.join(OUT_DIR, "vector_projection.png")
plt.savefig(fig_path2, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存向量投影示意圖至:", fig_path2)

print()
print("全部範例執行完畢。")
