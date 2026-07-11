"""
第 10 章：奇異值分解（SVD）（Python 實作）

本腳本示範：
1. 用 numpy.linalg.svd 對矩陣做奇異值分解，並驗證 U @ Sigma @ V^T ≈ A
2. SVD 與 A^T A、A A^T 特徵分解的關係
3. 用 SVD 判斷矩陣的秩（rank）
4. Moore-Penrose 偽逆（pseudo-inverse）：用 SVD 手算並與 np.linalg.pinv 比較
5. 幾何意義：2x2 矩陣如何把單位圓映射成橢圓（旋轉 -> 縮放 -> 旋轉）

執行方式：
    python ch10_svd.py
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

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. SVD 的定義：A = U Sigma V^T
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. SVD 的定義：A = U Sigma V^T")
print("=" * 60)

# 用一個 3x2（長方形）矩陣示範一般情況
A = np.array([
    [3.0, 0.0],
    [4.0, 5.0],
    [0.0, 0.0],
])
print("矩陣 A (3x2) =")
print(A)

# full_matrices=True: U 是 m x m、Vt 是 n x n（完整版 SVD）
U, s, Vt = np.linalg.svd(A, full_matrices=True)
print("\nU (3x3, 正交矩陣) =")
print(U)
print("\n奇異值 s (由大到小排列) =", s)
print("\nV^T (2x2, 正交矩陣) =")
print(Vt)

# 把一維奇異值陣列 s 組裝成 m x n 的對角矩陣 Sigma
m, n = A.shape
Sigma = np.zeros((m, n))
Sigma[: len(s), : len(s)] = np.diag(s)
print("\nSigma (3x2, 對角矩陣) =")
print(Sigma)

# 驗證 A = U @ Sigma @ V^T
A_reconstructed = U @ Sigma @ Vt
print("\n重建 U @ Sigma @ V^T =")
print(A_reconstructed)
print("驗證 U @ Sigma @ V^T ≈ A ?", np.allclose(A_reconstructed, A))
assert np.allclose(A_reconstructed, A), "SVD 重建失敗！"
print("[通過] U @ Sigma @ V^T 與原矩陣 A 一致。")

# 順便驗證 U、V 皆為正交矩陣：U^T U = I、V^T V = I
V = Vt.T
print("\n驗證 U 為正交矩陣 (U^T U ≈ I) ?", np.allclose(U.T @ U, np.eye(m)))
print("驗證 V 為正交矩陣 (V^T V ≈ I) ?", np.allclose(V.T @ V, np.eye(n)))


# ---------------------------------------------------------------------------
# 2. SVD 與特徵值分解的關係
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. SVD 與特徵值分解的關係")
print("=" * 60)

# A^T A 是 n x n 的對稱矩陣，其特徵值就是奇異值的平方，特徵向量就是 V 的欄
AtA = A.T @ A
eigvals_AtA, eigvecs_AtA = np.linalg.eigh(AtA)  # eigh: 專用於對稱矩陣，回傳由小到大排序
# 由大到小重新排序，才能與奇異值的順序對齊
order = np.argsort(eigvals_AtA)[::-1]
eigvals_AtA = eigvals_AtA[order]
eigvecs_AtA = eigvecs_AtA[:, order]

print("A^T A =")
print(AtA)
print("\nA^T A 的特徵值 (由大到小) =", eigvals_AtA)
print("奇異值的平方 s^2 =", s ** 2)
print("驗證：特徵值 ≈ 奇異值平方 ?", np.allclose(eigvals_AtA, s ** 2))

# AA^T 是 m x m 的對稱矩陣，非零特徵值同樣是奇異值的平方，特徵向量就是 U 的欄
AAt = A @ A.T
eigvals_AAt, eigvecs_AAt = np.linalg.eigh(AAt)
order2 = np.argsort(eigvals_AAt)[::-1]
eigvals_AAt = eigvals_AAt[order2]
eigvecs_AAt = eigvecs_AAt[:, order2]

print("\nAA^T 的特徵值 (由大到小) =", eigvals_AAt)
print("（前 len(s) 個應與奇異值平方一致，其餘應接近 0，因為 rank(A) <= min(m, n)）")

# 特徵向量與 U、V 的欄位方向可能差一個正負號，比較時取絕對值
print("\nV 的欄（|V|）=")
print(np.abs(V))
print("A^T A 的特徵向量（|eigvecs_AtA|，取前 len(s) 欄）=")
print(np.abs(eigvecs_AtA[:, : len(s)]))
print("兩者是否一致（取絕對值比較）？",
      np.allclose(np.abs(V[:, : len(s)]), np.abs(eigvecs_AtA[:, : len(s)])))


# ---------------------------------------------------------------------------
# 3. 手算範例：2x2 矩陣完整算出 U, Sigma, V
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 手算範例：2x2 矩陣的 SVD")
print("=" * 60)

# 這個矩陣的 B^T B 特徵值恰為簡單數字，適合手算
B = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
])
print("矩陣 B (2x2) =")
print(B)

BtB = B.T @ B
print("\nB^T B =")
print(BtB)

eigvals_B, eigvecs_B = np.linalg.eigh(BtB)
order_B = np.argsort(eigvals_B)[::-1]
eigvals_B = eigvals_B[order_B]
eigvecs_B = eigvecs_B[:, order_B]
singular_values_B = np.sqrt(eigvals_B)

print("\nB^T B 的特徵值 (由大到小) =", eigvals_B)
print("手算奇異值 = sqrt(特徵值) =", singular_values_B)

U_B, s_B, Vt_B = np.linalg.svd(B)
print("\nnumpy 計算的奇異值 s =", s_B)
print("驗證手算奇異值與 numpy 結果一致？", np.allclose(singular_values_B, s_B))


# ---------------------------------------------------------------------------
# 4. 用 SVD 計算矩陣的秩
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 用 SVD 計算矩陣的秩 (rank)")
print("=" * 60)

# 建立一個明顯秩不足的矩陣：第二列是第一列的 2 倍
C = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [1.0, 0.0, 1.0],
])
print("矩陣 C (3x3) =")
print(C)

_, s_C, _ = np.linalg.svd(C)
print("\n奇異值 s =", s_C)

tol = 1e-10
rank_from_svd = int(np.sum(s_C > tol))
rank_numpy = np.linalg.matrix_rank(C)

print(f"非零奇異值個數（容忍誤差 {tol}）=", rank_from_svd)
print("np.linalg.matrix_rank(C) =", rank_numpy)
print("兩者是否一致？", rank_from_svd == rank_numpy)
assert rank_from_svd == rank_numpy, "秩的計算不一致！"
print("[通過] 用 SVD 求出的秩與 numpy 內建函式一致，rank(C) =", rank_numpy)


# ---------------------------------------------------------------------------
# 5. Moore-Penrose 偽逆（pseudo-inverse）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. Moore-Penrose 偽逆 (pseudo-inverse)")
print("=" * 60)

# 用第 1 節的長方形矩陣 A (3x2, 非方陣) 來示範偽逆
print("矩陣 A (3x2) =")
print(A)

# 用 SVD 手算偽逆：A^+ = V Sigma^+ U^T
# Sigma^+ 是把 Sigma 的非零元素取倒數後轉置 (n x m)
Sigma_pinv = np.zeros((n, m))
for i in range(len(s)):
    if s[i] > tol:
        Sigma_pinv[i, i] = 1.0 / s[i]

A_pinv_manual = V @ Sigma_pinv @ U.T
print("\n手算偽逆 A^+ = V Sigma^+ U^T =")
print(A_pinv_manual)

A_pinv_numpy = np.linalg.pinv(A)
print("\nnp.linalg.pinv(A) =")
print(A_pinv_numpy)

print("\n驗證手算偽逆與 np.linalg.pinv 一致？", np.allclose(A_pinv_manual, A_pinv_numpy))
assert np.allclose(A_pinv_manual, A_pinv_numpy), "偽逆計算不一致！"
print("[通過] 手算（SVD）偽逆與 np.linalg.pinv 結果一致。")

# 偽逆的用途：非方陣或不可逆矩陣時求「最小二乘解」 x = A^+ b（第 11 章會深入介紹）
b = np.array([1.0, 2.0, 3.0])
x_lstsq = A_pinv_numpy @ b
print("\n以 b =", b, "為例，最小二乘解 x = A^+ b =", x_lstsq)
x_check, *_ = np.linalg.lstsq(A, b, rcond=None)
print("np.linalg.lstsq(A, b) 的解 =", x_check)
print("兩者是否一致？", np.allclose(x_lstsq, x_check))
assert np.allclose(x_lstsq, x_check), "最小二乘解不一致！"
print("[通過] 用偽逆求出的最小二乘解與 np.linalg.lstsq 一致。")

# 也驗證偽逆滿足四個 Moore-Penrose 條件之一：A A^+ A = A
print("\n驗證 A A^+ A ≈ A ?", np.allclose(A @ A_pinv_numpy @ A, A))


# ---------------------------------------------------------------------------
# 6. 幾何意義：2x2 矩陣如何把單位圓映射成橢圓
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. 幾何意義：單位圓 -> 橢圓（旋轉 -> 縮放 -> 旋轉）")
print("=" * 60)

# 選一個容易觀察「旋轉+縮放」效果的 2x2 矩陣
M = np.array([
    [3.0, 1.0],
    [1.0, 2.0],
])
print("矩陣 M (2x2) =")
print(M)

U_M, s_M, Vt_M = np.linalg.svd(M)
V_M = Vt_M.T
print("\nU =")
print(U_M)
print("奇異值 s =", s_M)
print("V =")
print(V_M)

# 產生單位圓上的點
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(theta), np.sin(theta)])  # shape (2, 200)

# 映射：每個圓上的點 x 經過 M 變成 Mx，畫出來會是橢圓
ellipse = M @ circle

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左圖：原本的單位圓，並畫出 V 的兩個正交方向（輸入端的主軸）
ax = axes[0]
ax.plot(circle[0], circle[1], color="tab:blue", label="單位圓")
for i in range(2):
    vec = V_M[:, i]
    ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1,
               color=f"C{i+1}", label=f"v{i+1} 方向 (V 第 {i+1} 欄)")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("映射前：單位圓與 V 的正交方向")
ax.legend(loc="upper left", fontsize=9)

# 右圖：映射後的橢圓，並畫出 U 的方向乘上對應奇異值（輸出端的主軸半長就是奇異值）
ax = axes[1]
ax.plot(ellipse[0], ellipse[1], color="tab:blue", label="M 映射後的橢圓")
for i in range(2):
    vec = U_M[:, i] * s_M[i]
    ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1,
               color=f"C{i+1}", label=f"sigma{i+1} * u{i+1}（奇異值={s_M[i]:.3f}）")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("映射後：橢圓與主軸（長度 = 奇異值）")
ax.legend(loc="upper left", fontsize=9)

fig.suptitle("SVD 的幾何意義：旋轉 -> 縮放 -> 旋轉（單位圓變橢圓）")
fig_path = os.path.join(OUT_DIR, "svd_circle_to_ellipse.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("\n已儲存單位圓映射成橢圓示意圖至:", fig_path)

print()
print("全部範例執行完畢。")
