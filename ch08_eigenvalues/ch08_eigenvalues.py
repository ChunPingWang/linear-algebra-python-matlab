"""
第 8 章：特徵值與特徵向量（Python 實作）

本腳本示範：
1. 特徵值/特徵向量的定義 Av = λv，並用 numpy.linalg.eig 求解
2. 驗證 A @ v ≈ λ * v（np.allclose）
3. 特徵值的性質：所有特徵值之和 = trace(A)、所有特徵值之積 = det(A)
4. 對角化 A = P D P^-1，並驗證 P @ D @ inv(P) ≈ A
5. 幾何意義：畫出特徵向量方向在矩陣變換下不改變、只被縮放

執行方式：
    python ch08_eigenvalues.py
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
# 1. 特徵值與特徵向量的定義：Av = λv
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 特徵值與特徵向量的定義：Av = lambda * v")
print("=" * 60)

A = np.array([[4.0, 1.0],
              [2.0, 3.0]])
print("矩陣 A =")
print(A)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n用 numpy.linalg.eig(A) 求得：")
print("特徵值 (eigenvalues) =", eigenvalues)
print("特徵向量矩陣 (每一行 column 對應一個特徵向量) =")
print(eigenvectors)
print("\n註：numpy.linalg.eig 對一般矩陣採用通用演算法，即使特徵值皆為實數，")
print("回傳的 dtype 仍可能是 complex128（虛部為 0）。本例特徵值虛部皆為 0，")
print("可以用 .real 取出實數部分，方便閱讀：")
eigenvalues_real = eigenvalues.real
print("eigenvalues.real =", eigenvalues_real)


# ---------------------------------------------------------------------------
# 2. 驗證 A @ v ≈ λ * v
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 驗證 A @ v ≈ lambda * v")
print("=" * 60)

for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    lhs = A @ v          # A v
    rhs = lam * v        # lambda v
    ok = np.allclose(lhs, rhs)
    print(f"\n第 {i+1} 組: lambda_{i+1} = {lam:.6f}")
    print(f"  v_{i+1} = {v}")
    print(f"  A @ v_{i+1}      = {lhs}")
    print(f"  lambda_{i+1} * v_{i+1} = {rhs}")
    print(f"  np.allclose(A@v, lambda*v) = {ok}  {'-> 驗證通過' if ok else '-> 驗證失敗'}")


# ---------------------------------------------------------------------------
# 3. 手算範例對照：2x2 矩陣完整解特徵方程式
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 手算範例對照：det(A - lambda*I) = 0")
print("=" * 60)

# 手算過程（見 .md 教學文件）得到：
#   特徵方程式: lambda^2 - 7*lambda + 10 = 0 -> (lambda-5)(lambda-2)=0
#   lambda_1 = 5, 對應特徵向量方向 (1, 1)
#   lambda_2 = 2, 對應特徵向量方向 (1, -2)
hand_eigenvalues = np.array([5.0, 2.0])
hand_v1 = np.array([1.0, 1.0])
hand_v2 = np.array([1.0, -2.0])

print("手算特徵值：", sorted(hand_eigenvalues, reverse=True))
print("numpy 求得特徵值（排序後）：", sorted(eigenvalues, reverse=True))
print("兩者是否一致：", np.allclose(sorted(eigenvalues, reverse=True),
                                 sorted(hand_eigenvalues, reverse=True)))

# 驗證手算特徵向量也滿足 Av = λv（方向相同即可，特徵向量可任意縮放）
print("\n驗證手算特徵向量 v1=(1,1) 對應 lambda=5:")
print("  A @ v1 =", A @ hand_v1, " ; 5 * v1 =", 5 * hand_v1)
print("  是否相等:", np.allclose(A @ hand_v1, 5 * hand_v1))

print("\n驗證手算特徵向量 v2=(1,-2) 對應 lambda=2:")
print("  A @ v2 =", A @ hand_v2, " ; 2 * v2 =", 2 * hand_v2)
print("  是否相等:", np.allclose(A @ hand_v2, 2 * hand_v2))


# ---------------------------------------------------------------------------
# 4. 特徵值的性質：和 = trace(A)，積 = det(A)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 特徵值的性質：sum(eigenvalues) = trace(A), prod(eigenvalues) = det(A)")
print("=" * 60)

trace_A = np.trace(A)
det_A = np.linalg.det(A)
sum_eig = np.sum(eigenvalues)
prod_eig = np.prod(eigenvalues)

print("trace(A) =", trace_A)
print("sum(特徵值) =", sum_eig)
print("兩者是否相等(allclose):", np.allclose(trace_A, sum_eig))

print()
print("det(A) =", det_A)
print("prod(特徵值) =", prod_eig)
print("兩者是否相等(allclose):", np.allclose(det_A, prod_eig))


# ---------------------------------------------------------------------------
# 5. 對角化 A = P D P^-1
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. 對角化 A = P D P^-1")
print("=" * 60)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv

print("P (特徵向量組成的矩陣) =")
print(P)
print("\nD (特徵值對角矩陣) =")
print(D)
print("\nP^-1 =")
print(P_inv)

print("\n重建 P @ D @ P^-1 =")
print(A_reconstructed)
print("\n原矩陣 A =")
print(A)

is_diagonalizable_check = np.allclose(A_reconstructed, A)
print("\nnp.allclose(P @ D @ P^-1, A) =", is_diagonalizable_check,
      "  -> 對角化驗證通過" if is_diagonalizable_check else "  -> 對角化驗證失敗")

# 說明何時矩陣可對角化：n 個線性獨立的特徵向量（等價於 P 為可逆矩陣）
rank_P = np.linalg.matrix_rank(P)
n = A.shape[0]
print(f"\nP 的秩 (rank) = {rank_P}，矩陣大小 n = {n}")
print("秩 = n，代表 2 個特徵向量線性獨立，P 可逆，因此 A 可對角化。")


# ---------------------------------------------------------------------------
# 6. 對角化的應用範例：不可對角化的矩陣（缺特徵向量）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. 反例：不可對角化的矩陣（重根但特徵向量不足）")
print("=" * 60)

B = np.array([[2.0, 1.0],
              [0.0, 2.0]])
eig_B, vec_B = np.linalg.eig(B)
rank_vec_B = np.linalg.matrix_rank(vec_B)
print("矩陣 B =")
print(B)
print("特徵值 =", eig_B, " (重根 lambda=2)")
print("特徵向量矩陣 P_B 的秩 =", rank_vec_B, "（< 2，只有 1 個線性獨立的特徵向量）")
print("因此 B 不可對角化（defective matrix）。")


# ---------------------------------------------------------------------------
# 7. 幾何意義：畫出矩陣變換下，特徵向量方向不變、只被縮放
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("7. 繪製幾何意義示意圖")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左圖：一般向量在 A 變換下方向改變
ax = axes[0]
general_vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
colors_gen = ["tab:blue", "tab:orange", "tab:purple"]
origin = np.array([0.0, 0.0])
for v, c in zip(general_vectors, colors_gen):
    Av = A @ v
    ax.quiver(*origin, *v, angles="xy", scale_units="xy", scale=1,
              color=c, alpha=0.4, width=0.01)
    ax.quiver(*origin, *Av, angles="xy", scale_units="xy", scale=1,
              color=c, width=0.01)
ax.set_xlim(-2, 6)
ax.set_ylim(-2, 6)
ax.set_aspect("equal")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_title("一般向量：經 A 變換後方向改變（淺色=原向量，深色=變換後）")
ax.set_xlabel("x")
ax.set_ylabel("y")

# 右圖：特徵向量在 A 變換下方向不變，只被縮放
ax2 = axes[1]
eig_dirs = [hand_v1 / np.linalg.norm(hand_v1), hand_v2 / np.linalg.norm(hand_v2)]
eig_lams = [5.0, 2.0]
colors_eig = ["tab:green", "tab:red"]
for v, lam, c in zip(eig_dirs, eig_lams, colors_eig):
    Av = A @ v
    ax2.quiver(*origin, *v, angles="xy", scale_units="xy", scale=1,
               color=c, alpha=0.4, width=0.012, label=f"v (lambda={lam:.0f}的特徵向量方向)")
    ax2.quiver(*origin, *Av, angles="xy", scale_units="xy", scale=1,
               color=c, width=0.012, label=f"A v = {lam:.0f} v（同方向，被放大 {lam:.0f} 倍）")
ax2.set_xlim(-3, 6)
ax2.set_ylim(-3, 6)
ax2.set_aspect("equal")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_title("特徵向量：經 A 變換後方向不變，只被縮放 lambda 倍")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(loc="upper left", fontsize=8)

fig_path = os.path.join(OUT_DIR, "eigenvectors_geometry.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存特徵向量幾何意義示意圖至:", fig_path)

print()
print("全部範例執行完畢。")
