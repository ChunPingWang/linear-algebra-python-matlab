"""
第 12 章：PCA 與 SVD 實務應用（Python 實作）

本腳本示範：
Part A - 主成分分析 (PCA)
    1. 產生一組具有相關性的 2D 模擬資料
    2. 資料中心化（減去平均值）
    3. 計算共變異數矩陣，並對其做特徵分解求主成分方向
    4. 用 SVD 對中心化資料直接求主成分方向，並與特徵分解結果交叉驗證
    5. 繪製原始資料點與主成分方向、投影到第一主成分的降維結果

Part B - SVD 影像/矩陣壓縮
    1. 用外積組合出一個有明顯圖案的簡單「圖像」矩陣（不依賴外部檔案）
    2. 對其做 SVD，並用前 k 個奇異值做低秩近似重建
    3. 比較不同 k 值的重建誤差（Frobenius norm）與壓縮率，驗證誤差隨 k 遞減
    4. 繪製原始矩陣與不同 k 值重建結果的視覺化比較

執行方式：
    python ch12_pca_applications.py
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

# 固定 random seed，確保每次執行結果可重現
np.random.seed(0)


# =============================================================================
# Part A：主成分分析 (PCA)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 產生一組具相關性的 2D 模擬資料
# ---------------------------------------------------------------------------
print("=" * 60)
print("Part A - 1. 產生模擬資料（兩個變數具明顯相關性）")
print("=" * 60)

n_samples = 200

# 先在一個「傾斜」的方向上產生資料，讓兩個變數之間有明顯的線性相關
# 做法：在主軸方向上用較大的變異數，次軸方向用較小的變異數，
#       再旋轉一個角度，讓資料在 x, y 兩個座標軸上呈現相關性
theta = np.radians(30)
rotation = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
])

# 在旋轉前的座標系中，沿著座標軸的標準差分別為 3.0（主軸）與 0.8（次軸）
latent = np.random.randn(n_samples, 2) * np.array([3.0, 0.8])
X = latent @ rotation.T

# 再加上一個非零的平均值，示範「中心化」這一步驟確實有作用
true_mean = np.array([5.0, 2.0])
X = X + true_mean

print("資料矩陣 X 的形狀 (n_samples, n_features) =", X.shape)
print("X 的前 5 筆樣本：")
print(X[:5])
print("X 的平均值（中心化前）=", X.mean(axis=0))


# ---------------------------------------------------------------------------
# 2. 資料中心化（減去平均值）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part A - 2. 資料中心化 (centering)")
print("=" * 60)

X_mean = X.mean(axis=0)
X_centered = X - X_mean

print("平均值 mean(X) =", X_mean)
print("中心化後 X_centered 的平均值（應接近 0）=", X_centered.mean(axis=0))

# 註：是否要再除以標準差（做「標準化 / standardization」）取決於各變數尺度是否
# 相近。本範例兩個變數尺度相近，這裡只做中心化，不做標準差縮放；
# 若變數單位差異很大（例如身高用公分、體重用公斤），建議額外除以各自標準差。


# ---------------------------------------------------------------------------
# 3. 共變異數矩陣與特徵分解
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part A - 3. 共變異數矩陣的特徵分解")
print("=" * 60)

# 共變異數矩陣：C = (1 / (n-1)) * X_centered^T @ X_centered
n = X_centered.shape[0]
cov_matrix = (X_centered.T @ X_centered) / (n - 1)
print("共變異數矩陣 C =")
print(cov_matrix)

# 也可以直接用 np.cov 驗證手算的共變異數矩陣是否正確
cov_matrix_np = np.cov(X_centered, rowvar=False)
print("np.cov 計算結果（驗證）=")
print(cov_matrix_np)
print("兩者是否一致：", np.allclose(cov_matrix, cov_matrix_np))

# 對共變異數矩陣做特徵分解：C v = lambda v
eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # C 為對稱矩陣，用 eigh 更穩定精確

# eigh 回傳的特徵值是「由小到大」排序，PCA 習慣由大到小排序（變異量最大的方向優先）
order = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

print("\n特徵值（由大到小排序）=", eigvals_sorted)
print("特徵向量（每一行 column 對應一個主成分方向）=")
print(eigvecs_sorted)

pc1_eig = eigvecs_sorted[:, 0]
pc2_eig = eigvecs_sorted[:, 1]
print("\n第一主成分方向 PC1（特徵分解法）=", pc1_eig)
print("第二主成分方向 PC2（特徵分解法）=", pc2_eig)

explained_ratio_eig = eigvals_sorted / eigvals_sorted.sum()
print("各主成分解釋的變異量比例 =", explained_ratio_eig)


# ---------------------------------------------------------------------------
# 4. 用 SVD 重做一次 PCA，並與特徵分解結果交叉驗證
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part A - 4. 用 SVD 重做 PCA，並驗證與特徵分解結果一致")
print("=" * 60)

# 對中心化後的資料矩陣直接做 SVD：X_centered = U @ diag(S) @ Vt
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# 關鍵關係：
#   X_centered 的共變異數矩陣 C = X_centered^T X_centered / (n-1)
#            = V @ diag(S^2/(n-1)) @ V^T
#   所以：SVD 的右奇異向量 V（Vt 的每一列）就是共變異數矩陣的特徵向量，
#         且對應的特徵值 = S^2 / (n-1)
pc_svd = Vt.T  # 轉成「每一行 column 對應一個主成分方向」，方便與 eigvecs_sorted 比較
eigvals_from_svd = (S ** 2) / (n - 1)

print("SVD 奇異值 S =", S)
print("由奇異值換算出的特徵值 S^2/(n-1) =", eigvals_from_svd)
print("與特徵分解求得的特徵值比較：", eigvals_sorted)
print("兩者是否一致：", np.allclose(eigvals_from_svd, eigvals_sorted))

pc1_svd = pc_svd[:, 0]
pc2_svd = pc_svd[:, 1]
print("\n第一主成分方向 PC1（SVD 法，未校正正負號）=", pc1_svd)
print("第一主成分方向 PC1（特徵分解法）           =", pc1_eig)


def align_sign(v_ref, v_target):
    """
    特徵向量（或奇異向量）的方向在數學上不唯一，可能相差一個正負號
    （v 與 -v 都是同一個特徵值對應的合法特徵向量）。
    這裡用內積正負號來對齊 v_target 與 v_ref 的方向，方便比較數值是否一致。
    """
    if np.dot(v_ref, v_target) < 0:
        return -v_target
    return v_target


pc1_svd_aligned = align_sign(pc1_eig, pc1_svd)
pc2_svd_aligned = align_sign(pc2_eig, pc2_svd)

print("\n校正正負號後的 PC1（SVD 法）=", pc1_svd_aligned)
print("校正正負號後的 PC2（SVD 法）=", pc2_svd_aligned)

pc1_match = np.allclose(pc1_eig, pc1_svd_aligned)
pc2_match = np.allclose(pc2_eig, pc2_svd_aligned)
print("\nPC1 特徵分解 vs SVD（校正正負號後）是否一致：", pc1_match)
print("PC2 特徵分解 vs SVD（校正正負號後）是否一致：", pc2_match)

if pc1_match and pc2_match:
    print(">>> 驗證通過：特徵分解與 SVD 兩種方法求出的主成分方向一致。")
else:
    print(">>> 驗證失敗：兩種方法結果不一致，請檢查計算過程。")


# ---------------------------------------------------------------------------
# 5. 投影到第一主成分：資料降維
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part A - 5. 投影到第一主成分（2D -> 1D 降維）")
print("=" * 60)

# 用特徵分解得到的 PC1 方向做投影：z = X_centered @ pc1
z_1d = X_centered @ pc1_eig
print("投影到 PC1 後的 1D 座標 z（前 5 筆）=", z_1d[:5])

# 從 1D 座標還原回 2D（近似重建，因為只保留第一主成分，會有資訊損失）
X_reconstructed_1pc = np.outer(z_1d, pc1_eig) + X_mean
reconstruction_error = np.linalg.norm(X - X_reconstructed_1pc, "fro")
print("只用第一主成分重建的 Frobenius 誤差 =", reconstruction_error)
print("原始資料的變異量中，被保留（第一主成分解釋）的比例 =", explained_ratio_eig[0])


# ---------------------------------------------------------------------------
# 6. 繪圖：原始資料點 + 主成分方向 + 投影結果
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part A - 6. 繪製 PCA 圖示")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# 左圖：原始資料點 + 主成分方向
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5, color="tab:blue", label="原始資料點")

# 主成分方向：以平均值為起點，長度依特徵值（標準差）比例縮放，方便觀察相對重要性
scale = 2.0
pc1_end = X_mean + scale * np.sqrt(eigvals_sorted[0]) * pc1_eig
pc2_end = X_mean + scale * np.sqrt(eigvals_sorted[1]) * pc2_eig

ax.annotate("", xy=pc1_end, xytext=X_mean,
            arrowprops=dict(arrowstyle="->", color="tab:red", linewidth=2.5))
ax.annotate("", xy=pc2_end, xytext=X_mean,
            arrowprops=dict(arrowstyle="->", color="tab:green", linewidth=2.5))
ax.plot([], [], color="tab:red", linewidth=2.5, label="PC1（第一主成分，變異最大方向）")
ax.plot([], [], color="tab:green", linewidth=2.5, label="PC2（第二主成分）")
ax.scatter(*X_mean, color="black", marker="x", s=80, label="資料平均值")

ax.set_title("原始資料點與主成分方向")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axis("equal")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="best", fontsize=9)

# 右圖：投影到第一主成分後的降維結果（1D 座標，用 0 當作 y 軸方便視覺化）
ax2 = axes[1]
ax2.scatter(z_1d, np.zeros_like(z_1d), s=15, alpha=0.5, color="tab:red")
ax2.axhline(0, color="gray", linewidth=1)
ax2.set_title("投影到第一主成分後的 1D 降維結果")
ax2.set_xlabel("PC1 座標 z")
ax2.set_yticks([])
ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "pca_projection.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存 PCA 圖示至:", fig_path)


# =============================================================================
# Part B：SVD 影像/矩陣壓縮
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 用外積組合出一個有明顯圖案的簡單「圖像」矩陣
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part B - 1. 生成有結構的簡單圖像矩陣")
print("=" * 60)

size = 100
coords = np.linspace(-3, 3, size)

# 用幾個「外積」疊加出一個有明顯條紋 + 漸層圖案的矩陣，
# 這種做法本質上就是「低秩結構」：矩陣主要是由少數幾個 rank-1 外積成分組成，
# 因此很適合用來示範 SVD 低秩近似的效果。
pattern_1 = np.outer(np.sin(coords), np.cos(coords))          # 棋盤狀條紋
pattern_2 = np.outer(np.exp(-coords ** 2 / 4), np.ones(size))  # 中央亮、邊緣暗的漸層
pattern_3 = np.outer(np.ones(size), np.linspace(-1, 1, size))  # 水平方向線性漸層
pattern_4 = np.outer(np.cos(2 * coords), np.sin(3 * coords))   # 較高頻的細節條紋

# 再加入一點點微小雜訊，讓矩陣不是「精確」低秩，
# 這樣前幾個奇異值之後仍會有（很小但非零的）殘餘誤差，更貼近真實圖像壓縮的情境。
rng = np.random.RandomState(42)
noise = 0.05 * rng.randn(size, size)

image = 3.0 * pattern_1 + 2.0 * pattern_2 + 1.0 * pattern_3 + 0.5 * pattern_4 + noise

print("圖像矩陣形狀 =", image.shape)
print("圖像矩陣的秩（理論上主要由 4 個 rank-1 成分疊加 + 微小雜訊組成，因此數值上接近滿秩，"
      "但奇異值會呈現「前幾個很大、後面快速衰減」的型態）=",
      np.linalg.matrix_rank(image))


# ---------------------------------------------------------------------------
# 2. 對圖像矩陣做 SVD
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part B - 2. 對圖像矩陣做 SVD")
print("=" * 60)

U_img, S_img, Vt_img = np.linalg.svd(image, full_matrices=False)
print("U 形狀 =", U_img.shape, "; S 形狀 =", S_img.shape, "; Vt 形狀 =", Vt_img.shape)
print("前 10 個奇異值 =", S_img[:10])
print("奇異值快速遞減，代表大部分「能量」集中在前幾個成分中，適合做低秩近似。")


# ---------------------------------------------------------------------------
# 3. 用前 k 個奇異值做低秩近似重建，比較誤差與壓縮率
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part B - 3. 低秩近似重建：比較不同 k 值的誤差與壓縮率")
print("=" * 60)


def low_rank_approx(U, S, Vt, k):
    """用前 k 個奇異值/奇異向量重建矩陣 A_k = U_k @ diag(S_k) @ Vt_k"""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


k_values = [1, 2, 3, 5, 10, 20, 50]
m, n_cols = image.shape
full_rank = min(m, n_cols)

errors = []
compression_ratios = []

print(f"{'k':>4} | {'Frobenius 誤差':>16} | {'相對誤差':>10} | {'壓縮率':>10}")
print("-" * 52)
for k in k_values:
    A_k = low_rank_approx(U_img, S_img, Vt_img, k)
    err = np.linalg.norm(image - A_k, "fro")
    rel_err = err / np.linalg.norm(image, "fro")

    # 壓縮率：儲存低秩近似只需存 U_k (m x k)、S_k (k)、Vt_k (k x n)，
    # 相對於原始 m x n 個元素的比例
    storage_k = k * (m + n_cols + 1)
    storage_full = m * n_cols
    compression_ratio = storage_k / storage_full

    errors.append(err)
    compression_ratios.append(compression_ratio)
    print(f"{k:>4} | {err:>16.6f} | {rel_err:>10.4%} | {compression_ratio:>10.4%}")

# 驗證誤差是否隨 k 遞增而單調遞減
errors_arr = np.array(errors)
is_monotonic_decreasing = np.all(np.diff(errors_arr) <= 1e-10)
print()
if is_monotonic_decreasing:
    print(">>> 驗證通過：重建誤差（Frobenius norm）隨 k 增加而單調遞減（或持平）。")
else:
    print(">>> 驗證失敗：重建誤差沒有隨 k 增加而遞減，請檢查計算過程。")

# 額外驗證：k = full_rank 時應完全重建（誤差趨近於 0）
A_full = low_rank_approx(U_img, S_img, Vt_img, full_rank)
full_rank_error = np.linalg.norm(image - A_full, "fro")
print(f"k = full_rank ({full_rank}) 時的重建誤差 = {full_rank_error:.2e}（應接近 0）")


# ---------------------------------------------------------------------------
# 4. 繪圖：原始矩陣 vs 不同 k 值的重建結果
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Part B - 4. 繪製原始矩陣與低秩近似重建結果")
print("=" * 60)

k_to_plot = [1, 3, 10, 50]
vmin, vmax = image.min(), image.max()

fig, axes = plt.subplots(1, len(k_to_plot) + 1, figsize=(4 * (len(k_to_plot) + 1), 4.2))

im0 = axes[0].imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("原始矩陣（滿秩）")
axes[0].axis("off")

for ax, k in zip(axes[1:], k_to_plot):
    A_k = low_rank_approx(U_img, S_img, Vt_img, k)
    err = np.linalg.norm(image - A_k, "fro")
    ax.imshow(A_k, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"k = {k}\n誤差 = {err:.3f}")
    ax.axis("off")

fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.02)
fig_path2 = os.path.join(OUT_DIR, "svd_compression.png")
plt.savefig(fig_path2, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存 SVD 低秩近似比較圖至:", fig_path2)

# 額外繪製「誤差 vs k」的折線圖，直觀呈現遞減趨勢
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_values, errors, marker="o", color="tab:purple")
ax.set_title("重建誤差（Frobenius norm）隨 k 增加而遞減")
ax.set_xlabel("保留的奇異值個數 k")
ax.set_ylabel("Frobenius 誤差 ||A - A_k||_F")
ax.grid(True, linestyle="--", alpha=0.5)
fig_path3 = os.path.join(OUT_DIR, "svd_error_curve.png")
plt.savefig(fig_path3, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存誤差遞減曲線圖至:", fig_path3)

print()
print("全部範例執行完畢。")
