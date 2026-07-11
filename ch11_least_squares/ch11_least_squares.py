"""
第 11 章：最小二乘法與線性迴歸（Python 實作）

本腳本示範：
1. 超定方程組 (overdetermined system)：資料點數量多於未知參數，Ax=b 通常無解
2. 正規方程式 (normal equation) 手刻求解：A^T A x = A^T b
3. 用 QR 分解求解最小二乘（數值上更穩定）
4. 用 np.linalg.lstsq 求解，並與前兩種方法交叉驗證結果一致
5. 模擬帶雜訊的線性資料 y = 2x + 1 + noise，求出最佳擬合直線
6. 繪製資料點散佈圖與擬合直線（含圖示）

執行方式：
    python ch11_least_squares.py
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
# 1. 超定方程組：一個簡單的手算範例
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 超定方程組 (overdetermined system) 範例")
print("=" * 60)

# 3 個方程式、2 個未知數（x1, x2），一般沒有精確解
A_demo = np.array([
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0],
])
b_demo = np.array([2.0, 3.0, 5.0])

print("矩陣 A (3x2, 3 個方程式、2 個未知數) =")
print(A_demo)
print("向量 b =", b_demo)
print(
    "因為方程式數量 (3) 大於未知數數量 (2)，一般而言 Ax=b 無精確解，"
    "只能求「最小平方誤差」意義下的最佳近似解。"
)

x_demo, residuals_demo, rank_demo, sv_demo = np.linalg.lstsq(A_demo, b_demo, rcond=None)
print("最小二乘解 x =", x_demo)
print("殘差平方和 ||Ax-b||^2 =", np.sum((A_demo @ x_demo - b_demo) ** 2))


# ---------------------------------------------------------------------------
# 2. 模擬帶雜訊的線性資料 y = 2x + 1 + noise
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 模擬資料：y = 2x + 1 + noise")
print("=" * 60)

np.random.seed(42)  # 固定隨機種子，確保結果可重現

n_points = 30
x_data = np.linspace(0, 10, n_points)
true_slope = 2.0
true_intercept = 1.0
noise = np.random.normal(loc=0.0, scale=1.5, size=n_points)
y_data = true_slope * x_data + true_intercept + noise

print(f"資料點數量: {n_points}")
print(f"真實斜率 (true slope) = {true_slope}, 真實截距 (true intercept) = {true_intercept}")
print("前 5 筆 (x, y):")
for i in range(5):
    print(f"  ({x_data[i]:.4f}, {y_data[i]:.4f})")

# 將問題寫成矩陣形式 A x = b
# A 的欄為 [x, 1]，x = [斜率 m, 截距 c]^T，b = y
A = np.column_stack([x_data, np.ones(n_points)])
b = y_data

print()
print("矩陣形式 A x = b：")
print(f"A 的形狀 = {A.shape} (每列是 [x_i, 1])，b 的形狀 = {b.shape}")
print("A 的前 5 列 =")
print(A[:5])


# ---------------------------------------------------------------------------
# 3. 方法一：正規方程式 (normal equation) 手刻求解
#    A^T A x = A^T b  =>  x = (A^T A)^{-1} A^T b
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 方法一：正規方程式 (normal equation) 手刻求解")
print("=" * 60)

AtA = A.T @ A
Atb = A.T @ b
print("A^T A =")
print(AtA)
print("A^T b =", Atb)

x_normal = np.linalg.inv(AtA) @ Atb
slope_normal, intercept_normal = x_normal
print(f"正規方程式解得：斜率 m = {slope_normal:.6f}, 截距 c = {intercept_normal:.6f}")


# ---------------------------------------------------------------------------
# 4. 方法二：QR 分解求解最小二乘
#    A = QR  =>  A^T A x = A^T b  可化簡為  R x = Q^T b（數值上更穩定）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 方法二：QR 分解求解最小二乘")
print("=" * 60)

Q, R = np.linalg.qr(A)
print(f"Q 的形狀 = {Q.shape}, R 的形狀 = {R.shape}")
print("R (上三角矩陣) =")
print(R)

Qtb = Q.T @ b
x_qr = np.linalg.solve(R, Qtb)  # R 是上三角矩陣，可用回代法快速求解
slope_qr, intercept_qr = x_qr
print(f"QR 分解解得：斜率 m = {slope_qr:.6f}, 截距 c = {intercept_qr:.6f}")


# ---------------------------------------------------------------------------
# 5. 方法三：np.linalg.lstsq
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. 方法三：np.linalg.lstsq")
print("=" * 60)

x_lstsq, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
slope_lstsq, intercept_lstsq = x_lstsq
print(f"np.linalg.lstsq 解得：斜率 m = {slope_lstsq:.6f}, 截距 c = {intercept_lstsq:.6f}")
print(f"矩陣 A 的秩 (rank) = {rank}")


# ---------------------------------------------------------------------------
# 6. 交叉驗證：三種方法結果應一致
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. 交叉驗證：正規方程式 vs QR 分解 vs np.linalg.lstsq")
print("=" * 60)

print("正規方程式解 x =", x_normal)
print("QR 分解解     x =", x_qr)
print("lstsq 解      x =", x_lstsq)

match_normal_qr = np.allclose(x_normal, x_qr)
match_normal_lstsq = np.allclose(x_normal, x_lstsq)
match_qr_lstsq = np.allclose(x_qr, x_lstsq)

print("正規方程式 與 QR 分解 是否一致 (np.allclose):", match_normal_qr)
print("正規方程式 與 lstsq   是否一致 (np.allclose):", match_normal_lstsq)
print("QR 分解     與 lstsq   是否一致 (np.allclose):", match_qr_lstsq)

assert match_normal_qr and match_normal_lstsq and match_qr_lstsq, "三種方法結果不一致！"
print("驗證通過：三種方法求出的最小二乘解完全一致。")

print()
print(f"最終擬合直線: y = {slope_lstsq:.4f} * x + {intercept_lstsq:.4f}")
print(f"（真實直線為 y = {true_slope} * x + {true_intercept}，因雜訊影響，估計值會有些微誤差）")

residual_sum_sq = np.sum((A @ x_lstsq - b) ** 2)
print(f"殘差平方和 ||Ax-b||^2 = {residual_sum_sq:.6f}")


# ---------------------------------------------------------------------------
# 7. 繪圖：資料點散佈圖 + 擬合直線
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("7. 繪圖：資料點與最佳擬合直線")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_data, y_data, color="tab:blue", alpha=0.7, label="模擬資料點 (帶雜訊)")

x_line = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = slope_lstsq * x_line + intercept_lstsq
y_true = true_slope * x_line + true_intercept

ax.plot(x_line, y_fit, color="tab:red", linewidth=2,
        label=f"最小二乘擬合: y = {slope_lstsq:.3f}x + {intercept_lstsq:.3f}")
ax.plot(x_line, y_true, color="tab:green", linewidth=1.5, linestyle="--",
        label=f"真實直線: y = {true_slope}x + {true_intercept}")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("最小二乘法線性迴歸：資料點與最佳擬合直線")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

fig_path = os.path.join(OUT_DIR, "regression_fit.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
print("已儲存資料擬合圖至:", fig_path)

print()
print("全部範例執行完畢。")
