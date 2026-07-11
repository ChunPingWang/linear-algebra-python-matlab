"""
第 2 章：線性組合、生成空間與線性獨立
======================================

本腳本示範：
1. 線性組合 (linear combination) 的計算
2. 生成空間 (span) 的幾何意義（2D 平面示意圖）
3. 線性獨立 / 線性相依的判斷
   - 方法一：手刻函式，解齊次方程組 c1*v1 + c2*v2 + ... = 0，檢查是否只有零解
   - 方法二：用 np.linalg.matrix_rank 判斷矩陣的秩
4. 幾何直觀圖示：三個線性相依向量（其中一個是另兩個的線性組合）

執行方式：
    python ch02_span_independence.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # 不開啟互動視窗，只存檔
import matplotlib.pyplot as plt
import numpy as np

# 設定中文字型，避免圖片中的中文標籤顯示為方框（缺字警告）
plt.rcParams["font.sans-serif"] = [
    "PingFang TC", "Heiti TC", "Arial Unicode MS", "Microsoft JhengHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 圖片輸出資料夾：與本腳本同一層的 ch02_span_independence 資料夾
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_section(title: str) -> None:
    """印出一個清楚的段落標題，方便在終端機閱讀輸出。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. 線性組合 (Linear Combination)
# ----------------------------------------------------------------------
def demo_linear_combination() -> None:
    print_section("1. 線性組合 (Linear Combination)")

    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    c1, c2 = 3, -2

    combo = c1 * v1 + c2 * v2
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"線性組合 c1*v1 + c2*v2 = {combo}")

    # 判斷目標向量 b 是否可以用 v1, v2 的線性組合表示
    # 也就是要解 A c = b，其中 A 的欄向量是 v1, v2
    b = np.array([5, -1])
    A = np.column_stack([v1, v2])
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ coeffs - b
    print(f"\n目標向量 b = {b}")
    print(f"解出的係數 (c1, c2) = {coeffs}")
    print(f"驗證 A @ coeffs - b = {residual} (應接近 0 向量，代表 b 確實在 span{{v1, v2}} 中)")


# ----------------------------------------------------------------------
# 2. 生成空間 (Span) 的幾何意義
# ----------------------------------------------------------------------
def demo_span_plot() -> None:
    print_section("2. 生成空間 (Span) 的幾何意義")

    v1 = np.array([2, 1])
    v2 = np.array([-1, 1])
    print(f"v1 = {v1}, v2 = {v2} 彼此不平行，兩者可以生成 (span) 整個 2D 平面 R^2。")
    print("這代表：對任何 2D 向量 b，都能找到 c1, c2 使得 c1*v1 + c2*v2 = b。")

    # 隨機取樣許多係數組合，畫出它們的線性組合，直觀呈現 span 幾乎覆蓋整個平面
    rng = np.random.default_rng(42)
    coeffs = rng.uniform(-2, 2, size=(400, 2))
    points = coeffs @ np.array([v1, v2])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.4, color="steelblue",
               label="c1*v1 + c2*v2 的隨機取樣點")
    ax.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
              color="red", width=0.01, label="v1")
    ax.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
              color="green", width=0.01, label="v2")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.set_title("span{v1, v2}：兩個不平行向量生成整個 2D 平面")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(OUT_DIR, "span_demo.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"圖片已存至: {out_path}")


# ----------------------------------------------------------------------
# 3. 線性獨立 / 線性相依的判斷
# ----------------------------------------------------------------------
def is_independent_by_homogeneous_system(vectors: list[np.ndarray], tol: float = 1e-10) -> bool:
    """
    手刻函式：判斷一組向量是否線性獨立。

    做法：把向量當作矩陣 A 的欄向量，解齊次方程組 A c = 0。
    - 用高斯消去法（透過 numpy 的列梯形化簡概念，這裡直接用最小平方/秩的方式驗證）
      實務上我們用「化簡列梯形式後，主元 (pivot) 的數量是否等於向量個數」來判斷。
    - 若唯一解是 c = 0（零向量），代表線性獨立；
      若存在非零解，代表線性相依。

    這裡用列梯形化簡 (Gaussian elimination) 手動實作，不直接呼叫 matrix_rank，
    以展示「解齊次方程組」背後的邏輯。
    """
    A = np.column_stack(vectors).astype(float)
    n_rows, n_cols = A.shape

    # 高斯消去法，化簡為列梯形式 (row echelon form)
    M = A.copy()
    pivot_row = 0
    pivot_cols = []
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # 找此欄中，從 pivot_row 開始絕對值最大的列，作為主元（部分主元法，避免數值誤差）
        max_row = pivot_row + np.argmax(np.abs(M[pivot_row:, col]))
        if np.abs(M[max_row, col]) < tol:
            continue  # 此欄沒有主元，跳到下一欄（代表這一欄相對前面的欄是線性相依的）
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]  # 交換列
        # 消去 pivot_row 以下的列，使該欄下方元素變成 0
        for r in range(pivot_row + 1, n_rows):
            factor = M[r, col] / M[pivot_row, col]
            M[r, :] -= factor * M[pivot_row, :]
        pivot_cols.append(col)
        pivot_row += 1

    num_pivots = len(pivot_cols)
    num_vectors = n_cols
    # 主元數量 = 秩。若秩等於向量個數，代表齊次方程組只有零解 -> 線性獨立
    return num_pivots == num_vectors


def demo_independence() -> None:
    print_section("3. 線性獨立 / 線性相依的判斷")

    # 範例 A：兩個不平行的 2D 向量 -> 線性獨立
    v1 = np.array([1, 2])
    v2 = np.array([3, 1])
    vectors_a = [v1, v2]
    A = np.column_stack(vectors_a)
    print(f"範例 A: v1 = {v1}, v2 = {v2}")
    print(f"  手刻函式判斷 (解齊次方程組是否只有零解): "
          f"{'線性獨立' if is_independent_by_homogeneous_system(vectors_a) else '線性相依'}")
    print(f"  matrix_rank(A) = {np.linalg.matrix_rank(A)}, 向量個數 = {len(vectors_a)} "
          f"-> {'線性獨立' if np.linalg.matrix_rank(A) == len(vectors_a) else '線性相依'}")

    # 範例 B：三個 2D 向量，其中 v3 = v1 + v2，必定線性相依（2D 空間最多只能有 2 個獨立向量）
    v1b = np.array([1, 0])
    v2b = np.array([0, 1])
    v3b = v1b + 2 * v2b  # v3 是 v1, v2 的線性組合
    vectors_b = [v1b, v2b, v3b]
    B = np.column_stack(vectors_b)
    print(f"\n範例 B: v1 = {v1b}, v2 = {v2b}, v3 = v1 + 2*v2 = {v3b}")
    print(f"  手刻函式判斷: "
          f"{'線性獨立' if is_independent_by_homogeneous_system(vectors_b) else '線性相依'}")
    print(f"  matrix_rank(B) = {np.linalg.matrix_rank(B)}, 向量個數 = {len(vectors_b)} "
          f"-> {'線性獨立' if np.linalg.matrix_rank(B) == len(vectors_b) else '線性相依'}")
    print("  說明：因為 v3 = 1*v1 + 2*v2，所以 1*v1 + 2*v2 + (-1)*v3 = 0 是一組非零解，"
          "代表這三個向量線性相依。")

    # 範例 C：3D 空間中三個線性獨立的向量
    v1c = np.array([1, 0, 0])
    v2c = np.array([0, 1, 0])
    v3c = np.array([0, 0, 1])
    vectors_c = [v1c, v2c, v3c]
    C = np.column_stack(vectors_c)
    print(f"\n範例 C (3D 標準基底): v1 = {v1c}, v2 = {v2c}, v3 = {v3c}")
    print(f"  matrix_rank(C) = {np.linalg.matrix_rank(C)}, 向量個數 = {len(vectors_c)} "
          f"-> {'線性獨立' if np.linalg.matrix_rank(C) == len(vectors_c) else '線性相依'}")


# ----------------------------------------------------------------------
# 4. 秩 (Rank) 的直觀意義
# ----------------------------------------------------------------------
def demo_rank_intuition() -> None:
    print_section("4. 秩 (Rank) 的直觀意義（第 7 章會有更完整的討論）")

    examples = {
        "滿秩方陣 (2x2 獨立)": np.array([[1, 3], [2, 1]]),
        "秩為 1 的方陣 (兩列成比例)": np.array([[1, 2], [2, 4]]),
        "3 個向量在 2D 空間 (最多秩 2)": np.array([[1, 0, 1], [0, 1, 2]]),
    }
    for name, M in examples.items():
        r = np.linalg.matrix_rank(M)
        print(f"{name}:\n{M}\n  matrix_rank = {r}  (矩陣形狀 {M.shape})\n")

    print("直觀意義：秩代表這組向量「實際能撐出的空間維度」。")
    print("- 秩 = 向量個數 -> 線性獨立，沒有「浪費」的方向")
    print("- 秩 < 向量個數 -> 線性相依，至少有一個向量是其他向量的線性組合，方向重複了")


# ----------------------------------------------------------------------
# 5. 幾何直觀圖示：三個線性相依向量
# ----------------------------------------------------------------------
def demo_dependent_plot() -> None:
    print_section("5. 幾何直觀圖示：2D 平面中三個線性相依向量")

    v1 = np.array([2, 1])
    v2 = np.array([-1, 1])
    v3 = v1 + v2  # v3 是 v1, v2 的線性組合，因此三者線性相依
    print(f"v1 = {v1}, v2 = {v2}, v3 = v1 + v2 = {v3}")
    print("由於 v3 可以表示成 v1 與 v2 的線性組合，這三個向量落在同一個 2D 平面內，"
          "彼此線性相依（v1*1 + v2*1 + v3*(-1) = 0 是非零解）。")

    fig, ax = plt.subplots(figsize=(6, 6))
    origin = np.array([0, 0])
    colors = {"v1": "red", "v2": "green", "v3": "purple"}
    for name, v in zip(["v1", "v2", "v3"], [v1, v2, v3]):
        ax.quiver(*origin, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                   color=colors[name], width=0.012, label=f"{name} = {v}")

    # 用虛線畫出平行四邊形法則，顯示 v3 = v1 + v2
    ax.plot([v1[0], v3[0]], [v1[1], v3[1]], "k--", alpha=0.5)
    ax.plot([v2[0], v3[0]], [v2[1], v3[1]], "k--", alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-2, 4)
    ax.set_ylim(-1, 3)
    ax.set_aspect("equal")
    ax.set_title("線性相依示意圖：v3 = v1 + v2（三者共平面，v3 沒有提供新方向）")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(OUT_DIR, "dependent_vectors_demo.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"圖片已存至: {out_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    demo_linear_combination()
    demo_span_plot()
    demo_independence()
    demo_rank_intuition()
    demo_dependent_plot()
    print_section("完成")
    print("本章所有示範已執行完畢，圖片已存至 ch02_span_independence/ 資料夾中。")


if __name__ == "__main__":
    main()
