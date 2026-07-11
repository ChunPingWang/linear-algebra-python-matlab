"""
第 7 章：向量空間、基底與秩
======================================

本腳本示範：
1. 向量空間公理的直觀說明（封閉性、加法/純量乘法規則）
2. 子空間 (subspace) 的判斷條件
3. 零空間 (null space)：Ax = 0 的解集合，並求其基底
4. 行空間 (column space)：A 的行向量所張成的空間，並求其基底
5. 基底 (basis) 與維度 (dimension)
6. 秩-零度定理 (Rank-Nullity Theorem)：rank(A) + nullity(A) = n
   在兩個不同範例矩陣上驗證

執行方式：
    python ch07_vector_spaces.py
"""

import numpy as np
from scipy.linalg import null_space


def print_section(title: str) -> None:
    """印出一個清楚的段落標題，方便在終端機閱讀輸出。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. 向量空間公理的直觀說明
# ----------------------------------------------------------------------
def demo_vector_space_axioms() -> None:
    print_section("1. 向量空間公理的直觀說明")

    print("向量空間 (vector space) 是一個集合 V，裡面的向量可以做「加法」與")
    print("「純量乘法」，並且運算結果仍然留在 V 裡面（這就是「封閉性」）。")
    print("嚴格定義有 8 條公理，但初學者只需要記住兩個核心規則：")
    print("  規則 1（加法封閉）：若 u, v 屬於 V，則 u + v 也屬於 V")
    print("  規則 2（純量乘法封閉）：若 u 屬於 V，c 是任意純量，則 c*u 也屬於 V")
    print("其餘公理（交換律、結合律、有零向量、有反向量、分配律等）")
    print("在 R^n（例如 R^2、R^3）這種「一般座標空間」中都自動成立，")
    print("所以本章重點會放在 R^n 的「子空間」上，而不是抽象證明。")

    # 用具體例子驗證封閉性
    u = np.array([1.0, 2.0])
    v = np.array([3.0, -1.0])
    c = 2.5
    print(f"\n範例：u = {u}, v = {v}, 純量 c = {c}")
    print(f"  u + v = {u + v}  (仍是 R^2 中的向量 -> 加法封閉)")
    print(f"  c * u = {c * u}  (仍是 R^2 中的向量 -> 純量乘法封閉)")


# ----------------------------------------------------------------------
# 2. 子空間 (Subspace) 的判斷
# ----------------------------------------------------------------------
def is_subspace_candidate_closed(
    vectors: list[np.ndarray], combos: list[tuple[float, float]]
) -> bool:
    """
    檢查一組向量在給定的線性組合係數下，是否仍然滿足某個子空間的定義方程式。
    這裡示範用「代入檢查」的方式，直觀展示封閉性。
    """
    results = []
    for c1, c2 in combos:
        w = c1 * vectors[0] + c2 * vectors[1]
        results.append(w)
    return results


def demo_subspace() -> None:
    print_section("2. 子空間 (Subspace) 的判斷")

    print("子空間是向量空間 V 裡面的一個子集合 W，且 W 自己也必須是一個向量空間。")
    print("判斷 W 是否為子空間，只需檢查 3 個條件：")
    print("  (1) 零向量在 W 中")
    print("  (2) 加法封閉：u, v 屬於 W -> u + v 也屬於 W")
    print("  (3) 純量乘法封閉：u 屬於 W, c 為純量 -> c*u 也屬於 W")

    print("\n範例 A：W = { (x, y) in R^2 : y = 2x }（通過原點的一條直線）")
    print("  這是子空間：零向量 (0,0) 滿足 y=2x；")
    print("  任兩點相加、任一點乘上純量後，仍然滿足 y = 2x（自行代入可驗證）。")
    p1 = np.array([1.0, 2.0])   # 滿足 y = 2x
    p2 = np.array([-3.0, -6.0])  # 滿足 y = 2x
    s = p1 + p2
    print(f"    p1 = {p1} (在 W 中), p2 = {p2} (在 W 中)")
    print(f"    p1 + p2 = {s} -> y/x = {s[1] / s[0]:.1f}，仍滿足 y = 2x，封閉成立")

    print("\n範例 B：W' = { (x, y) in R^2 : y = 2x + 1 }（不過原點的直線）")
    print("  這「不是」子空間，因為零向量 (0,0) 不滿足 y = 2x + 1（0 != 1）。")
    q1 = np.array([0.0, 1.0])   # 滿足 y = 2x+1
    q2 = np.array([1.0, 3.0])   # 滿足 y = 2x+1
    s2 = q1 + q2
    print(f"    q1 = {q1} (在 W' 中), q2 = {q2} (在 W' 中)")
    print(f"    q1 + q2 = {s2} -> 檢查 y = 2x+1 ? {s2[1]} vs {2 * s2[0] + 1} -> 不相等，封閉性失敗")
    print("    結論：任何不過原點的集合都不可能是子空間。")


# ----------------------------------------------------------------------
# 3. 手算範例矩陣：求零空間、行空間、秩，並驗證秩-零度定理
# ----------------------------------------------------------------------
def rref_with_pivots(A: np.ndarray, tol: float = 1e-10):
    """
    手刻高斯-喬登消去法，將矩陣化簡為簡化列梯形式 (RREF)，
    並回傳 (RREF 矩陣, 主元欄位索引列表)。
    這個函式用來找出行空間基底所對應的「原始矩陣」欄位。
    """
    M = A.astype(float).copy()
    n_rows, n_cols = M.shape
    pivot_row = 0
    pivot_cols = []

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # 部分主元法：找此欄中，從 pivot_row 開始絕對值最大的列
        max_row = pivot_row + np.argmax(np.abs(M[pivot_row:, col]))
        if np.abs(M[max_row, col]) < tol:
            continue  # 此欄沒有主元
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]]
        M[pivot_row, :] = M[pivot_row, :] / M[pivot_row, col]  # 主元歸一化為 1
        for r in range(n_rows):
            if r != pivot_row:
                M[r, :] -= M[r, col] * M[pivot_row, :]
        pivot_cols.append(col)
        pivot_row += 1

    return M, pivot_cols


def analyze_matrix(A: np.ndarray, name: str) -> None:
    """對矩陣 A 做完整分析：秩、零空間基底、行空間基底，並驗證秩-零度定理。"""
    m, n = A.shape
    print(f"\n矩陣 {name} ({m} x {n}):")
    print(A)

    # --- 秩 ---
    rank_A = np.linalg.matrix_rank(A)
    print(f"\nrank({name}) = {rank_A}  (用 numpy.linalg.matrix_rank 計算)")

    # --- RREF 與主元欄位（手算過程） ---
    rref_A, pivot_cols = rref_with_pivots(A)
    print(f"\n{name} 的簡化列梯形式 (RREF):")
    print(np.round(rref_A, 4))
    print(f"主元欄位 (0-indexed): {pivot_cols}  "
          f"(對應原矩陣第 {[c + 1 for c in pivot_cols]} 欄，1-indexed)")

    # --- 行空間基底：取原矩陣中，主元欄位所在的「原始」欄向量 ---
    col_space_basis = A[:, pivot_cols]
    print(f"\n行空間 (column space) 的基底：取原矩陣中主元欄位對應的欄向量")
    for i, c in enumerate(pivot_cols):
        print(f"  基底向量 {i + 1} = {name} 的第 {c + 1} 欄 = {A[:, c]}")
    print(f"行空間維度 = 主元個數 = rank({name}) = {rank_A}")

    # --- 零空間基底：用 scipy.linalg.null_space ---
    ns = null_space(A)
    nullity = ns.shape[1]
    print(f"\n零空間 (null space) 的基底 (用 scipy.linalg.null_space 計算，"
          f"欄向量為正交歸一化後的基底向量):")
    if nullity == 0:
        print("  零空間只有零向量 {0}，nullity = 0（A 的各欄線性獨立）")
    else:
        for i in range(nullity):
            print(f"  基底向量 {i + 1} = {ns[:, i]}")
    print(f"nullity({name}) = 零空間維度 = {nullity}")

    # 驗證 A @ v ≈ 0 對零空間中每個基底向量成立
    if nullity > 0:
        residual = A @ ns
        print(f"驗證 {name} @ (零空間基底) ≈ 0:\n{np.round(residual, 10)}")
        print(f"  最大絕對誤差 = {np.max(np.abs(residual)):.2e} (應接近 0)")

    # --- 秩-零度定理驗證 ---
    print(f"\n秩-零度定理驗證：rank({name}) + nullity({name}) = 欄數 n ?")
    print(f"  {rank_A} + {nullity} = {rank_A + nullity}，欄數 n = {n}")
    if rank_A + nullity == n:
        print(f"  驗證通過！rank({name}) + nullity({name}) = n = {n}")
    else:
        raise AssertionError(f"秩-零度定理驗證失敗！{name}")
    assert rank_A + nullity == n


def demo_worked_example() -> None:
    print_section("3. 手算範例：零空間、行空間、秩，與秩-零度定理")

    print("範例矩陣 A (3x4)：")
    print("  A = [[1, 2, 0, 3],")
    print("       [0, 0, 1, 2],")
    print("       [1, 2, 1, 5]]")
    print("\n手算過程（高斯消去法化簡為 RREF）：")
    print("  R3 = R3 - R1  ->  [[1, 2, 0, 3], [0, 0, 1, 2], [0, 0, 1, 2]]")
    print("  R3 = R3 - R2  ->  [[1, 2, 0, 3], [0, 0, 1, 2], [0, 0, 0, 0]]")
    print("  已是簡化列梯形式：主元在第 1、3 欄，第 2、4 欄是自由變數欄")
    print("  -> rank(A) = 2（兩個主元）")
    print("\n零空間手算：設 x2 = s, x4 = t（自由變數），由 RREF 列出方程式：")
    print("  x1 + 2*x2 + 3*x4 = 0  ->  x1 = -2s - 3t")
    print("  x3 + 2*x4 = 0         ->  x3 = -2t")
    print("  解向量 = s*(-2, 1, 0, 0) + t*(-3, 0, -2, 1)")
    print("  -> 零空間基底 = { (-2,1,0,0), (-3,0,-2,1) }，nullity(A) = 2")
    print("\n行空間手算：主元欄位是第 1、3 欄，取「原矩陣」第 1、3 欄作為行空間基底：")
    print("  基底 = { (1,0,1)^T, (0,1,1)^T }（分別是 A 的第 1 欄與第 3 欄）")
    print("\n秩-零度定理：rank(A) + nullity(A) = 2 + 2 = 4 = n（A 有 4 欄），驗證成立！")

    print("\n以下用程式重新計算，驗證手算結果：")
    A = np.array([
        [1, 2, 0, 3],
        [0, 0, 1, 2],
        [1, 2, 1, 5],
    ], dtype=float)
    analyze_matrix(A, "A")


# ----------------------------------------------------------------------
# 4. 第二個範例矩陣：再次驗證秩-零度定理
# ----------------------------------------------------------------------
def demo_second_example() -> None:
    print_section("4. 第二個範例矩陣（4x3）：再次驗證秩-零度定理")

    print("範例矩陣 C (4x3)：")
    print("  C = [[1, 2, 3],")
    print("       [2, 4, 6],")
    print("       [1, 0, 1],")
    print("       [0, 1, 1]]")
    print("觀察：第 2 列 = 2 * 第 1 列，且第 3 欄 = 第 1 欄 + 第 2 欄，")
    print("預期矩陣的秩會小於欄數與列數。")

    C = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=float)
    analyze_matrix(C, "C")


# ----------------------------------------------------------------------
# 5. 基底與維度：滿秩方陣範例
# ----------------------------------------------------------------------
def demo_basis_dimension() -> None:
    print_section("5. 基底 (Basis) 與維度 (Dimension) 補充範例")

    print("基底：一組「線性獨立」且能「生成 (span)」整個空間的向量集合。")
    print("維度：基底中向量的個數，也就是這個空間「自由度」的數量。")

    D = np.array([
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0],
    ], dtype=float)
    print("\n範例矩陣 D (3x3)：")
    print(D)
    rank_D = np.linalg.matrix_rank(D)
    ns_D = null_space(D)
    print(f"rank(D) = {rank_D}，nullity(D) = {ns_D.shape[1]}")
    if rank_D == 3:
        print("D 的秩 = 3 = 欄數，代表 D 的三個欄向量線性獨立，")
        print("它們本身就是 R^3 的一組基底，行空間 = R^3（維度 3），零空間只有 {0}（維度 0）。")
        print(f"秩-零度定理：{rank_D} + {ns_D.shape[1]} = {rank_D + ns_D.shape[1]} = 3 (欄數)，驗證成立！")
    assert rank_D + ns_D.shape[1] == D.shape[1]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    demo_vector_space_axioms()
    demo_subspace()
    demo_worked_example()
    demo_second_example()
    demo_basis_dimension()
    print_section("完成")
    print("本章所有示範已執行完畢，秩-零度定理已在 3 個不同範例矩陣（A, C, D）上驗證通過。")


if __name__ == "__main__":
    main()
