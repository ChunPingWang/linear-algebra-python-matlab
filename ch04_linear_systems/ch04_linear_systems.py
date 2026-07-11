"""
第 4 章：線性方程組與高斯消去法（Python 實作）

本腳本示範：
1. 線性方程組的矩陣表示 Ax = b
2. 增廣矩陣 [A|b] 的建立
3. 手刻高斯消去法函式 gaussian_elimination()，將增廣矩陣化為 RREF
4. 與 np.linalg.solve 交叉驗證（唯一解的情況）
5. 三種解的情況：唯一解、無限多解（自由變數）、無解（矛盾方程式）

執行方式：
    python ch04_linear_systems.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. 線性方程組的矩陣表示 Ax = b
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 線性方程組的矩陣表示 Ax = b")
print("=" * 60)

print("""
考慮下列線性方程組：
    2x + 1y - 1z =  8
   -3x - 1y + 2z = -11
   -2x + 1y + 2z = -3

可以寫成矩陣形式 Ax = b，其中：
    A = [[ 2,  1, -1],       x = [x, y, z]^T
         [-3, -1,  2],
         [-2,  1,  2]]
    b = [8, -11, -3]^T
""")

A_demo = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
], dtype=float)
b_demo = np.array([8, -11, -3], dtype=float)

print("A =")
print(A_demo)
print("b =", b_demo)


# ---------------------------------------------------------------------------
# 2. 增廣矩陣 [A|b]
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 增廣矩陣 [A|b]")
print("=" * 60)


def augmented_matrix(A, b):
    """將係數矩陣 A 與常數向量 b 組合成增廣矩陣 [A|b]。"""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    return np.hstack([A, b])


aug_demo = augmented_matrix(A_demo, b_demo)
print("增廣矩陣 [A|b] =")
print(aug_demo)


# ---------------------------------------------------------------------------
# 3. 高斯消去法的三種列運算（說明）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 高斯消去法的三種基本列運算")
print("=" * 60)
print("""
(1) 交換兩列 (Row Swap):        R_i <-> R_j
(2) 某列乘上非零常數 (Row Scale): R_i -> k * R_i, k != 0
(3) 某列加上另一列的倍數 (Row Addition): R_i -> R_i + k * R_j

這三種運算都不會改變方程組的解集合，稱為「基本列運算」
(elementary row operations)。
""")


# ---------------------------------------------------------------------------
# 4. 手算範例：逐步高斯消去（列印每一步的矩陣狀態）
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. 手算範例：逐步高斯消去")
print("=" * 60)

print("初始增廣矩陣 [A|b]：")
M = aug_demo.copy()
print(M)

print("\n步驟 1: R2 -> R2 - (-3/2) R1 = R2 + 1.5 R1，消去 R2 的第一個元素")
M[1] = M[1] + 1.5 * M[0]
print(M)

print("\n步驟 2: R3 -> R3 - (-2/2) R1 = R3 + 1.0 R1，消去 R3 的第一個元素")
M[2] = M[2] + 1.0 * M[0]
print(M)

print("\n步驟 3: R3 -> R3 - (2/0.5) R2 = R3 - 4 R2，消去 R3 的第二個元素")
M[2] = M[2] - (M[2][1] / M[1][1]) * M[1]
print(M)
print("（此時矩陣已達列梯形式 Row Echelon Form, REF）")

print("\n步驟 4: 將主元 (pivot) 都化為 1")
M[0] = M[0] / M[0][0]
M[1] = M[1] / M[1][1]
M[2] = M[2] / M[2][2]
print(M)

print("\n步驟 5: 回代消去，將主元上方的元素也化為 0（化為 RREF）")
# 消去第 2 欄在 R1 的元素
M[0] = M[0] - M[0][1] * M[1]
print("R1 -> R1 - (R1[1]) * R2:")
print(M)

# 消去第 3 欄在 R1, R2 的元素
M[1] = M[1] - M[1][2] * M[2]
M[0] = M[0] - M[0][2] * M[2]
print("R2 -> R2 - (R2[2]) * R3, R1 -> R1 - (R1[2]) * R3:")
print(M)

print("\n最終簡化列梯形式 (RREF)：")
print(M)
print("由最後一欄可直接讀出解：x =", M[0, -1], ", y =", M[1, -1], ", z =", M[2, -1])
print("(正確解應為 x=2, y=3, z=-1，可自行代入原方程組驗證)")


# ---------------------------------------------------------------------------
# 5. 手刻高斯消去法函式（教學用途）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. 手刻高斯消去法函式 gaussian_elimination()")
print("=" * 60)


def gaussian_elimination(A, b, verbose=False):
    """
    以高斯消去法求解 Ax = b，回傳簡化列梯形式 (RREF) 的增廣矩陣，
    並嘗試判斷解的類型。

    參數:
        A : (n, n) 或 (m, n) 係數矩陣
        b : (n,) 或 (m,) 常數向量
        verbose : 是否印出每一步的矩陣狀態

    回傳:
        rref_matrix : 化簡後的增廣矩陣 (numpy array)
        solution_type : "unique" | "infinite" | "none"
        x : 若為唯一解，回傳解向量；否則回傳 None
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    M = np.hstack([A, b])
    n_rows, n_cols = M.shape
    n_vars = n_cols - 1

    pivot_row = 0
    pivot_cols = []

    # --- 前向消去：化為列梯形式 (REF)，並使用列主元 (partial pivoting) 增加穩定性 ---
    for col in range(n_vars):
        if pivot_row >= n_rows:
            break

        # 在目前欄位中，尋找絕對值最大的列作為主元列（部分主元法，避免除以過小的數）
        max_row = pivot_row + np.argmax(np.abs(M[pivot_row:, col]))
        if np.isclose(M[max_row, col], 0.0):
            continue  # 此欄全為 0（或接近 0），沒有主元，跳到下一欄（表示自由變數）

        if max_row != pivot_row:
            M[[pivot_row, max_row]] = M[[max_row, pivot_row]]
            if verbose:
                print(f"交換 R{pivot_row + 1} <-> R{max_row + 1}:")
                print(M)

        # 將主元化為 1
        M[pivot_row] = M[pivot_row] / M[pivot_row, col]

        # 消去該欄其他所有列的元素
        for r in range(n_rows):
            if r != pivot_row and not np.isclose(M[r, col], 0.0):
                M[r] = M[r] - M[r, col] * M[pivot_row]

        if verbose:
            print(f"以 R{pivot_row + 1} 為主元列，消去第 {col + 1} 欄其餘元素:")
            print(M)

        pivot_cols.append(col)
        pivot_row += 1

    rank_A = len(pivot_cols)

    # --- 判斷解的類型 ---
    # 無解：存在一列形如 [0, 0, ..., 0 | c]，c != 0
    inconsistent = False
    for r in range(n_rows):
        if np.allclose(M[r, :n_vars], 0.0) and not np.isclose(M[r, -1], 0.0):
            inconsistent = True
            break

    if inconsistent:
        solution_type = "none"
        x = None
    elif rank_A < n_vars:
        solution_type = "infinite"
        x = None
    else:
        solution_type = "unique"
        x = np.zeros(n_vars)
        for i, col in enumerate(pivot_cols):
            x[col] = M[i, -1]

    return M, solution_type, x


# 以第 4 節的範例測試，並顯示詳細步驟
print("以 verbose=True 重新示範上方範例，觀察函式內部每一步：")
rref_result, sol_type, x_result = gaussian_elimination(A_demo, b_demo, verbose=True)
print("\n最終 RREF：")
print(rref_result)
print("解的類型:", sol_type)
print("解 x =", x_result)


# ---------------------------------------------------------------------------
# 6. 與 np.linalg.solve 交叉驗證
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. 與 np.linalg.solve 交叉驗證")
print("=" * 60)

x_numpy = np.linalg.solve(A_demo, b_demo)
print("np.linalg.solve 結果 x =", x_numpy)
print("gaussian_elimination 結果 x =", x_result)

is_close = np.allclose(x_numpy, x_result)
print("兩者是否一致 (np.allclose)?", is_close)
assert is_close, "手刻高斯消去法結果與 np.linalg.solve 不一致！"
print("驗證通過：手刻高斯消去法與 np.linalg.solve 結果一致。")


# ---------------------------------------------------------------------------
# 7. 解的三種情況範例
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("7. 解的三種情況")
print("=" * 60)

# --- 7.1 唯一解 (unique solution) ---
print("\n--- 7.1 唯一解 (Unique Solution) ---")
print("方程組：")
print("  x + y = 3")
print(" 2x - y = 0")
A_unique = np.array([[1, 1], [2, -1]], dtype=float)
b_unique = np.array([3, 0], dtype=float)
rref_u, type_u, x_u = gaussian_elimination(A_unique, b_unique)
print("RREF [A|b] =")
print(rref_u)
print("解的類型:", type_u)
print("解 x =", x_u, "（意即 x=1, y=2）")
x_u_numpy = np.linalg.solve(A_unique, b_unique)
print("np.linalg.solve 驗證:", x_u_numpy, " 一致?", np.allclose(x_u, x_u_numpy))

# --- 7.2 無限多解 (infinite solutions, 自由變數) ---
print("\n--- 7.2 無限多解 (Infinite Solutions) ---")
print("方程組：")
print("  x + y + z = 6")
print(" 2x + 2y + 2z = 12   (此式為第一式的 2 倍，屬於重複資訊)")
A_inf = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
b_inf = np.array([6, 12], dtype=float)
rref_inf, type_inf, x_inf = gaussian_elimination(A_inf, b_inf)
print("RREF [A|b] =")
print(rref_inf)
print("解的類型:", type_inf)
print("說明: 只有一個主元（rank=1）但有 3 個未知數，")
print("      因此有 3 - 1 = 2 個自由變數，解集合為一個平面，屬於無限多解。")

# --- 7.3 無解 (no solution, 矛盾方程式) ---
print("\n--- 7.3 無解 (No Solution) ---")
print("方程組：")
print("  x + y = 2")
print("  x + y = 5   (與第一式矛盾: x+y 不可能同時等於 2 和 5)")
A_none = np.array([[1, 1], [1, 1]], dtype=float)
b_none = np.array([2, 5], dtype=float)
rref_none, type_none, x_none = gaussian_elimination(A_none, b_none)
print("RREF [A|b] =")
print(rref_none)
print("解的類型:", type_none)
print("說明: 化簡後出現形如 [0, 0 | c] 且 c != 0 的列，代表矛盾方程式，無解。")


print()
print("=" * 60)
print("全部範例執行完畢。")
print("=" * 60)
