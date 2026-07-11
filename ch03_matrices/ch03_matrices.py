"""
第 3 章：矩陣與矩陣運算
========================
本腳本對應 ch03_matrices.md 的教學內容，示範：
- 矩陣的定義與形狀 (shape)
- 矩陣加法、純量乘法
- 矩陣乘法（@ 與 np.matmul），並驗證手算範例
- 矩陣乘法不滿足交換律 (AB != BA)
- 轉置矩陣
- 特殊矩陣：零矩陣、單位矩陣、對角矩陣、對稱矩陣、上/下三角矩陣
- 跡 (trace) 及其性質

直接執行：
    python ch03_matrices.py
"""

import numpy as np


def section(title: str) -> None:
    """印出區塊標題，方便閱讀輸出。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. 矩陣的定義與形狀 (shape)
# ----------------------------------------------------------------------
section("1. 矩陣的定義與形狀 (shape)")

A_demo = np.array([[1, 2, 3],
                    [4, 5, 6]])
print("A_demo =\n", A_demo)
print("A_demo 的形狀 (shape):", A_demo.shape, "  (2 列 x 3 行)")
print("a_12 (第1列第2行, 0-indexed: [0,1]) =", A_demo[0, 1])
print("a_23 (第2列第3行, 0-indexed: [1,2]) =", A_demo[1, 2])


# ----------------------------------------------------------------------
# 2. 矩陣加法與純量乘法
# ----------------------------------------------------------------------
section("2. 矩陣加法與純量乘法")

M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])

M_sum = M1 + M2
M_scalar = 2 * M1

print("M1 =\n", M1)
print("M2 =\n", M2)
print("M1 + M2 =\n", M_sum)
print("預期結果 [[6, 8], [10, 12]] -> 驗證:", np.array_equal(M_sum, [[6, 8], [10, 12]]))

print("\n2 * M1 =\n", M_scalar)
print("預期結果 [[2, 4], [6, 8]] -> 驗證:", np.array_equal(M_scalar, [[2, 4], [6, 8]]))


# ----------------------------------------------------------------------
# 3. 矩陣乘法：定義、維度規則、手算範例驗證
# ----------------------------------------------------------------------
section("3. 矩陣乘法")

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("A =\n", A)
print("B =\n", B)
print("A 的形狀:", A.shape, " B 的形狀:", B.shape)

# 用 @ 運算子（建議寫法）
AB_at = A @ B
# 用 np.matmul（等價寫法）
AB_matmul = np.matmul(A, B)

print("\nAB = A @ B =\n", AB_at)
print("AB = np.matmul(A, B) =\n", AB_matmul)
print("兩種寫法結果一致:", np.array_equal(AB_at, AB_matmul))

# 與手算範例比對：手算結果為 [[19, 22], [43, 50]]
expected_AB = np.array([[19, 22], [43, 50]])
print("\n手算範例結果應為 [[19, 22], [43, 50]]")
print("程式計算結果與手算結果一致:", np.array_equal(AB_at, expected_AB))

# 逐步印出手算過程對照 (c_ij = A 第 i 列與 B 第 j 行的內積)
print("\n逐步驗證每個元素:")
for i in range(2):
    for j in range(2):
        row = A[i, :]
        col = B[:, j]
        value = np.dot(row, col)
        print(f"  c_{i+1}{j+1} = A第{i+1}列 . B第{j+1}行 "
              f"= {row.tolist()} . {col.tolist()} = {value}")


# ----------------------------------------------------------------------
# 3.1 矩陣乘法不滿足交換律
# ----------------------------------------------------------------------
section("3.1 矩陣乘法不滿足交換律 (AB != BA)")

BA = B @ A
expected_BA = np.array([[23, 34], [31, 46]])

print("BA = B @ A =\n", BA)
print("手算範例結果應為 [[23, 34], [31, 46]]")
print("程式計算結果與手算結果一致:", np.array_equal(BA, expected_BA))

print("\nAB =\n", AB_at)
print("BA =\n", BA)
print("AB == BA ?", np.array_equal(AB_at, BA), " -> 證實矩陣乘法一般不滿足交換律")


# ----------------------------------------------------------------------
# 4. 轉置矩陣
# ----------------------------------------------------------------------
section("4. 轉置矩陣")

A_wide = np.array([[1, 2, 3],
                    [4, 5, 6]])
A_wide_T = A_wide.T

print("A_wide (2x3) =\n", A_wide)
print("A_wide.T (3x2) =\n", A_wide_T)
print("預期轉置 [[1,4],[2,5],[3,6]] -> 驗證:",
      np.array_equal(A_wide_T, [[1, 4], [2, 5], [3, 6]]))

# 驗證 (A^T)^T = A
print("(A_wide.T).T == A_wide ?", np.array_equal(A_wide_T.T, A_wide))

# 驗證 (AB)^T = B^T A^T
lhs = (A @ B).T
rhs = B.T @ A.T
print("\n驗證 (AB)^T = B^T A^T:")
print("(AB)^T =\n", lhs)
print("B^T A^T =\n", rhs)
print("兩者相等:", np.array_equal(lhs, rhs))


# ----------------------------------------------------------------------
# 5. 特殊矩陣
# ----------------------------------------------------------------------
section("5. 特殊矩陣")

# 零矩陣
zero_mat = np.zeros((2, 3))
print("零矩陣 (2x3):\n", zero_mat)

# 單位矩陣
I3 = np.eye(3)
print("\n單位矩陣 I3:\n", I3)

# 驗證 A*I = I*A = A（用 3x3 方陣示範）
A3 = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
print("\nA3 =\n", A3)
print("A3 @ I3 =\n", A3 @ I3)
print("I3 @ A3 =\n", I3 @ A3)
print("A3 @ I3 == A3 == I3 @ A3 ?",
      np.array_equal(A3 @ I3, A3) and np.array_equal(I3 @ A3, A3))

# 對角矩陣
D = np.diag([2, 5, -1])
print("\n對角矩陣 D = diag(2, 5, -1):\n", D)

# 對稱矩陣
S = np.array([[1, 2], [2, 3]])
print("\n對稱矩陣 S =\n", S)
print("S.T =\n", S.T)
print("S 是否對稱 (S == S.T) ?", np.array_equal(S, S.T))

# 上三角矩陣
U = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
print("\n上三角矩陣 U =\n", U)
print("用 np.triu 從任意矩陣取上三角部分:\n", np.triu(A3))

# 下三角矩陣
L = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
print("\n下三角矩陣 L =\n", L)
print("用 np.tril 從任意矩陣取下三角部分:\n", np.tril(A3))


# ----------------------------------------------------------------------
# 6. 跡 (trace)
# ----------------------------------------------------------------------
section("6. 跡 (trace)")

print("A =\n", A)
trace_A = np.trace(A)
print("trace(A) =", trace_A, " (預期 1+4=5)")

# 跡的線性
C = np.array([[2, 0], [1, 3]])
print("\nC =\n", C)
print("trace(A) + trace(C) =", np.trace(A) + np.trace(C))
print("trace(A + C) =", np.trace(A + C))
print("線性驗證 (trace(A+C) == trace(A)+trace(C)):",
      np.trace(A + C) == np.trace(A) + np.trace(C))

k = 3
print(f"\ntrace({k}*A) =", np.trace(k * A))
print(f"{k} * trace(A) =", k * np.trace(A))
print("純量倍數驗證:", np.trace(k * A) == k * np.trace(A))

# trace(AB) == trace(BA)
trace_AB = np.trace(A @ B)
trace_BA = np.trace(B @ A)
print("\ntrace(AB) =", trace_AB, " (預期 19+50=69)")
print("trace(BA) =", trace_BA, " (預期 23+46=69)")
print("trace(AB) == trace(BA) ?", trace_AB == trace_BA,
      "  -> 即使 AB != BA，兩者的跡仍相等")


section("全部驗證完成")
print("本章所有數值計算已與教學文件 (ch03_matrices.md) 的手算範例比對一致。")
