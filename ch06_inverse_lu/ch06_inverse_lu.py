"""
第 6 章：逆矩陣與 LU 分解（Python 實作）

本腳本示範：
1. 2×2 矩陣的逆矩陣（手算公式對照 numpy.linalg.inv）
2. 3×3 矩陣的逆矩陣（對照講義中高斯-若丹消去法手算的結果）
3. 用 numpy.linalg.inv 求逆矩陣，並驗證 A @ A_inv ≈ I
4. 判斷奇異矩陣（singular matrix）無法求逆
5. 逆矩陣的性質：(AB)^-1 = B^-1 A^-1、(A^T)^-1 = (A^-1)^T、(A^-1)^-1 = A
6. 用 scipy.linalg.lu 做 LU 分解，並驗證 L @ U ≈ P @ A

執行方式：
    python ch06_inverse_lu.py
"""

import numpy as np
from scipy.linalg import lu

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. 2×2 逆矩陣
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 2x2 逆矩陣")
print("=" * 60)

A2 = np.array([[2.0, 1.0],
               [1.0, 1.0]])
print("矩陣 A =")
print(A2)

det_A2 = np.linalg.det(A2)
print("det(A) =", det_A2)

A2_inv = np.linalg.inv(A2)
print("A 的逆矩陣 A_inv =")
print(A2_inv)

# 手算公式對照： (1/det) * [[d, -b], [-c, a]]
a, b = A2[0, 0], A2[0, 1]
c, d = A2[1, 0], A2[1, 1]
A2_inv_formula = (1.0 / det_A2) * np.array([[d, -b], [-c, a]])
print("用公式手算的逆矩陣 =")
print(A2_inv_formula)
print("公式結果與 numpy 結果是否一致:", np.allclose(A2_inv, A2_inv_formula))

I2 = np.eye(2)
check1 = np.allclose(A2 @ A2_inv, I2)
print("驗證 A @ A_inv ≈ I:", check1)
assert check1, "2x2 逆矩陣驗證失敗！"


# ---------------------------------------------------------------------------
# 2. 3×3 逆矩陣（對照講義高斯-若丹消去法手算範例）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 3x3 逆矩陣（對照高斯-若丹消去法手算結果）")
print("=" * 60)

A3 = np.array([[1.0, 2.0, 3.0],
               [0.0, 1.0, 4.0],
               [5.0, 6.0, 0.0]])
print("矩陣 A =")
print(A3)

det_A3 = np.linalg.det(A3)
print("det(A) =", det_A3)

A3_inv = np.linalg.inv(A3)
print("numpy 計算出的逆矩陣 A_inv =")
print(A3_inv)

# 講義中高斯-若丹消去法手算得到的結果
A3_inv_hand = np.array([[-24.0, 18.0, 5.0],
                         [20.0, -15.0, -4.0],
                         [-5.0, 4.0, 1.0]])
print("講義手算的逆矩陣 =")
print(A3_inv_hand)
print("numpy 結果與手算結果是否一致:", np.allclose(A3_inv, A3_inv_hand))

I3 = np.eye(3)
check2 = np.allclose(A3 @ A3_inv, I3)
print("驗證 A @ A_inv ≈ I:", check2)
assert check2, "3x3 逆矩陣驗證失敗！"


# ---------------------------------------------------------------------------
# 3. 奇異矩陣（singular matrix）無法求逆
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 奇異矩陣（det(A) = 0）無法求逆")
print("=" * 60)

S = np.array([[2.0, 4.0],
              [1.0, 2.0]])
print("矩陣 S =")
print(S)
det_S = np.linalg.det(S)
print("det(S) =", det_S, "（接近 0，代表 S 為奇異矩陣）")

try:
    S_inv = np.linalg.inv(S)
    print("S 的逆矩陣 =", S_inv)
except np.linalg.LinAlgError as e:
    print("np.linalg.inv 拋出例外，如預期:", e)


# ---------------------------------------------------------------------------
# 4. 逆矩陣的性質
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 逆矩陣的性質")
print("=" * 60)

A = np.array([[2.0, 1.0],
              [1.0, 1.0]])
B = np.array([[3.0, 0.0],
              [1.0, 2.0]])

A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# 性質 1: (AB)^-1 = B^-1 A^-1
AB_inv = np.linalg.inv(A @ B)
B_inv_A_inv = B_inv @ A_inv
print("性質 1: (AB)^-1 == B^-1 A^-1 ?", np.allclose(AB_inv, B_inv_A_inv))

# 性質 2: (A^T)^-1 = (A^-1)^T
AT_inv = np.linalg.inv(A.T)
A_inv_T = A_inv.T
print("性質 2: (A^T)^-1 == (A^-1)^T ?", np.allclose(AT_inv, A_inv_T))

# 性質 3: (A^-1)^-1 = A
A_inv_inv = np.linalg.inv(A_inv)
print("性質 3: (A^-1)^-1 == A ?", np.allclose(A_inv_inv, A))

assert np.allclose(AB_inv, B_inv_A_inv)
assert np.allclose(AT_inv, A_inv_T)
assert np.allclose(A_inv_inv, A)


# ---------------------------------------------------------------------------
# 5. LU 分解（不需 pivoting 的簡單範例，對照講義手算）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. LU 分解：不需 pivoting 的簡單範例")
print("=" * 60)

# 選擇第一列主元已是該行絕對值最大的元素，
# 這樣即使 scipy 預設採用 partial pivoting，也不會實際交換列，
# 方便與講義中的手算範例直接對照。
M = np.array([[6.0, 3.0],
              [4.0, 3.0]])
print("矩陣 M =")
print(M)

P, L, U = lu(M)
print("排列矩陣 P =")
print(P)
print("下三角矩陣 L =")
print(L)
print("上三角矩陣 U =")
print(U)

# scipy 慣例: M = P @ L @ U，等價於 L @ U = P.T @ M
check3 = np.allclose(L @ U, P.T @ M)
print("驗證 L @ U ≈ P.T @ M:", check3)
assert check3, "LU 分解驗證失敗！"

# 因為此範例消去過程不需要交換列，P 應為單位矩陣
print("P 是否為單位矩陣（代表此例不需要 pivoting）:", np.allclose(P, np.eye(2)))

# 用 LU 驗證行列式計算：det(M) = det(L) * det(U)
det_M = np.linalg.det(M)
det_via_LU = np.linalg.det(L) * np.linalg.det(U) * np.linalg.det(P)
print("det(M) (直接計算) =", det_M)
print("det(L)*det(U)*det(P) (透過 LU 分解計算) =", det_via_LU)
print("兩者是否一致:", np.isclose(det_M, det_via_LU))


# ---------------------------------------------------------------------------
# 6. LU 分解：3×3 範例（含 permutation，展示一般情況）
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("6. LU 分解：3x3 範例（一般情況，含 permutation）")
print("=" * 60)

N = np.array([[1.0, 2.0, 3.0],
              [0.0, 1.0, 4.0],
              [5.0, 6.0, 0.0]])
print("矩陣 N =")
print(N)

P2, L2, U2 = lu(N)
print("排列矩陣 P =")
print(P2)
print("下三角矩陣 L =")
print(L2)
print("上三角矩陣 U =")
print(U2)

check4 = np.allclose(L2 @ U2, P2.T @ N)
print("驗證 L @ U ≈ P.T @ N:", check4)
assert check4, "3x3 LU 分解驗證失敗！"

# 應用：用 LU 分解快速解 Nx = b
b_vec = np.array([1.0, 2.0, 3.0])
x_via_lu = np.linalg.solve(N, b_vec)  # numpy 內部也是用 LU 分解求解
x_check = N @ x_via_lu
print("解 Nx = b, 其中 b =", b_vec)
print("解得 x =", x_via_lu)
print("驗證 N @ x ≈ b:", np.allclose(x_check, b_vec))


print()
print("=" * 60)
print("全部範例執行完畢，逆矩陣與 LU 分解驗證皆通過。")
print("=" * 60)
