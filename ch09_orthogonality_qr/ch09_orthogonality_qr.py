"""
第 9 章：正交性、Gram-Schmidt 與 QR 分解（Python 實作）

本腳本示範：
1. 正交向量的判斷（內積是否為 0）
2. 正交矩陣的性質：Q^T Q = Q Q^T = I，並驗證保長度變換
3. 手刻 Gram-Schmidt 正交化流程（含逐步印出每個正交化向量）
4. 用 np.linalg.qr 驗證手刻結果一致（比較正交性與張成空間）
5. 驗證 Q @ R ≈ A 與 Q.T @ Q ≈ I

執行方式：
    python ch09_orthogonality_qr.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. 正交向量的判斷
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. 正交向量的判斷")
print("=" * 60)

u = np.array([1, 0])
v = np.array([0, 1])
print("u =", u, ", v =", v)
print("u . v =", np.dot(u, v), "-> 正交" if np.isclose(np.dot(u, v), 0) else "-> 不正交")

a = np.array([1, 2])
b = np.array([3, -1])
print("\na =", a, ", b =", b)
print("a . b =", np.dot(a, b), "-> 正交" if np.isclose(np.dot(a, b), 0) else "-> 不正交")


# ---------------------------------------------------------------------------
# 2. 正交矩陣的性質
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. 正交矩陣的性質：Q^T Q = Q Q^T = I，且保持向量長度")
print("=" * 60)

theta = np.pi / 3  # 60 度旋轉矩陣
Q_rot = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])
print("旋轉矩陣 Q (theta=60°):\n", Q_rot)
print("Q^T Q:\n", Q_rot.T @ Q_rot)
print("Q Q^T:\n", Q_rot @ Q_rot.T)
print("Q^T Q ≈ I ?", np.allclose(Q_rot.T @ Q_rot, np.eye(2)))

x = np.array([3.0, 4.0])
Qx = Q_rot @ x
print("\n原向量 x =", x, ", |x| =", np.linalg.norm(x))
print("變換後 Qx =", Qx, ", |Qx| =", np.linalg.norm(Qx))
print("長度是否保持不變 ?", np.isclose(np.linalg.norm(x), np.linalg.norm(Qx)))


# ---------------------------------------------------------------------------
# 3. 手刻 Gram-Schmidt 正交化流程
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. 手刻 Gram-Schmidt 正交化流程")
print("=" * 60)


def gram_schmidt(A, verbose=False):
    """對矩陣 A（行向量為欲正交化的向量組）做 Gram-Schmidt 正交化與標準化。

    參數:
        A: shape (m, n) 的矩陣，n 個線性獨立的 m 維行向量
        verbose: 是否印出每一步的中間結果

    回傳:
        Q: shape (m, n)，行向量為標準正交基底
        R: shape (n, n)，上三角矩陣，滿足 A = Q @ R
    """
    m, n = A.shape
    U = np.zeros((m, n))   # 尚未標準化的正交向量 u_1, ..., u_n
    Q = np.zeros((m, n))   # 標準化後的 e_1, ..., e_n
    R = np.zeros((n, n))

    for j in range(n):
        a_j = A[:, j].astype(float).copy()
        u_j = a_j.copy()
        for i in range(j):
            coeff = Q[:, i] @ a_j  # 投影係數 = e_i . a_j
            R[i, j] = coeff
            u_j -= coeff * Q[:, i]
            if verbose:
                print(f"  a_{j+1} 在 e_{i+1} 方向上的投影係數 = {coeff:.4f}")
        norm_u = np.linalg.norm(u_j)
        R[j, j] = norm_u
        U[:, j] = u_j
        Q[:, j] = u_j / norm_u
        if verbose:
            print(f"  u_{j+1} (尚未標準化) = {u_j}")
            print(f"  ||u_{j+1}|| = {norm_u:.4f}")
            print(f"  e_{j+1} (標準化後) = {Q[:, j]}")
            print()

    return Q, R, U


# 使用 .md 教學文件中的手算範例
a1 = np.array([1, 1, 0])
a2 = np.array([2, 0, 2])
a3 = np.array([3, 3, 3])
A = np.column_stack([a1, a2, a3]).astype(float)

print("原始向量:")
print("a1 =", a1)
print("a2 =", a2)
print("a3 =", a3)
print("\n矩陣 A = [a1 a2 a3]:\n", A)

print("\n逐步執行 Gram-Schmidt:\n")
Q_manual, R_manual, U_manual = gram_schmidt(A, verbose=True)

print("正交化後（尚未標準化）的向量:")
print("u1 =", U_manual[:, 0])
print("u2 =", U_manual[:, 1])
print("u3 =", U_manual[:, 2])

print("\n驗證正交性（內積應為 0）:")
print("u1 . u2 =", np.dot(U_manual[:, 0], U_manual[:, 1]))
print("u1 . u3 =", np.dot(U_manual[:, 0], U_manual[:, 2]))
print("u2 . u3 =", np.dot(U_manual[:, 1], U_manual[:, 2]))

print("\n標準化後的標準正交基底 Q（各行向量）:")
print("e1 =", Q_manual[:, 0], ", ||e1|| =", np.linalg.norm(Q_manual[:, 0]))
print("e2 =", Q_manual[:, 1], ", ||e2|| =", np.linalg.norm(Q_manual[:, 1]))
print("e3 =", Q_manual[:, 2], ", ||e3|| =", np.linalg.norm(Q_manual[:, 2]))

print("\n手刻 Gram-Schmidt 得到的 R (上三角矩陣):\n", R_manual)


# ---------------------------------------------------------------------------
# 4. 用 np.linalg.qr 驗證手刻結果一致
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. 用 np.linalg.qr 驗證手刻 Gram-Schmidt 結果")
print("=" * 60)

Q_np, R_np = np.linalg.qr(A)

print("np.linalg.qr 得到的 Q:\n", Q_np)
print("\nnp.linalg.qr 得到的 R:\n", R_np)

# np.linalg.qr 與手刻結果的 Q 每一行可能差一個正負號（正交基底不唯一，
# 但張成的子空間相同），因此逐行比較「絕對值後」是否一致，
# 或比較兩者的正交性與是否張成同一空間，會是更穩健的驗證方式。
print("\n比較：手刻 Q 與 np.linalg.qr 的 Q 是否只差正負號（逐行絕對值比較）:")
same_up_to_sign = np.allclose(np.abs(Q_manual), np.abs(Q_np))
print("  ->", same_up_to_sign)

print("\n驗證手刻 Q 的行向量兩兩正交（Q_manual^T @ Q_manual ≈ I）:")
gram_manual = Q_manual.T @ Q_manual
print(gram_manual)
print("  ->", np.allclose(gram_manual, np.eye(3)))

print("\n驗證 np.linalg.qr 的 Q 的行向量兩兩正交（Q_np^T @ Q_np ≈ I）:")
gram_np = Q_np.T @ Q_np
print(gram_np)
print("  ->", np.allclose(gram_np, np.eye(3)))


# ---------------------------------------------------------------------------
# 5. 驗證 Q @ R ≈ A 與 Q.T @ Q ≈ I
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. 最終驗證：Q @ R ≈ A 與 Q.T @ Q ≈ I")
print("=" * 60)

print("手刻版本:")
print("  Q_manual @ R_manual =\n", Q_manual @ R_manual)
print("  是否等於原始 A ?", np.allclose(Q_manual @ R_manual, A))
print("  Q_manual.T @ Q_manual ≈ I ?", np.allclose(Q_manual.T @ Q_manual, np.eye(3)))

print("\nnp.linalg.qr 版本:")
print("  Q_np @ R_np =\n", Q_np @ R_np)
print("  是否等於原始 A ?", np.allclose(Q_np @ R_np, A))
print("  Q_np.T @ Q_np ≈ I ?", np.allclose(Q_np.T @ Q_np, np.eye(3)))

all_checks_passed = (
    np.allclose(Q_manual @ R_manual, A)
    and np.allclose(Q_manual.T @ Q_manual, np.eye(3))
    and np.allclose(Q_np @ R_np, A)
    and np.allclose(Q_np.T @ Q_np, np.eye(3))
    and same_up_to_sign
)

print()
print("=" * 60)
if all_checks_passed:
    print("所有驗證通過：手刻 Gram-Schmidt 與 np.linalg.qr 結果一致，")
    print("且 QR = A、Q^T Q = I 皆成立。")
else:
    print("警告：部分驗證未通過，請檢查程式碼。")
print("=" * 60)

print("\n全部範例執行完畢。")
