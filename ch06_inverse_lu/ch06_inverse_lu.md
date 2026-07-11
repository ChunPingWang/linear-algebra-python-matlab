# 第 6 章：逆矩陣與 LU 分解

## 學習目標

讀完本章後，你應該能夠：

- 說出逆矩陣的定義，以及一個方陣「可逆（invertible / non-singular）」的條件
- 手算 2×2 矩陣的逆矩陣公式
- 使用高斯-若丹消去法（Gauss-Jordan elimination）手算 3×3 矩陣的逆矩陣
- 說明並運用逆矩陣的四個基本性質
- 理解 LU 分解 A = LU 的意義，以及它為什麼能加快「多組 Ax = b」與行列式的計算
- 手算一個不需要 pivoting 的簡單 LU 分解範例
- 用 Python（NumPy / SciPy）與 MATLAB 分別計算逆矩陣與 LU 分解，並驗證結果

## 概念說明

### 1. 逆矩陣的定義

給定一個 $n \times n$ 的方陣 $A$，如果存在另一個 $n \times n$ 矩陣 $A^{-1}$，使得

$$
A A^{-1} = A^{-1} A = I
$$

（$I$ 為 $n \times n$ 單位矩陣），則稱 $A^{-1}$ 為 $A$ 的**逆矩陣（inverse matrix）**，並稱 $A$ 是**可逆的（invertible）**或**非奇異的（non-singular）**。

**存在條件**：$A^{-1}$ 存在的充要條件是

$$
\det(A) \neq 0
$$

- 只有**方陣**才可能有逆矩陣（非方陣不在討論範圍內）。
- 若 $\det(A) = 0$，則稱 $A$ 為**奇異矩陣（singular matrix）**，此時 $A^{-1}$ **不存在**。
- 這與第 5 章「行列式」的概念直接相關：行列式告訴我們矩陣是否可逆。

直覺理解：逆矩陣就像數字的倒數。對純量而言，$a \cdot a^{-1} = 1$（但 $a = 0$ 時沒有倒數）；對矩陣而言，$A \cdot A^{-1} = I$（但 $\det(A) = 0$ 時沒有逆矩陣）。

### 2. 2×2 逆矩陣公式

對於一個 2×2 矩陣

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

其逆矩陣公式為：

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix} d & -b \\ -c & a \end{bmatrix},
\quad \det(A) = ad - bc
$$

也就是：**對角線元素互換位置，副對角線元素變號，整體再除以行列式**。

**手算範例**

設

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}
$$

**第一步：算行列式**

$$
\det(A) = (2)(1) - (1)(1) = 2 - 1 = 1
$$

因為 $\det(A) = 1 \neq 0$，所以 $A$ 可逆。

**第二步：套公式**

$$
A^{-1} = \frac{1}{1}
\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}
=
\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}
$$

**第三步：驗證** $A A^{-1} = I$

$$
A A^{-1} =
\begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}
=
\begin{bmatrix} 2 \cdot 1 + 1 \cdot(-1) & 2\cdot(-1)+1\cdot 2 \\ 1\cdot 1 + 1\cdot(-1) & 1\cdot(-1)+1\cdot 2 \end{bmatrix}
=
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

驗證成功，結果為單位矩陣 $I$。

### 3. 用高斯-若丹消去法求 3×3 逆矩陣

對於較大的矩陣（3×3 以上），常用**高斯-若丹消去法（Gauss-Jordan elimination）**：把 $[A \mid I]$ 這個增廣矩陣，透過列運算化簡成 $[I \mid A^{-1}]$。

**手算範例**

設

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{bmatrix}
$$

**第一步：寫出增廣矩陣 $[A \mid I]$**

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
5 & 6 & 0 & 0 & 0 & 1
\end{array}\right]
$$

**第二步：消去第一行下方元素** —— $R_3 \leftarrow R_3 - 5R_1$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & -4 & -15 & -5 & 0 & 1
\end{array}\right]
$$

**第三步：消去第二行的上下元素**

$R_1 \leftarrow R_1 - 2R_2$：

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & -5 & 1 & -2 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & -4 & -15 & -5 & 0 & 1
\end{array}\right]
$$

$R_3 \leftarrow R_3 + 4R_2$：

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & -5 & 1 & -2 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

**第四步：消去第三行的上方元素**

$R_1 \leftarrow R_1 + 5R_3$：

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

$R_2 \leftarrow R_2 - 4R_3$：

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

**第五步：讀出結果**

左半邊已經化成單位矩陣 $I$，右半邊就是 $A^{-1}$：

$$
A^{-1} = \begin{bmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{bmatrix}
$$

（可自行代入 $A A^{-1} = I$ 驗證；下方 Python 實作也會用程式驗證這個結果。）

### 4. 逆矩陣的性質

設 $A$、$B$ 皆為同階可逆方陣，以下性質經常用來簡化計算：

1. **乘積的逆**：$(AB)^{-1} = B^{-1} A^{-1}$（順序要反過來，就像穿脫衣服要反序）
2. **轉置的逆**：$(A^\top)^{-1} = (A^{-1})^\top$
3. **逆的逆**：$(A^{-1})^{-1} = A$
4. （補充）純量倍數的逆：$(kA)^{-1} = \frac{1}{k} A^{-1}$，其中 $k \neq 0$

這些性質在推導公式或化簡矩陣方程式時非常有用，例如求解 $ABx = c$ 時可以寫成 $x = B^{-1}A^{-1}c = (AB)^{-1}c$。

### 5. LU 分解的概念

**LU 分解（LU decomposition）**是把一個方陣 $A$ 拆解成

$$
A = LU
$$

其中：

- $L$（Lower）是**下三角矩陣**，對角線通常為 1（稱為單位下三角矩陣）
- $U$（Upper）是**上三角矩陣**，也就是高斯消去法化簡後得到的列梯形矩陣

**為什麼要做 LU 分解？**

1. **快速解多組 $Ax=b$**：如果同一個 $A$ 要對很多個不同的 $b$ 求解 $Ax=b$，每次都重新做高斯消去很浪費。有了 $A = LU$ 之後，只要做兩次簡單的三角形方程組求解：
   - 先解 $Ly = b$（前代法 forward substitution，因為 $L$ 是下三角）
   - 再解 $Ux = y$（回代法 back substitution，因為 $U$ 是上三角）
   
   三角形方程組的求解非常快，不需要每次都重跑整個消去過程。

2. **更有效率地計算行列式**：因為 $\det(A) = \det(L)\det(U)$，而三角矩陣的行列式就是對角線元素相乘，所以

   $$
   \det(A) = \left(\prod_i L_{ii}\right)\left(\prod_i U_{ii}\right)
   $$

   當 $L$ 為單位下三角（對角線全為 1）時，$\det(A) = \prod_i U_{ii}$，比用餘因子展開快得多。

### 6. LU 分解手算範例（不需 pivoting）

設

$$
A = \begin{bmatrix} 6 & 3 \\ 4 & 3 \end{bmatrix}
$$

我們用高斯消去法把 $A$ 化為上三角矩陣 $U$，同時記錄消去過程中用到的乘數，組成 $L$。

**第一步：消去 $A_{21}$**

乘數（multiplier）$m_{21} = \dfrac{4}{6} = \dfrac{2}{3}$，也就是 $R_2 \leftarrow R_2 - \dfrac{2}{3} R_1$：

$$
\begin{bmatrix} 6 & 3 \\ 4 & 3 \end{bmatrix}
\xrightarrow{R_2 - \frac{2}{3}R_1}
\begin{bmatrix} 6 & 3 \\ 0 & 1 \end{bmatrix} = U
$$

**第二步：組出 $L$**

$L$ 的對角線為 1，非對角線元素就是消去過程用到的乘數 $m_{21}$（注意：不變號，直接填入）：

$$
L = \begin{bmatrix} 1 & 0 \\ \frac{2}{3} & 1 \end{bmatrix}
$$

**第三步：驗證 $A = LU$**

$$
LU =
\begin{bmatrix} 1 & 0 \\ \frac{2}{3} & 1 \end{bmatrix}
\begin{bmatrix} 6 & 3 \\ 0 & 1 \end{bmatrix}
=
\begin{bmatrix} 6 & 3 \\ \frac{2}{3}\times6+0 & \frac{2}{3}\times3+1 \end{bmatrix}
=
\begin{bmatrix} 6 & 3 \\ 4 & 3 \end{bmatrix}
= A
$$

驗證成功。

**驗證行列式**：$\det(A) = 6\times3 - 3\times4 = 18-12=6$；而 $\det(L)\det(U) = 1 \times (6 \times 1) = 6$，兩者一致。

> **注意**：本範例刻意選擇第一列主元（pivot）已經是該行絕對值最大的元素，所以消去過程完全不需要交換列（row swap），$A = LU$ 因此成立。若消去過程中主元為 0，或不是該行絕對值最大的元素，數值計算上通常會交換列以提升穩定性（稱為 partial pivoting），這時要引入一個排列矩陣 $P$，寫成 $PA = LU$。下方 Python 實作使用 `scipy.linalg.lu`，它**預設一律採用 partial pivoting**（即使理論上不換列也可能換），並回傳滿足 $A = PLU$ 的 $P, L, U$（等價於 $P^\top A = LU$）；本範例矩陣的寫法確保 scipy 實際計算時 $P$ 剛好是單位矩陣，方便與手算結果直接對照。

## Python 實作

完整程式碼請見同資料夾的 [`ch06_inverse_lu.py`](./ch06_inverse_lu.py)，重點摘要如下：

```python
import numpy as np
from scipy.linalg import lu

# 1. 用 numpy.linalg.inv 求逆矩陣
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)
A_inv = np.linalg.inv(A)

# 驗證 A @ A_inv ≈ I
I3 = np.eye(3)
print(np.allclose(A @ A_inv, I3))  # True

# 2. 用 scipy.linalg.lu 做 LU 分解（含 permutation）
B = np.array([[6, 3],
              [4, 3]], dtype=float)
P, L, U = lu(B)

# 驗證 L @ U ≈ P @ A（scipy 的慣例：A = P @ L @ U，即 P.T @ A = L @ U）
print(np.allclose(L @ U, P.T @ B))  # True
```

**關鍵重點**：

- `numpy.linalg.inv(A)` 直接回傳逆矩陣；若 `A` 為奇異矩陣，會拋出 `LinAlgError`。
- `scipy.linalg.lu(A)` 回傳 `P, L, U` 三個矩陣，且遵循慣例 $A = P L U$（注意不是 $PA=LU$，而是 $P^\top A = LU$，兩者等價，只是 $P$ 擺放位置不同，程式中務必看清楚）。
- 用 `np.allclose` 而非 `==` 來比較浮點數矩陣，避免因浮點誤差誤判失敗。

## MATLAB 實作

完整程式碼請見同資料夾的 [`ch06_inverse_lu.m`](./ch06_inverse_lu.m)，重點摘要如下：

```matlab
% 1. 用 inv() 求逆矩陣
A = [1 2 3; 0 1 4; 5 6 0];
A_inv = inv(A);

% 驗證 A * A_inv ≈ I
I3 = eye(3);
disp(norm(A * A_inv - I3) < 1e-10);  % 1 (true)

% 2. 用 lu() 做 LU 分解（含 permutation）
B = [4 3; 6 3];
[L, U, P] = lu(B);

% 驗證 L * U ≈ P * B
disp(norm(L * U - P * B) < 1e-10);  % 1 (true)
```

**關鍵重點**：

- MATLAB 的 `inv(A)` 與 Python 的 `numpy.linalg.inv(A)` 用法幾乎一樣。
- MATLAB 的 `[L, U, P] = lu(A)` 回傳順序與 SciPy 不同（`L, U, P` 而非 `P, L, U`），且遵循慣例 $P A = L U$，使用時要特別留意順序與慣例差異。
- 浮點數比較建議用 `norm(差) < 容忍值` 而不是直接用 `==`。

## 重點整理

- 逆矩陣 $A^{-1}$ 滿足 $AA^{-1}=A^{-1}A=I$，存在的充要條件是 $A$ 為方陣且 $\det(A)\neq 0$（non-singular）。
- 2×2 逆矩陣公式：$A^{-1} = \dfrac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$。
- 高斯-若丹消去法求逆矩陣：對增廣矩陣 $[A\mid I]$ 做列運算，化簡至 $[I\mid A^{-1}]$。
- 逆矩陣性質：$(AB)^{-1}=B^{-1}A^{-1}$、$(A^\top)^{-1}=(A^{-1})^\top$、$(A^{-1})^{-1}=A$。
- LU 分解 $A=LU$（$L$ 下三角、$U$ 上三角），可加速「多組 $Ax=b$」與行列式計算；若消去過程需要交換列，則寫成 $PA=LU$。
- Python 用 `numpy.linalg.inv` 與 `scipy.linalg.lu`；MATLAB 用 `inv()` 與 `lu()`，但回傳順序與慣例略有差異，務必留意。

## 練習題

1. 求 $A = \begin{bmatrix} 3 & 2 \\ 1 & 1 \end{bmatrix}$ 的逆矩陣，並驗證 $AA^{-1}=I$。

   **提示**：先算 $\det(A) = 3(1)-2(1)=1$，再套用 2×2 公式。

2. 判斷 $B = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}$ 是否可逆？為什麼？

   **提示**：先算行列式，$\det(B)=2(2)-4(1)=0$，因此……

3. 用高斯-若丹消去法求 $C = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 4 \end{bmatrix}$ 的逆矩陣。

   **提示**：對角矩陣的逆矩陣，就是把對角線元素各自取倒數，可用來驗證你的消去法答案是否正確。

4. 已知 $A$、$B$ 皆為 2×2 可逆矩陣，且 $A^{-1} = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$、$B^{-1} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$，求 $(AB)^{-1}$。

   **提示**：利用性質 $(AB)^{-1}=B^{-1}A^{-1}$，直接做矩陣乘法即可，不需要先求出 $A$、$B$ 本身。

5. 對 $D = \begin{bmatrix} 2 & 1 \\ 4 & 5 \end{bmatrix}$ 做不需 pivoting 的 LU 分解，寫出 $L$ 與 $U$，並驗證 $\det(D) = \det(L)\det(U)$。

   **提示**：乘數 $m_{21} = 4/2 = 2$，$R_2 \leftarrow R_2 - 2R_1$ 消去左下角元素，即可得到 $U$；$L$ 的非對角線元素直接填入 $m_{21}$。
