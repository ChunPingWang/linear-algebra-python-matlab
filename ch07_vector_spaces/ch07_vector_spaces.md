# 第 7 章：向量空間、基底與秩

## 學習目標

讀完本章後，你應該能夠：

- 用直觀的方式理解向量空間的核心規則（封閉性）
- 判斷一個集合是否為子空間 (subspace)
- 求出矩陣的零空間 (null space) 與行空間 (column space)，並理解它們的幾何意義
- 理解基底 (basis) 與維度 (dimension) 的定義
- 用秩-零度定理 (Rank-Nullity Theorem) 連結秩、零度與矩陣欄數的關係
- 手算一個具體矩陣的零空間基底、行空間基底與秩

## 概念說明

### 1. 向量空間的公理（直觀版）

**向量空間 (vector space)** 是一個集合 $V$，裡面的元素（向量）可以做「加法」與「純量乘法」，且滿足 8 條公理（交換律、結合律、分配律、存在零向量與反向量等）。對初學者而言，不需要逐條背誦證明，只要記住兩個核心規則：

- **加法封閉**：若 $\mathbf{u}, \mathbf{v} \in V$，則 $\mathbf{u} + \mathbf{v} \in V$
- **純量乘法封閉**：若 $\mathbf{u} \in V$，$c$ 為任意純量，則 $c\mathbf{u} \in V$

簡單說：在向量空間裡「怎麼加、怎麼縮放，都不會跑出這個空間」。我們最常用的向量空間就是 $\mathbb{R}^n$（例如 $\mathbb{R}^2$ 平面、$\mathbb{R}^3$ 空間），它們天生滿足所有公理，所以本章會把重點放在 $\mathbb{R}^n$ 裡的**子空間**上。

### 2. 子空間 (Subspace)

$\mathbb{R}^n$ 中的一個子集合 $W$，如果本身也構成一個向量空間，就稱為 $\mathbb{R}^n$ 的**子空間**。判斷 $W$ 是否為子空間，只需檢查三個條件：

1. **零向量在 $W$ 中**（$\mathbf{0} \in W$）
2. **加法封閉**：$\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
3. **純量乘法封閉**：$\mathbf{u} \in W,\ c \in \mathbb{R} \Rightarrow c\mathbf{u} \in W$

**判斷技巧**：只要子空間是「通過原點」的直線、平面，或更高維的「平坦」集合，通常就滿足以上條件。

- $W = \{(x, y) : y = 2x\}$（通過原點的直線）**是**子空間。
- $W' = \{(x, y) : y = 2x + 1\}$（不過原點的直線）**不是**子空間，因為 $(0,0) \notin W'$（零向量不在集合裡，條件 1 就失敗了）。

**結論**：任何不通過原點的集合，一定不是子空間。

### 3. 零空間 (Null Space / Kernel)

給定一個 $m \times n$ 矩陣 $A$，它的**零空間**定義為：

$$
\text{Null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}
$$

也就是所有能讓 $A\mathbf{x} = \mathbf{0}$ 成立的解向量 $\mathbf{x}$ 所形成的集合。零空間永遠是 $\mathbb{R}^n$ 的一個子空間（可以驗證：$\mathbf{0}$ 一定在其中，且滿足加法與純量乘法封閉）。

**幾何意義**：零空間告訴我們「哪些方向會被矩陣 $A$ 壓縮成 0」。如果零空間只有 $\{\mathbf{0}\}$，代表 $A$ 不會把任何非零向量壓扁成原點（$A$ 的各欄線性獨立）；如果零空間維度大於 0，代表存在某些方向，$A$ 作用後會讓它們「消失」（塌縮成原點），這也對應到 $A\mathbf{x} = \mathbf{b}$ 若有解則有無窮多解的情況。

### 4. 行空間 (Column Space)

矩陣 $A$ 的**行空間**（欄空間）是 $A$ 所有欄向量的線性組合所形成的集合：

$$
\text{Col}(A) = \text{span}\{\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n\}
$$

其中 $\mathbf{a}_1, \dots, \mathbf{a}_n$ 是 $A$ 的各欄向量。

**與 $A\mathbf{x} = \mathbf{b}$ 的關係**：方程式 $A\mathbf{x} = \mathbf{b}$ 有解，若且唯若 $\mathbf{b} \in \text{Col}(A)$。這是因為 $A\mathbf{x}$ 本質上就是欄向量的線性組合（係數是 $\mathbf{x}$ 的分量），所以「$A\mathbf{x} = \mathbf{b}$ 有解」等價於「$\mathbf{b}$ 可以被 $A$ 的欄向量線性組合出來」。

### 5. 基底 (Basis) 與維度 (Dimension)

給定一個向量空間（或子空間）$V$，一組向量 $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ 稱為 $V$ 的**基底**，若同時滿足：

1. **線性獨立**：沒有向量是其他向量的線性組合（沒有「浪費」的方向）
2. **生成 (span) $V$**：$V$ 中任何向量都能表示成這組向量的線性組合

一個空間的基底不唯一，但基底中向量的「個數」是固定的，這個數字稱為該空間的**維度 (dimension)**。

- $\text{rank}(A)$ = 行空間的維度 = 主元 (pivot) 的數量
- $\text{nullity}(A)$ = 零空間的維度 = 自由變數的數量

### 6. 秩-零度定理 (Rank-Nullity Theorem)

對於任意 $m \times n$ 矩陣 $A$：

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

其中 $n$ 是 $A$ 的**欄數**。直觀理解：矩陣的 $n$ 個欄位（也就是 $\mathbb{R}^n$ 的 $n$ 個自由度），一部分變成「主元欄」貢獻給行空間的維度（$\text{rank}(A)$），剩下的變成「自由變數欄」貢獻給零空間的維度（$\text{nullity}(A)$）。兩者相加，正好等於總欄數 $n$。

### 手算範例：求零空間基底、行空間基底、秩，並驗證秩-零度定理

給定矩陣：

$$
A = \begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 5 \end{bmatrix}
$$

這是一個 $3 \times 4$ 矩陣（$n = 4$ 欄）。

**步驟 1：高斯消去法化簡為簡化列梯形式 (RREF)**

$$
R_3 \leftarrow R_3 - R_1:\quad
\begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 1 & 2 \end{bmatrix}
$$

$$
R_3 \leftarrow R_3 - R_2:\quad
\begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

這已經是簡化列梯形式：主元位於第 1 欄與第 3 欄；第 2、4 欄沒有主元，是自由變數欄。

**步驟 2：求秩**

主元數量 = 2，所以 $\text{rank}(A) = 2$。

**步驟 3：求零空間基底**

由 RREF 寫出方程式（設 $x_2 = s$、$x_4 = t$ 為自由變數）：

$$
x_1 + 2x_2 + 3x_4 = 0 \Rightarrow x_1 = -2s - 3t
$$
$$
x_3 + 2x_4 = 0 \Rightarrow x_3 = -2t
$$

於是通解為：

$$
\mathbf{x} = s\begin{bmatrix}-2\\1\\0\\0\end{bmatrix} + t\begin{bmatrix}-3\\0\\-2\\1\end{bmatrix}
$$

**零空間基底**：

$$
\left\{ \begin{bmatrix}-2\\1\\0\\0\end{bmatrix}, \begin{bmatrix}-3\\0\\-2\\1\end{bmatrix} \right\}
$$

所以 $\text{nullity}(A) = 2$。

**步驟 4：求行空間基底**

主元欄位是第 1、3 欄，取**原矩陣** $A$（不是 RREF！）對應的第 1、3 欄，作為行空間基底：

$$
\left\{ \begin{bmatrix}1\\0\\1\end{bmatrix}, \begin{bmatrix}0\\1\\1\end{bmatrix} \right\}
$$

**步驟 5：驗證秩-零度定理**

$$
\text{rank}(A) + \text{nullity}(A) = 2 + 2 = 4 = n
$$

驗證成立！（$A$ 有 4 欄，$n = 4$）

## Python 實作

本章使用 `numpy.linalg.matrix_rank` 求秩，用 `scipy.linalg.null_space` 求零空間基底，並手刻高斯消去法找出行空間基底所對應的原始欄位。完整程式碼請見 [`ch07_vector_spaces.py`](ch07_vector_spaces.py)，以下節錄核心片段：

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([
    [1, 2, 0, 3],
    [0, 0, 1, 2],
    [1, 2, 1, 5],
], dtype=float)

# 1. 求秩
rank_A = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank_A}")

# 2. 求零空間基底
ns = null_space(A)
nullity = ns.shape[1]
print(f"nullity(A) = {nullity}")
print("零空間基底（欄向量）：\n", ns)

# 3. 求行空間基底：先用 RREF 找主元欄位，再取原矩陣對應欄
#    （rref_with_pivots 為手刻函式，詳見完整程式碼）
rref_A, pivot_cols = rref_with_pivots(A)
col_space_basis = A[:, pivot_cols]
print("行空間基底（取原矩陣主元欄）：\n", col_space_basis)

# 4. 驗證秩-零度定理
n = A.shape[1]
assert rank_A + nullity == n
print(f"驗證通過：rank(A) + nullity(A) = {rank_A} + {nullity} = {n}")
```

執行結果摘要：

```
rank(A) = 2
nullity(A) = 2
主元欄位 (0-indexed): [0, 2]  -> 對應原矩陣第 1、3 欄
行空間基底 = { [1,0,1], [0,1,1] }
秩-零度定理驗證通過：2 + 2 = 4 = n
```

程式碼另外用第二個矩陣 $C$（$4\times3$，秩 2）與第三個矩陣 $D$（$3\times3$ 滿秩）重複驗證秩-零度定理，皆通過。

## MATLAB 實作

MATLAB 提供內建函式 `rank()` 與 `null()`，可以直接對矩陣求秩與零空間基底。完整程式碼請見 [`ch07_vector_spaces.m`](ch07_vector_spaces.m)，以下節錄核心片段：

```matlab
A = [1 2 0 3;
     0 0 1 2;
     1 2 1 5];

% 1. 求秩
rank_A = rank(A);
fprintf("rank(A) = %d\n", rank_A);

% 2. 求零空間基底（'r' 選項回傳有理數形式，與手算結果一致）
ns = null(A, 'r');
nullity = size(ns, 2);
fprintf("nullity(A) = %d\n", nullity);
disp("零空間基底（欄向量）：");
disp(ns);

% 3. 求行空間基底：用 rref 找主元欄位，再取原矩陣對應欄
[R, pivot_cols] = rref(A);
col_space_basis = A(:, pivot_cols);
disp("行空間基底（取原矩陣主元欄）：");
disp(col_space_basis);

% 4. 驗證秩-零度定理
n = size(A, 2);
assert(rank_A + nullity == n);
fprintf("驗證通過：rank(A) + nullity(A) = %d + %d = %d\n", rank_A, nullity, n);
```

> 注意：本章 `.m` 檔案已用 GNU Octave 10.2 實際執行驗證通過，輸出數值與本章 Python 版本一致；尚未在正式 MATLAB 環境執行，但語法皆為標準 MATLAB 語法，建議你仍自行在 MATLAB 中重新執行一次確認。

## 重點整理

- **向量空間**的核心是「封閉性」：加法與純量乘法的結果都留在同一個集合中。
- **子空間**必須包含零向量，且對加法與純量乘法封閉；任何不過原點的集合都不是子空間。
- **零空間** $\text{Null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$：描述「被 $A$ 壓縮成原點」的方向。
- **行空間** $\text{Col}(A)$：$A$ 的欄向量張成的空間；$A\mathbf{x} = \mathbf{b}$ 有解 $\iff \mathbf{b} \in \text{Col}(A)$。
- **基底**：線性獨立且能生成整個空間的向量集合；**維度**是基底中向量的個數。
- **秩-零度定理**：$\text{rank}(A) + \text{nullity}(A) = n$（$n$ 為欄數），且 $\text{rank}(A)$ = 主元欄數 = 行空間維度，$\text{nullity}(A)$ = 自由變數數 = 零空間維度。
- 求行空間基底時，要取**原矩陣**中主元欄位對應的欄向量，而不是 RREF 化簡後的欄向量（RREF 只是用來「定位」哪些欄是主元欄）。

## 練習題

1. 判斷下列集合是否為 $\mathbb{R}^2$ 的子空間，並說明理由：
   $W = \{(x, y) : x + y = 0\}$

   > **提示**：檢查零向量是否在集合中，再檢查加法與純量乘法封閉性。答案：是子空間（通過原點的直線 $y = -x$）。

2. 判斷下列集合是否為 $\mathbb{R}^2$ 的子空間：
   $W = \{(x, y) : x \geq 0\}$（第一、四象限）

   > **提示**：純量乘法封閉性是否成立？若 $c = -1$ 呢？答案：不是子空間（乘上負數會跑出集合，例如 $(1,0)$ 乘 $-1$ 得 $(-1,0)$ 不在 $W$ 中）。

3. 給定矩陣 $B = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$，求 $\text{Null}(B)$ 的基底與 $\text{Col}(B)$ 的基底，並驗證秩-零度定理。

   > **提示**：第二列是第一列的 2 倍，秩為 1。答案：$\text{rank}(B) = 1$，零空間基底為 $\{(-2, 1)^T\}$（nullity = 1），行空間基底為 $\{(1, 2)^T\}$，驗證 $1 + 1 = 2 = n$。

4. 給定矩陣 $E = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \end{bmatrix}$，這是一個 $2 \times 3$ 矩陣。求秩、零空間基底、行空間基底，並驗證秩-零度定理。

   > **提示**：矩陣已經是 RREF 形式，主元在第 1、2 欄，第 3 欄是自由變數欄。答案：$\text{rank}(E) = 2$，零空間基底為 $\{(-2, -3, 1)^T\}$（nullity = 1），行空間基底為 $\{(1,0)^T, (0,1)^T\}$，驗證 $2 + 1 = 3 = n$。

5. 承第 4 題，請問 $\text{Col}(E)$ 等於整個 $\mathbb{R}^2$ 嗎？換句話說，對任意 $\mathbf{b} \in \mathbb{R}^2$，方程式 $E\mathbf{x} = \mathbf{b}$ 是否一定有解？

   > **提示**：行空間的維度是否等於 $\mathbb{R}^2$ 的維度？答案：是的，因為 $\text{rank}(E) = 2$ 等於列數（也是 $\mathbb{R}^2$ 的維度），行空間基底 $\{(1,0)^T, (0,1)^T\}$ 已經生成整個 $\mathbb{R}^2$，所以對任意 $\mathbf{b}$，$E\mathbf{x} = \mathbf{b}$ 恆有解（事實上因為有自由變數，會有無窮多解）。
