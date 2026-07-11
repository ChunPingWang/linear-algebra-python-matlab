# 第 3 章：矩陣與矩陣運算

## 學習目標

讀完本章後，你應該能夠：

- 說出矩陣的定義、記號，並讀出矩陣的形狀（shape，$m \times n$）
- 進行矩陣加法與純量乘法
- 依照維度規則判斷兩個矩陣能否相乘，並手算矩陣乘法
- 解釋為什麼矩陣乘法「一般不滿足交換律」
- 計算矩陣的轉置，並辨認零矩陣、單位矩陣、對角矩陣、對稱矩陣、上/下三角矩陣
- 計算矩陣的跡（trace），並使用跡的線性與 $\mathrm{trace}(AB) = \mathrm{trace}(BA)$ 性質

## 概念說明

### 1. 矩陣的定義與記號

矩陣（matrix）是一個由數字排成的矩形陣列，寫成 $m$ 列（row）、$n$ 行（column）的形式，稱為 $m \times n$ 矩陣（讀作「$m$ by $n$」）：

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

其中 $a_{ij}$ 表示第 $i$ 列、第 $j$ 行的元素（entry）。矩陣的**形狀（shape）**就是 $(m, n)$：$m$ 是列數，$n$ 是行數。

> 記憶口訣：「列先行後」——先數有幾「列」（橫的），再數有幾「行」（直的）。

例如：

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
$$

是一個 $2 \times 3$ 矩陣（2 列、3 行），其中 $a_{12} = 2$、$a_{23} = 6$。

向量其實是矩陣的特例：$n \times 1$ 矩陣稱為行向量（column vector），$1 \times n$ 矩陣稱為列向量（row vector）。

### 2. 矩陣加法與純量乘法

**矩陣加法**：兩個矩陣形狀必須相同，才能相加，做法是對應元素相加。

$$
A + B = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} + \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix} = \begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix}
$$

例如：

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

**純量乘法**：純量 $c$ 乘上矩陣 $A$，就是把每個元素都乘上 $c$：

$$
cA = c\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} = \begin{bmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{bmatrix}
$$

矩陣加法與純量乘法都滿足交換律、結合律、分配律，性質與向量運算完全一致（矩陣其實就是把數字排成表格的「廣義向量」）。

### 3. 矩陣乘法

矩陣乘法**不是**對應元素相乘，而是有特定的定義規則。

**維度規則**：若 $A$ 是 $m \times n$ 矩陣，$B$ 是 $n \times p$ 矩陣（$A$ 的行數必須等於 $B$ 的列數），則乘積 $C = AB$ 是一個 $m \times p$ 矩陣：

$$
\underbrace{A}_{m \times n} \; \underbrace{B}_{n \times p} = \underbrace{C}_{m \times p}
$$

**元素定義**：$C$ 的第 $i$ 列第 $j$ 行元素，是 $A$ 的第 $i$ 列與 $B$ 的第 $j$ 行的內積：

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
$$

#### 手算範例：完整步驟

設

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}_{2\times 2}, \qquad
B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}_{2\times 2}
$$

因為 $A$ 是 $2\times 2$、$B$ 是 $2\times 2$，內側維度 $2=2$ 相符，所以 $C = AB$ 存在且為 $2\times 2$。

**第一步**：算 $c_{11}$（$A$ 第 1 列 與 $B$ 第 1 行的內積）

$$
c_{11} = (1)(5) + (2)(7) = 5 + 14 = 19
$$

**第二步**：算 $c_{12}$（$A$ 第 1 列 與 $B$ 第 2 行的內積）

$$
c_{12} = (1)(6) + (2)(8) = 6 + 16 = 22
$$

**第三步**：算 $c_{21}$（$A$ 第 2 列 與 $B$ 第 1 行的內積）

$$
c_{21} = (3)(5) + (4)(7) = 15 + 28 = 43
$$

**第四步**：算 $c_{22}$（$A$ 第 2 列 與 $B$ 第 2 行的內積）

$$
c_{22} = (3)(6) + (4)(8) = 18 + 32 = 50
$$

**結果**：

$$
AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

#### 矩陣乘法不滿足交換律

用同樣的 $A, B$ 反過來算 $BA$：

$$
BA = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
= \begin{bmatrix} (5)(1)+(6)(3) & (5)(2)+(6)(4) \\ (7)(1)+(8)(3) & (7)(2)+(8)(4) \end{bmatrix}
= \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}
$$

顯然 $AB \neq BA$。**這是矩陣乘法與一般數字乘法最大的不同：一般來說 $AB \neq BA$**，因此矩陣乘法「不滿足交換律」。順序很重要！

（矩陣乘法仍滿足結合律 $(AB)C = A(BC)$ 與對加法的分配律 $A(B+C) = AB+AC$。）

### 4. 轉置矩陣

矩陣 $A$ 的轉置（transpose），記為 $A^T$，是把 $A$ 的列與行互換：若 $A$ 是 $m \times n$，則 $A^T$ 是 $n \times m$，且 $(A^T)_{ij} = A_{ji}$。

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Longrightarrow \quad
A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

常用性質：

- $(A^T)^T = A$
- $(A+B)^T = A^T + B^T$
- $(cA)^T = cA^T$
- $(AB)^T = B^T A^T$（順序會反過來！）

### 5. 特殊矩陣

| 名稱 | 定義 | 範例 |
|---|---|---|
| **零矩陣**（zero matrix）$O$ | 所有元素皆為 0 | $\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$ |
| **單位矩陣**（identity matrix）$I$ | 主對角線為 1，其餘為 0 的方陣；滿足 $AI = IA = A$ | $I_3=\begin{bmatrix} 1&0&0\\0&1&0\\0&0&1 \end{bmatrix}$ |
| **對角矩陣**（diagonal matrix） | 只有主對角線可能非零，其餘皆為 0 | $\begin{bmatrix} 2&0&0\\0&5&0\\0&0&-1 \end{bmatrix}$ |
| **對稱矩陣**（symmetric matrix） | 方陣且滿足 $A^T = A$ | $\begin{bmatrix} 1&2\\2&3 \end{bmatrix}$ |
| **上三角矩陣**（upper triangular） | 主對角線下方元素皆為 0 | $\begin{bmatrix} 1&2&3\\0&5&6\\0&0&9 \end{bmatrix}$ |
| **下三角矩陣**（lower triangular） | 主對角線上方元素皆為 0 | $\begin{bmatrix} 1&0&0\\4&5&0\\7&8&9 \end{bmatrix}$ |

單位矩陣 $I$ 在矩陣世界中扮演的角色，就像數字 $1$ 在一般乘法中的角色（乘上去不改變原矩陣）。

### 6. 跡（Trace）

**跡**（trace）只對方陣（$n \times n$）定義，是主對角線元素的總和：

$$
\mathrm{trace}(A) = \sum_{i=1}^{n} a_{ii}
$$

例如 $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$，則 $\mathrm{trace}(A) = 1 + 4 = 5$。

**跡的重要性質**：

1. **線性**：$\mathrm{trace}(A+B) = \mathrm{trace}(A) + \mathrm{trace}(B)$，且 $\mathrm{trace}(cA) = c\,\mathrm{trace}(A)$
2. **轉置不變**：$\mathrm{trace}(A^T) = \mathrm{trace}(A)$
3. **乘積的循環性**：$\mathrm{trace}(AB) = \mathrm{trace}(BA)$（即使 $AB \neq BA$，兩者的跡卻相等！）

以上一節的 $A, B$ 為例：$\mathrm{trace}(AB) = 19+50 = 69$，而 $\mathrm{trace}(BA) = 23+46 = 69$，兩者確實相等，即使 $AB \ne BA$ 這兩個矩陣本身完全不同。

## Python 實作

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C1 = A @ B          # 建議用法：@ 運算子
C2 = np.matmul(A, B)  # 等價寫法
```

完整程式碼請見 [`ch03_matrices.py`](ch03_matrices.py)，可直接執行：

```bash
python ch03_matrices/ch03_matrices.py
```

## MATLAB 實作

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];

C = A * B;      % 矩陣乘法用 *
At = A';        % 轉置用 '
I = eye(2);     % 單位矩陣
d = diag([2 5 -1]);  % 對角矩陣
t = trace(A);   % 跡
```

完整程式碼請見 [`ch03_matrices.m`](ch03_matrices.m)。

> 注意：本章 `.m` 檔案已用 GNU Octave 10.2 實際執行驗證通過，輸出數值與本章 Python 版本一致；尚未在正式 MATLAB 環境執行，但語法皆為標準 MATLAB 語法，建議你仍自行在 MATLAB 中重新執行一次確認。

## 重點整理

- 矩陣是 $m \times n$ 的數字陣列，$m$ 為列數、$n$ 為行數。
- 矩陣加法／純量乘法：形狀相同才能相加；純量乘法對每個元素作用。
- 矩陣乘法 $A_{m\times n} B_{n\times p} = C_{m\times p}$：內側維度必須相符，$c_{ij}$ 是 $A$ 第 $i$ 列與 $B$ 第 $j$ 行的內積。
- **矩陣乘法一般不滿足交換律**：$AB \neq BA$。
- 轉置 $A^T$ 把列與行互換，且 $(AB)^T = B^T A^T$。
- 特殊矩陣：零矩陣、單位矩陣 $I$、對角矩陣、對稱矩陣、上/下三角矩陣。
- 跡 $\mathrm{trace}(A)$ 是方陣主對角線元素之和，滿足線性且 $\mathrm{trace}(AB)=\mathrm{trace}(BA)$。

## 練習題

1. 設 $A = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}$，$B = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$。手算 $A+B$、$2A$、$AB$、$BA$，並確認 $AB \neq BA$。

   > 提示：$A+B=\begin{bmatrix}4&1\\2&1\end{bmatrix}$，$2A=\begin{bmatrix}2&0\\4&-2\end{bmatrix}$，$AB=\begin{bmatrix}3&1\\6&0\end{bmatrix}$，$BA=\begin{bmatrix}5&-1\\4&-2\end{bmatrix}$。

2. 若 $A$ 是 $3 \times 4$ 矩陣、$B$ 是 $4 \times 2$ 矩陣，$AB$ 的形狀是什麼？$BA$ 是否存在？

   > 提示：$AB$ 是 $3\times 2$；$BA$ 因內側維度 $2 \neq 3$ 不存在。

3. 求 $A = \begin{bmatrix} 2 & 7 \\ -1 & 5 \end{bmatrix}$ 的轉置 $A^T$，並驗證 $(A^T)^T = A$。

   > 提示：$A^T = \begin{bmatrix} 2 & -1 \\ 7 & 5 \end{bmatrix}$。

4. 判斷 $A = \begin{bmatrix} 4 & -1 & 2 \\ -1 & 3 & 0 \\ 2 & 0 & 5 \end{bmatrix}$ 是否為對稱矩陣，並求 $\mathrm{trace}(A)$。

   > 提示：$A^T = A$，是對稱矩陣；$\mathrm{trace}(A) = 4+3+5 = 12$。

5. 設 $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$，$B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$。分別計算 $\mathrm{trace}(AB)$ 與 $\mathrm{trace}(BA)$，驗證兩者相等。

   > 提示：$AB=\begin{bmatrix}2&1\\4&3\end{bmatrix}$，$\mathrm{trace}(AB)=5$；$BA=\begin{bmatrix}3&4\\1&2\end{bmatrix}$，$\mathrm{trace}(BA)=5$。
