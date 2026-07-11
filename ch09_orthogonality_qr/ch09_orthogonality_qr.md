# 第 9 章：正交性、Gram-Schmidt 與 QR 分解

## 學習目標

讀完本章後，你應該能夠：

- 判斷兩向量是否正交（orthogonal），並說出正交集合、正交基底與標準正交基底（orthonormal basis）的差異
- 說出正交矩陣（orthogonal matrix）的定義與關鍵性質：$Q^TQ = QQ^T = I$，以及正交矩陣不改變向量長度
- 手算 Gram-Schmidt 正交化流程，把一組線性獨立向量轉換成正交基底，再標準化成標準正交基底
- 說出 QR 分解 $A = QR$ 的意義，並解釋它與 Gram-Schmidt 過程的關係
- 用 Python（`np.linalg.qr`）與 MATLAB（`qr()`）計算 QR 分解，並驗證 $Q^TQ=I$、$QR=A$
- 說出 QR 分解的實務用途（例如之後第 11 章的最小二乘法）

## 概念說明

### 1. 正交向量（Orthogonal Vectors）

兩個向量 $u, v$ 若滿足內積為 0，就稱它們**正交（orthogonal）**：

$$
u \cdot v = u^T v = 0
$$

幾何上，這代表兩向量互相垂直（夾角 90°）。例如：

$$
u = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
\quad \Longrightarrow \quad
u \cdot v = (1)(0) + (0)(1) = 0
$$

所以 $u, v$ 正交。再看一個非正交的例子：$a=(1,2)$、$b=(3,-1)$，$a \cdot b = 3-2=1 \neq 0$，兩者不正交。

> 提醒：零向量與任何向量的內積都是 0，數學上零向量與任何向量「正交」，但在討論正交集合時，通常只考慮非零向量。

### 2. 正交集合、正交基底、標準正交基底

- **正交集合（orthogonal set）**：一組向量 $\{v_1, v_2, \dots, v_k\}$，其中任兩個不同向量都互相正交，即 $v_i \cdot v_j = 0 \; (i \neq j)$。
- **正交基底（orthogonal basis）**：一個正交集合，同時也是某個向量空間的基底（即線性獨立且能生成整個空間）。
- **標準正交基底（orthonormal basis）**：正交基底中每個向量的長度都是 1（單位向量）。也就是：

$$
v_i \cdot v_j =
\begin{cases}
1, & i = j \\
0, & i \neq j
\end{cases}
$$

把一個正交基底變成標準正交基底很簡單：把每個向量除以自己的範數（長度）即可，這個動作稱為**標準化（normalize）**。

正交基底的一大優點是：計算某向量在此基底下的座標時，不需要解聯立方程組，只需要做內積即可（這也是本章 Gram-Schmidt 方法有用的原因之一）。

### 3. 正交矩陣（Orthogonal Matrix）

若方陣 $Q$ 的各行向量（column vectors）構成一組**標準正交基底**，則稱 $Q$ 為**正交矩陣**。正交矩陣有以下等價且重要的性質：

$$
Q^T Q = I \qquad \Longleftrightarrow \qquad Q^{-1} = Q^T
$$

當 $Q$ 是方陣時，$Q^TQ = I$ 也蘊含 $QQ^T = I$（兩邊互為逆矩陣）：

$$
Q^T Q = Q Q^T = I
$$

**正交矩陣保持向量長度（保長度變換）**：對任意向量 $x$，

$$
\|Qx\| = \|x\|
$$

證明：$\|Qx\|^2 = (Qx)^T(Qx) = x^T Q^T Q x = x^T I x = x^T x = \|x\|^2$。

正交矩陣也保持向量間的夾角與內積：$(Qx)\cdot(Qy) = x\cdot y$。幾何上，正交矩陣代表的線性變換是**旋轉**或**鏡射**，不會拉伸或壓縮空間。

例如旋轉矩陣：

$$
Q = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

滿足 $Q^TQ = I$，對任何 $\theta$ 皆成立（可自行代入驗證）。

### 4. Gram-Schmidt 正交化流程

給定一組**線性獨立**的向量 $\{a_1, a_2, \dots, a_n\}$，Gram-Schmidt 流程可以把它們轉換成一組正交向量 $\{u_1, u_2, \dots, u_n\}$，張成的空間（span）與原本完全相同。基本想法是：每次從新向量中，減去它在前面已求出的正交向量方向上的投影分量，留下與前面所有向量都垂直的部分。

**投影公式回顧**：向量 $a$ 在向量 $u$ 方向上的投影是：

$$
\mathrm{proj}_u(a) = \frac{a \cdot u}{u \cdot u}\, u
$$

**Gram-Schmidt 遞迴公式**：

$$
\begin{aligned}
u_1 &= a_1 \\
u_2 &= a_2 - \mathrm{proj}_{u_1}(a_2) \\
u_3 &= a_3 - \mathrm{proj}_{u_1}(a_3) - \mathrm{proj}_{u_2}(a_3) \\
&\;\;\vdots \\
u_k &= a_k - \sum_{i=1}^{k-1} \mathrm{proj}_{u_i}(a_k)
\end{aligned}
$$

得到正交集合 $\{u_1, \dots, u_n\}$ 後，再各自除以範數，就得到標準正交基底：

$$
e_i = \frac{u_i}{\|u_i\|}
$$

#### 手算範例：完整步驟

設三個線性獨立的 3D 向量：

$$
a_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad
a_2 = \begin{bmatrix} 2 \\ 0 \\ 2 \end{bmatrix}, \quad
a_3 = \begin{bmatrix} 3 \\ 3 \\ 3 \end{bmatrix}
$$

**第一步：求 $u_1$**

直接取 $u_1 = a_1$：

$$
u_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

**第二步：求 $u_2$**

先算 $a_2$ 在 $u_1$ 方向上的投影係數：

$$
\frac{a_2 \cdot u_1}{u_1 \cdot u_1} = \frac{(2)(1)+(0)(1)+(2)(0)}{1^2+1^2+0^2} = \frac{2}{2} = 1
$$

所以：

$$
u_2 = a_2 - 1 \cdot u_1 = \begin{bmatrix} 2 \\ 0 \\ 2 \end{bmatrix} - \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}
$$

驗證正交：$u_1 \cdot u_2 = (1)(1)+(1)(-1)+(0)(2) = 0$ ✓

**第三步：求 $u_3$**

先算 $a_3$ 在 $u_1$、$u_2$ 方向上的投影係數：

$$
\frac{a_3 \cdot u_1}{u_1 \cdot u_1} = \frac{(3)(1)+(3)(1)+(3)(0)}{2} = \frac{6}{2} = 3
$$

$$
\frac{a_3 \cdot u_2}{u_2 \cdot u_2} = \frac{(3)(1)+(3)(-1)+(3)(2)}{1^2+(-1)^2+2^2} = \frac{3-3+6}{6} = \frac{6}{6} = 1
$$

所以：

$$
u_3 = a_3 - 3\,u_1 - 1\,u_2
= \begin{bmatrix} 3 \\ 3 \\ 3 \end{bmatrix}
- \begin{bmatrix} 3 \\ 3 \\ 0 \end{bmatrix}
- \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}
= \begin{bmatrix} -1 \\ 1 \\ 1 \end{bmatrix}
$$

驗證正交：$u_3 \cdot u_1 = -1+1+0 = 0$ ✓，$u_3 \cdot u_2 = -1-1+2 = 0$ ✓

**第四步：標準化，求標準正交基底**

計算各向量的範數：

$$
\|u_1\| = \sqrt{1^2+1^2+0^2} = \sqrt{2}, \quad
\|u_2\| = \sqrt{1^2+(-1)^2+2^2} = \sqrt{6}, \quad
\|u_3\| = \sqrt{(-1)^2+1^2+1^2} = \sqrt{3}
$$

$$
e_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \qquad
e_2 = \frac{1}{\sqrt{6}}\begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}, \qquad
e_3 = \frac{1}{\sqrt{3}}\begin{bmatrix} -1 \\ 1 \\ 1 \end{bmatrix}
$$

$\{e_1, e_2, e_3\}$ 就是 $\{a_1, a_2, a_3\}$ 張成空間（此例中為 $\mathbb{R}^3$）的一組**標準正交基底**。

### 5. QR 分解

把 Gram-Schmidt 過程整理成矩陣形式，就得到 **QR 分解**：任何一個「行向量線性獨立」的矩陣 $A$（$m \times n$，$m \ge n$），都可以分解成：

$$
A = QR
$$

其中：

- $Q$ 是 $m \times n$ 矩陣，行向量為標準正交基底 $\{e_1, \dots, e_n\}$（滿足 $Q^TQ=I$；若 $A$ 是方陣則 $Q$ 是正交矩陣）
- $R$ 是 $n \times n$ **上三角矩陣**，且對角線元素為正（在此慣例下唯一）

**$R$ 從哪裡來？** 把 Gram-Schmidt 的投影係數整理起來就是 $R$ 的元素：$R$ 的第 $i$ 列第 $j$ 行元素 $r_{ij}$（$i \le j$），是 $a_j$ 投影在 $e_i$ 方向上的係數 $e_i \cdot a_j$；對角線 $r_{ii} = \|u_i\|$（也就是尚未標準化的正交向量長度）。之所以 $R$ 是「上三角」，是因為 $u_k$ 的建構方式使得 $a_k$ 只會用到 $u_1, \dots, u_k$（也就是 $e_1,\dots,e_k$），不會用到 $e_{k+1},\dots$，所以 $r_{ij}=0 \;(i>j)$。

用上一節的手算範例來說：

$$
Q = \begin{bmatrix} e_1 & e_2 & e_3 \end{bmatrix}, \qquad
R = \begin{bmatrix}
\|u_1\| & \dfrac{a_2\cdot u_1}{u_1\cdot u_1}\|u_1\| & \dfrac{a_3\cdot u_1}{u_1\cdot u_1}\|u_1\| \\[4pt]
0 & \|u_2\| & \dfrac{a_3\cdot u_2}{u_2\cdot u_2}\|u_2\| \\[4pt]
0 & 0 & \|u_3\|
\end{bmatrix}
= \begin{bmatrix}
\sqrt{2} & \sqrt{2} & 3\sqrt{2} \\
0 & \sqrt{6} & \sqrt{6} \\
0 & 0 & \sqrt{3}
\end{bmatrix}
$$

你可以驗證 $QR$ 相乘後確實會還原出 $A = [a_1\ a_2\ a_3]$（下方 Python 程式會實際驗證這件事）。

**QR 分解的用途**：

- **解最小二乘問題**：第 11 章會看到，用 QR 分解求解 $Ax=b$ 的最小二乘近似解，比直接用正規方程式 $A^TAx = A^Tb$ 數值上更穩定。
- **數值穩定的正交化**：實務上電腦計算 QR 分解通常用 Householder 反射或 Givens 旋轉，而非直接套用古典 Gram-Schmidt（因為浮點數誤差會累積，讓 $Q$ 的行向量不夠正交），但數學意義相同。
- **求特徵值的 QR 演算法**、計算矩陣的秩、判斷線性獨立等，都會用到 QR 分解。

## Python 實作

```python
import numpy as np

def gram_schmidt(A):
    """手刻 Gram-Schmidt：對矩陣 A 的行向量做正交化與標準化，回傳 Q, R"""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        u = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            u -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]
    return Q, R

A = np.array([[1, 2, 3],
              [1, 0, 3],
              [0, 2, 3]], dtype=float)

Q, R = gram_schmidt(A)
Q_np, R_np = np.linalg.qr(A)

print("手刻 Gram-Schmidt 得到的 Q:\n", Q)
print("np.linalg.qr 得到的 Q:\n", Q_np)
print("Q @ R 是否還原 A:", np.allclose(Q @ R, A))
print("Q.T @ Q 是否為單位矩陣:", np.allclose(Q.T @ Q, np.eye(3)))
```

完整程式碼（含更多驗證步驟與詳細輸出）請見 [`ch09_orthogonality_qr.py`](ch09_orthogonality_qr.py)，可直接執行：

```bash
python ch09_orthogonality_qr/ch09_orthogonality_qr.py
```

也可以開啟對照的 Notebook 版本 [`ch09_orthogonality_qr.ipynb`](ch09_orthogonality_qr.ipynb)。

## MATLAB 實作

```matlab
A = [1 2 3; 1 0 3; 0 2 3];

[Q, R] = qr(A, 0);   % 'economy size' QR，Q 為 m×n，R 為 n×n（上三角）

disp('驗證 Q''*Q 是否為單位矩陣:')
disp(Q' * Q)

disp('驗證 Q*R 是否還原 A:')
disp(Q * R - A)
```

完整程式碼請見 [`ch09_orthogonality_qr.m`](ch09_orthogonality_qr.m)。

> 注意：本章 `.m` 檔案已用 GNU Octave 10.2 實際執行驗證通過，輸出數值與本章 Python 版本一致；尚未在正式 MATLAB 環境執行，但語法皆為標準 MATLAB 語法，建議你仍自行在 MATLAB 中重新執行一次確認。

## 重點整理

- 兩向量正交 $\iff$ 內積為 0：$u \cdot v = 0$。
- 正交集合：任兩向量互相正交；正交基底：正交集合且為基底；標準正交基底：正交基底中每個向量長度皆為 1。
- 正交矩陣 $Q$ 滿足 $Q^TQ=QQ^T=I$（即 $Q^{-1}=Q^T$），且保持向量長度與內積不變（幾何上代表旋轉或鏡射）。
- Gram-Schmidt 流程：依序讓每個新向量減去它在先前正交向量上的投影，得到正交集合，再各自除以範數即得標準正交基底。
- QR 分解 $A=QR$：$Q$ 的行向量是 Gram-Schmidt 得到的標準正交基底，$R$ 是上三角矩陣，紀錄了投影係數與各步驟的長度。
- QR 分解在數值上比古典 Gram-Schmidt 更穩定，廣泛用於最小二乘法、求秩、特徵值演算法等。

## 練習題

1. 判斷 $u=(2,-1,3)$ 與 $v=(1,5,1)$ 是否正交。

   > 提示：$u \cdot v = 2-5+3=0$，所以正交。

2. 已知 $\{v_1,v_2\}=\{(1,1),(1,-1)\}$ 是 $\mathbb{R}^2$ 的一組正交基底，求對應的標準正交基底。

   > 提示：$\|v_1\|=\|v_2\|=\sqrt2$，所以 $e_1=\frac{1}{\sqrt2}(1,1)$，$e_2=\frac{1}{\sqrt2}(1,-1)$。

3. 驗證下列矩陣是正交矩陣：$Q=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$（提示：檢查 $Q^TQ$ 是否為 $I$）。這個矩陣代表哪一種幾何變換？

   > 提示：$Q^TQ=\begin{bmatrix}1&0\\0&1\end{bmatrix}=I$，是正交矩陣；它把任意向量逆時針旋轉 90°。

4. 用 Gram-Schmidt 對 $a_1=(1,0,0)$、$a_2=(1,1,0)$、$a_3=(1,1,1)$ 做正交化，求 $u_1,u_2,u_3$（不必標準化）。

   > 提示：$u_1=(1,0,0)$；$u_2=a_2-\frac{a_2\cdot u_1}{u_1\cdot u_1}u_1=(0,1,0)$；$u_3=a_3-\frac{a_3\cdot u_1}{u_1\cdot u_1}u_1-\frac{a_3\cdot u_2}{u_2\cdot u_2}u_2=(0,0,1)$（這組向量其實已接近標準正交基底，因此結果就是標準基底本身）。

5. 對本章手算範例的矩陣 $A=[a_1\ a_2\ a_3]=\begin{bmatrix}1&2&3\\1&0&3\\0&2&3\end{bmatrix}$，你已經得到 $Q$（由 $e_1,e_2,e_3$ 組成）與 $R=\begin{bmatrix}\sqrt2&\sqrt2&3\sqrt2\\0&\sqrt6&\sqrt6\\0&0&\sqrt3\end{bmatrix}$。請說明為什麼 $R$ 必須是上三角矩陣（不能有非零的 $r_{21}$ 或 $r_{31}$、$r_{32}$ 之類的下三角元素）。

   > 提示：因為 Gram-Schmidt 建構 $u_k$（進而 $e_k$）時，只用到 $a_1,\dots,a_k$，所以 $e_k$ 落在 $\mathrm{span}\{a_1,\dots,a_k\}$ 中；反過來 $a_j$ 用 $e_1,\dots,e_j$ 展開時，係數 $r_{ij}=e_i\cdot a_j$ 對 $i>j$ 的部分必為 0，因為 $e_i\,(i>j)$ 與 $a_1,\dots,a_j$ 張成的空間正交。
