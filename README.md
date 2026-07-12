# 線性代數教程：使用 MATLAB 與 Python

這是一套從零開始、專為初學者設計的線性代數教程，涵蓋從最基礎的向量概念到 PCA、最小二乘法等實務應用。每一章都同時提供 **Python** 與 **MATLAB** 兩種程式碼實作，讓你可以交叉比對兩種語言的語法與思維方式。

## 這套教程的特色

- **中文講解**：每章皆以繁體中文說明數學概念、公式推導與手算範例。
- **雙語言對照**：同一個概念會分別用 Python（NumPy / SciPy）與 MATLAB 實作。
- **多種學習素材**：每章提供四種檔案，可依你的學習習慣選用。
- **已驗證**：所有 Python 程式碼皆已在本機執行並確認輸出正確；所有 MATLAB 程式碼皆已在 **MATLAB R2025a** 驗證通過，輸出數值與同章 Python 版本一致，同時亦相容於 GNU Octave 10.2。

## 每章檔案說明

以第 1 章為例，`ch01_vectors/` 資料夾內會有：

| 檔案 | 用途 |
|---|---|
| `ch01_vectors.md` | 教學文件：概念說明、公式、手算範例、程式碼與練習題 |
| `ch01_vectors.ipynb` | Jupyter Notebook：可互動執行的 Python 版本，已內含執行結果 |
| `ch01_vectors.py` | 獨立可執行的 Python 腳本版本 |
| `ch01_vectors.m` | 對應的 MATLAB 腳本（已在 MATLAB R2025a 與 GNU Octave 10.2 驗證通過） |

## 章節目錄

| 章節 | 主題 | 核心概念 |
|---|---|---|
| [第 1 章](ch01_vectors/ch01_vectors.md) | 向量基礎 | 向量定義、加法與純量乘法、範數、內積、投影 |
| [第 2 章](ch02_span_independence/ch02_span_independence.md) | 線性組合、生成空間與線性獨立 | 線性組合、span、線性獨立/相依、秩 |
| [第 3 章](ch03_matrices/ch03_matrices.md) | 矩陣與矩陣運算 | 矩陣加法/乘法、轉置、特殊矩陣、跡 |
| [第 4 章](ch04_linear_systems/ch04_linear_systems.md) | 線性方程組與高斯消去法 | Ax=b、增廣矩陣、列梯形式、解的類型 |
| [第 5 章](ch05_determinants/ch05_determinants.md) | 行列式 | 餘因子展開、行列式性質、幾何意義 |
| [第 6 章](ch06_inverse_lu/ch06_inverse_lu.md) | 逆矩陣與 LU 分解 | 逆矩陣求法與性質、LU 分解 |
| [第 7 章](ch07_vector_spaces/ch07_vector_spaces.md) | 向量空間、基底與秩 | 子空間、零空間、行空間、秩-零度定理 |
| [第 8 章](ch08_eigenvalues/ch08_eigenvalues.md) | 特徵值與特徵向量 | 特徵方程式、對角化 |
| [第 9 章](ch09_orthogonality_qr/ch09_orthogonality_qr.md) | 正交性與 QR 分解 | Gram-Schmidt 正交化、QR 分解 |
| [第 10 章](ch10_svd/ch10_svd.md) | 奇異值分解（SVD） | SVD 定義、幾何意義、偽逆 |
| [第 11 章](ch11_least_squares/ch11_least_squares.md) | 最小二乘法與線性迴歸 | 正規方程式、資料擬合 |
| [第 12 章](ch12_pca_applications/ch12_pca_applications.md) | PCA 與 SVD 實務應用 | 主成分分析、降維、影像壓縮 |

## 環境設定

### Python

本專案已包含一個虛擬環境設定範例，你也可以自行建立：

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

執行任一章節的腳本：

```bash
python ch01_vectors/ch01_vectors.py
```

開啟 Notebook：

```bash
jupyter notebook ch01_vectors/ch01_vectors.ipynb
```

### MATLAB

用 MATLAB（建議 R2020b 以上版本）開啟並執行 `.m` 檔案：

```matlab
run('ch01_vectors/ch01_vectors.m')
```

> 全部 12 個 `.m` 檔案皆已在 **MATLAB R2025a** 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2；為了同時相容 MATLAB 與 Octave，程式採用「function 檔案 + local function」寫法（檔案以 `function <檔名>()` 開頭），並避開了 `xline`/`yline`/`sgtitle`/`null(A,'r')` 等 Octave 尚未支援的較新函式。

### GNU Octave（免費、可跨平台取代 MATLAB）

如果你手邊沒有 MATLAB 授權，可以用免費開源的 [GNU Octave](https://octave.org/) 執行本教程的 `.m` 檔案，語法幾乎完全相容。

**安裝：**

```bash
# macOS（Homebrew）
brew install octave

# Ubuntu / Debian
sudo apt install octave

# Windows：至 https://octave.org/download 下載安裝檔
```

**執行章節腳本（在該章節資料夾內，用命令列直接執行檔名即可）：**

```bash
cd ch01_vectors
octave-cli ch01_vectors.m
```

或在 Octave 互動環境中執行：

```matlab
octave:1> cd ch01_vectors
octave:2> ch01_vectors
```

也可以在專案根目錄用 `run()` 執行（效果相同）：

```matlab
octave:1> run('ch01_vectors/ch01_vectors.m')
```

**沒有圖形視窗（例如透過 SSH 遠端連線）時**，加上 `--no-gui` 參數即可，繪圖仍會正常存成 PNG 檔案，不需要顯示器：

```bash
octave-cli --no-gui ch01_vectors.m
```

**已知的 MATLAB／Octave 差異**：本教程的 `.m` 檔案已刻意避開 Octave 尚未支援的較新 MATLAB 函式（如 `xline`/`yline`/`sgtitle`、`null(A,'r')` 選項），並統一採用「function 檔案 + local function」寫法（而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」），因此兩邊都能直接執行，不需要額外修改。

## 建議學習順序

第 1～7 章建立線性代數的基礎語言與運算能力，第 8～10 章進入特徵值與矩陣分解的核心理論，第 11～12 章則展示這些理論如何應用在資料科學（迴歸、降維、壓縮）上。建議依序閱讀，每章結尾的練習題可以幫助你確認是否掌握該章概念。
