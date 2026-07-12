% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 7 章：向量空間、基底與秩
% ======================================
%
% 本腳本示範：
%   1. 向量空間公理的直觀說明（封閉性、加法/純量乘法規則）
%   2. 子空間 (subspace) 的判斷條件
%   3. 零空間 (null space)：Ax = 0 的解集合，並求其基底
%   4. 行空間 (column space)：A 的行向量所張成的空間，並求其基底
%   5. 基底 (basis) 與維度 (dimension)
%   6. 秩-零度定理 (Rank-Nullity Theorem)：rank(A) + nullity(A) = n
%      在兩個不同範例矩陣上驗證
%
% 執行方式：
%   run('ch07_vector_spaces.m')
%
% 註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch07_vector_spaces()

clear; clc;

print_section('1. 向量空間公理的直觀說明');
demo_vector_space_axioms();

print_section('2. 子空間 (Subspace) 的判斷');
demo_subspace();

print_section('3. 手算範例：零空間、行空間、秩，與秩-零度定理');
demo_worked_example();

print_section('4. 第二個範例矩陣（4x3）：再次驗證秩-零度定理');
demo_second_example();

print_section('5. 基底 (Basis) 與維度 (Dimension) 補充範例');
demo_basis_dimension();

print_section('完成');
fprintf('本章所有示範已執行完畢，秩-零度定理已在 3 個不同範例矩陣（A, C, D）上驗證通過。\n');

end % ch07_vector_spaces（主函式結束）


% ========================================================================
% 輔助函式：印出段落標題
% ========================================================================
function print_section(title)
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('%s\n', title);
    fprintf('%s\n', repmat('=', 1, 60));
end


% ========================================================================
% 1. 向量空間公理的直觀說明
% ========================================================================
function demo_vector_space_axioms()
    fprintf('向量空間 (vector space) 是一個集合 V，裡面的向量可以做「加法」與\n');
    fprintf('「純量乘法」，並且運算結果仍然留在 V 裡面（這就是「封閉性」）。\n');
    fprintf('嚴格定義有 8 條公理，但初學者只需要記住兩個核心規則：\n');
    fprintf('  規則 1（加法封閉）：若 u, v 屬於 V，則 u + v 也屬於 V\n');
    fprintf('  規則 2（純量乘法封閉）：若 u 屬於 V，c 是任意純量，則 c*u 也屬於 V\n');
    fprintf('其餘公理（交換律、結合律、有零向量、有反向量、分配律等）\n');
    fprintf('在 R^n（例如 R^2、R^3）這種「一般座標空間」中都自動成立，\n');
    fprintf('所以本章重點會放在 R^n 的「子空間」上，而不是抽象證明。\n');

    % 用具體例子驗證封閉性
    u = [1.0, 2.0];
    v = [3.0, -1.0];
    c = 2.5;
    fprintf('\n範例：u = [%.1f %.1f], v = [%.1f %.1f], 純量 c = %.1f\n', u(1), u(2), v(1), v(2), c);
    uv = u + v;
    cu = c * u;
    fprintf('  u + v = [%.1f %.1f]  (仍是 R^2 中的向量 -> 加法封閉)\n', uv(1), uv(2));
    fprintf('  c * u = [%.1f %.1f]  (仍是 R^2 中的向量 -> 純量乘法封閉)\n', cu(1), cu(2));
end


% ========================================================================
% 2. 子空間 (Subspace) 的判斷
% ========================================================================
function demo_subspace()
    fprintf('子空間是向量空間 V 裡面的一個子集合 W，且 W 自己也必須是一個向量空間。\n');
    fprintf('判斷 W 是否為子空間，只需檢查 3 個條件：\n');
    fprintf('  (1) 零向量在 W 中\n');
    fprintf('  (2) 加法封閉：u, v 屬於 W -> u + v 也屬於 W\n');
    fprintf('  (3) 純量乘法封閉：u 屬於 W, c 為純量 -> c*u 也屬於 W\n');

    fprintf('\n範例 A：W = { (x, y) in R^2 : y = 2x }（通過原點的一條直線）\n');
    fprintf('  這是子空間：零向量 (0,0) 滿足 y=2x；\n');
    fprintf('  任兩點相加、任一點乘上純量後，仍然滿足 y = 2x（自行代入可驗證）。\n');
    p1 = [1.0, 2.0];   % 滿足 y = 2x
    p2 = [-3.0, -6.0]; % 滿足 y = 2x
    s = p1 + p2;
    fprintf('    p1 = [%.1f %.1f] (在 W 中), p2 = [%.1f %.1f] (在 W 中)\n', p1(1), p1(2), p2(1), p2(2));
    fprintf('    p1 + p2 = [%.1f %.1f] -> y/x = %.1f，仍滿足 y = 2x，封閉成立\n', s(1), s(2), s(2)/s(1));

    fprintf('\n範例 B：W'' = { (x, y) in R^2 : y = 2x + 1 }（不過原點的直線）\n');
    fprintf('  這「不是」子空間，因為零向量 (0,0) 不滿足 y = 2x + 1（0 != 1）。\n');
    q1 = [0.0, 1.0];   % 滿足 y = 2x+1
    q2 = [1.0, 3.0];   % 滿足 y = 2x+1
    s2 = q1 + q2;
    fprintf('    q1 = [%.1f %.1f] (在 W'' 中), q2 = [%.1f %.1f] (在 W'' 中)\n', q1(1), q1(2), q2(1), q2(2));
    fprintf('    q1 + q2 = [%.1f %.1f] -> 檢查 y = 2x+1 ? %.1f vs %.1f -> 不相等，封閉性失敗\n', ...
        s2(1), s2(2), s2(2), 2*s2(1) + 1);
    fprintf('    結論：任何不過原點的集合都不可能是子空間。\n');
end


% ========================================================================
% 手刻函式：用 rref 找主元欄位（MATLAB 的 rref 可直接回傳主元欄位索引）
% ========================================================================
function analyze_matrix(A, name)
    % 對矩陣 A 做完整分析：秩、零空間基底、行空間基底，並驗證秩-零度定理。
    [m, n] = size(A);
    fprintf('\n矩陣 %s (%d x %d):\n', name, m, n);
    disp(A);

    % --- 秩 ---
    rank_A = rank(A);
    fprintf('\nrank(%s) = %d  (用 rank() 函式計算)\n', name, rank_A);

    % --- RREF 與主元欄位（手算過程） ---
    [R, pivot_cols] = rref(A);
    fprintf('\n%s 的簡化列梯形式 (RREF):\n', name);
    disp(R);
    fprintf('主元欄位: %s\n', mat2str(pivot_cols));

    % --- 行空間基底：取原矩陣中，主元欄位所在的「原始」欄向量 ---
    col_space_basis = A(:, pivot_cols);
    fprintf('\n行空間 (column space) 的基底：取原矩陣中主元欄位對應的欄向量\n');
    for i = 1:length(pivot_cols)
        c = pivot_cols(i);
        fprintf('  基底向量 %d = %s 的第 %d 欄 = %s\n', i, name, c, mat2str(A(:, c)'));
    end
    fprintf('行空間維度 = 主元個數 = rank(%s) = %d\n', name, rank_A);

    % --- 零空間基底：用手刻的有理數基底函式（等價於 MATLAB 的 null(A,'r')，
    %     但 Octave 尚未實作 'r' 選項，故直接由 RREF 的 R 與主元欄位推導，
    %     兩邊環境皆可執行且結果一致，也與手算過程對應） ---
    ns = rational_null_basis(R, pivot_cols, n);
    nullity = size(ns, 2);
    fprintf('\n零空間 (null space) 的基底 (由 RREF 推導的有理數基底):\n');
    if nullity == 0
        fprintf('  零空間只有零向量 {0}，nullity = 0（%s 的各欄線性獨立）\n', name);
    else
        for i = 1:nullity
            fprintf('  基底向量 %d = %s\n', i, mat2str(ns(:, i)'));
        end
    end
    fprintf('nullity(%s) = 零空間維度 = %d\n', name, nullity);

    % 驗證 A * v ~ 0 對零空間中每個基底向量成立
    if nullity > 0
        residual = A * ns;
        fprintf('驗證 %s * (零空間基底) ~ 0:\n', name);
        disp(residual);
        fprintf('  最大絕對誤差 = %.2e (應接近 0)\n', max(abs(residual(:))));
    end

    % --- 秩-零度定理驗證 ---
    fprintf('\n秩-零度定理驗證：rank(%s) + nullity(%s) = 欄數 n ?\n', name, name);
    fprintf('  %d + %d = %d，欄數 n = %d\n', rank_A, nullity, rank_A + nullity, n);
    if rank_A + nullity == n
        fprintf('  驗證通過！rank(%s) + nullity(%s) = n = %d\n', name, name, n);
    else
        error('秩-零度定理驗證失敗！%s', name);
    end
end


% ========================================================================
% 3. 手算範例矩陣：求零空間、行空間、秩，並驗證秩-零度定理
% ========================================================================
function demo_worked_example()
    fprintf('範例矩陣 A (3x4)：\n');
    fprintf('  A = [1 2 0 3; 0 0 1 2; 1 2 1 5]\n');
    fprintf('\n手算過程（高斯消去法化簡為 RREF）：\n');
    fprintf('  R3 = R3 - R1  ->  [1 2 0 3; 0 0 1 2; 0 0 1 2]\n');
    fprintf('  R3 = R3 - R2  ->  [1 2 0 3; 0 0 1 2; 0 0 0 0]\n');
    fprintf('  已是簡化列梯形式：主元在第 1、3 欄，第 2、4 欄是自由變數欄\n');
    fprintf('  -> rank(A) = 2（兩個主元）\n');
    fprintf('\n零空間手算：設 x2 = s, x4 = t（自由變數），由 RREF 列出方程式：\n');
    fprintf('  x1 + 2*x2 + 3*x4 = 0  ->  x1 = -2s - 3t\n');
    fprintf('  x3 + 2*x4 = 0         ->  x3 = -2t\n');
    fprintf('  解向量 = s*(-2, 1, 0, 0) + t*(-3, 0, -2, 1)\n');
    fprintf('  -> 零空間基底 = { (-2,1,0,0), (-3,0,-2,1) }，nullity(A) = 2\n');
    fprintf('\n行空間手算：主元欄位是第 1、3 欄，取「原矩陣」第 1、3 欄作為行空間基底：\n');
    fprintf('  基底 = { (1,0,1)^T, (0,1,1)^T }（分別是 A 的第 1 欄與第 3 欄）\n');
    fprintf('\n秩-零度定理：rank(A) + nullity(A) = 2 + 2 = 4 = n（A 有 4 欄），驗證成立！\n');

    fprintf('\n以下用程式重新計算，驗證手算結果：\n');
    A = [1 2 0 3;
         0 0 1 2;
         1 2 1 5];
    analyze_matrix(A, 'A');
end


% ========================================================================
% 4. 第二個範例矩陣：再次驗證秩-零度定理
% ========================================================================
function demo_second_example()
    fprintf('範例矩陣 C (4x3)：\n');
    fprintf('  C = [1 2 3; 2 4 6; 1 0 1; 0 1 1]\n');
    fprintf('觀察：第 2 列 = 2 * 第 1 列，且第 3 欄 = 第 1 欄 + 第 2 欄，\n');
    fprintf('預期矩陣的秩會小於欄數與列數。\n');

    C = [1 2 3;
         2 4 6;
         1 0 1;
         0 1 1];
    analyze_matrix(C, 'C');
end


% ========================================================================
% 5. 基底與維度：滿秩方陣範例
% ========================================================================
function demo_basis_dimension()
    fprintf('基底：一組「線性獨立」且能「生成 (span)」整個空間的向量集合。\n');
    fprintf('維度：基底中向量的個數，也就是這個空間「自由度」的數量。\n');

    D = [2 1 1;
         1 3 2;
         1 0 0];
    fprintf('\n範例矩陣 D (3x3)：\n');
    disp(D);
    rank_D = rank(D);
    [R_D, pivot_D] = rref(D);
    ns_D = rational_null_basis(R_D, pivot_D, size(D, 2));
    nullity_D = size(ns_D, 2);
    fprintf('rank(D) = %d，nullity(D) = %d\n', rank_D, nullity_D);
    if rank_D == 3
        fprintf('D 的秩 = 3 = 欄數，代表 D 的三個欄向量線性獨立，\n');
        fprintf('它們本身就是 R^3 的一組基底，行空間 = R^3（維度 3），零空間只有 {0}（維度 0）。\n');
        fprintf('秩-零度定理：%d + %d = %d = 3 (欄數)，驗證成立！\n', rank_D, nullity_D, rank_D + nullity_D);
    end
    assert(rank_D + nullity_D == size(D, 2));
end


% ========================================================================
% 手刻函式：由 RREF 的 R 與主元欄位，推導零空間的「有理數基底」
% （等價於 MATLAB 的 null(A,'r')；Octave 尚未實作該選項，故自行實作，
%   確保在 MATLAB 與 Octave 中皆可執行且結果一致）
% ========================================================================
function ns = rational_null_basis(R, pivot_cols, n)
    free_cols = setdiff(1:n, pivot_cols);
    nullity = length(free_cols);
    ns = zeros(n, nullity);
    for k = 1:nullity
        f = free_cols(k);
        v = zeros(n, 1);
        v(f) = 1;
        for i = 1:length(pivot_cols)
            p = pivot_cols(i);
            v(p) = -R(i, f);
        end
        ns(:, k) = v;
    end
end
