% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 4 章：線性方程組與高斯消去法（MATLAB 實作）
%
% 本腳本示範：
%   1. 線性方程組的矩陣表示 Ax = b
%   2. 增廣矩陣 [A|b] 的建立
%   3. 使用左除運算子 A\b 求解
%   4. 使用內建 rref() 函式化為簡化列梯形式
%   5. 手刻高斯消去函式 gaussian_elimination_manual()
%   6. 三種解的情況：唯一解、無限多解、無解
%
% 執行方式：
%   run('ch04_linear_systems.m')
%
% 註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch04_linear_systems()

clear; clc;
format short

%% ------------------------------------------------------------------
%  1. 線性方程組的矩陣表示 Ax = b
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('1. 線性方程組的矩陣表示 Ax = b')
disp(repmat('=', 1, 60))

fprintf('\n考慮下列線性方程組：\n');
fprintf('  2x +  y -  z =   8\n');
fprintf(' -3x -  y + 2z = -11\n');
fprintf(' -2x +  y + 2z =  -3\n\n');

A_demo = [2 1 -1; -3 -1 2; -2 1 2];
b_demo = [8; -11; -3];

disp('A =')
disp(A_demo)
disp('b =')
disp(b_demo)

%% ------------------------------------------------------------------
%  2. 增廣矩陣 [A|b]
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('2. 增廣矩陣 [A|b]')
disp(repmat('=', 1, 60))

Aug_demo = [A_demo, b_demo];
disp('增廣矩陣 [A|b] =')
disp(Aug_demo)

%% ------------------------------------------------------------------
%  3. 高斯消去法的三種基本列運算（說明）
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('3. 高斯消去法的三種基本列運算')
disp(repmat('=', 1, 60))
fprintf('\n(1) 交換兩列 (Row Swap):        R_i <-> R_j\n');
fprintf('(2) 某列乘上非零常數 (Row Scale): R_i -> k * R_i, k ~= 0\n');
fprintf('(3) 某列加上另一列的倍數 (Row Addition): R_i -> R_i + k * R_j\n\n');
fprintf('這三種運算都不會改變方程組的解集合，稱為「基本列運算」。\n');

%% ------------------------------------------------------------------
%  4. 手算範例：逐步高斯消去
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('4. 手算範例：逐步高斯消去')
disp(repmat('=', 1, 60))

M = Aug_demo;
disp('初始增廣矩陣 [A|b]：')
disp(M)

fprintf('\n步驟 1: R2 -> R2 + 1.5*R1，消去 R2 的第一個元素\n');
M(2, :) = M(2, :) + 1.5 * M(1, :);
disp(M)

fprintf('\n步驟 2: R3 -> R3 + 1.0*R1，消去 R3 的第一個元素\n');
M(3, :) = M(3, :) + 1.0 * M(1, :);
disp(M)

fprintf('\n步驟 3: R3 -> R3 - 4*R2，消去 R3 的第二個元素\n');
M(3, :) = M(3, :) - (M(3, 2) / M(2, 2)) * M(2, :);
disp(M)
fprintf('（此時矩陣已達列梯形式 Row Echelon Form, REF）\n');

fprintf('\n步驟 4: 將主元 (pivot) 都化為 1\n');
M(1, :) = M(1, :) / M(1, 1);
M(2, :) = M(2, :) / M(2, 2);
M(3, :) = M(3, :) / M(3, 3);
disp(M)

fprintf('\n步驟 5: 回代消去，將主元上方的元素也化為 0（化為 RREF）\n');
M(1, :) = M(1, :) - M(1, 2) * M(2, :);
fprintf('R1 -> R1 - (R1(2))*R2:\n');
disp(M)

M(2, :) = M(2, :) - M(2, 3) * M(3, :);
M(1, :) = M(1, :) - M(1, 3) * M(3, :);
fprintf('R2 -> R2 - (R2(3))*R3, R1 -> R1 - (R1(3))*R3:\n');
disp(M)

fprintf('\n最終簡化列梯形式 (RREF)：\n');
disp(M)
fprintf('由最後一欄可直接讀出解: x = %g, y = %g, z = %g\n', M(1,4), M(2,4), M(3,4));
fprintf('(正確解應為 x=2, y=3, z=-1，可自行代入原方程組驗證)\n');

%% ------------------------------------------------------------------
%  5. 使用左除運算子 A\b 求解
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('5. 使用左除運算子 A\b 求解')
disp(repmat('=', 1, 60))

x_backslash = A_demo \ b_demo;
disp('x = A \ b =')
disp(x_backslash)

%% ------------------------------------------------------------------
%  6. 使用內建 rref() 函式
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('6. 使用內建 rref() 函式')
disp(repmat('=', 1, 60))

R_builtin = rref(Aug_demo);
disp('rref([A|b]) =')
disp(R_builtin)
fprintf('最後一欄即為解，與 A\\b 結果比較:\n');
disp(R_builtin(:, end))
fprintf('是否一致 (isequal 近似比較): %d\n', all(abs(R_builtin(:, end) - x_backslash) < 1e-10));

%% ------------------------------------------------------------------
%  7. 手刻高斯消去法函式（教學用途）
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('7. 手刻高斯消去法函式 gaussian_elimination_manual()')
disp(repmat('=', 1, 60))

[rref_result, sol_type, x_manual] = gaussian_elimination_manual(A_demo, b_demo, true);

fprintf('\n最終 RREF：\n');
disp(rref_result)
fprintf('解的類型: %s\n', sol_type);
disp('解 x =')
disp(x_manual)

fprintf('與 rref() / A\\b 交叉驗證: %d\n', ...
    all(abs(x_manual - x_backslash) < 1e-10));

%% ------------------------------------------------------------------
%  8. 解的三種情況範例
%  ------------------------------------------------------------------
disp(repmat('=', 1, 60))
disp('8. 解的三種情況')
disp(repmat('=', 1, 60))

% --- 8.1 唯一解 (Unique Solution) ---
fprintf('\n--- 8.1 唯一解 (Unique Solution) ---\n');
fprintf('方程組: x + y = 3, 2x - y = 0\n');
A_unique = [1 1; 2 -1];
b_unique = [3; 0];
[rref_u, type_u, x_u] = gaussian_elimination_manual(A_unique, b_unique, false);
disp('RREF [A|b] =')
disp(rref_u)
fprintf('解的類型: %s\n', type_u);
disp('解 x = (意即 x=1, y=2)')
disp(x_u)
x_u_backslash = A_unique \ b_unique;
fprintf('A\\b 驗證: 一致? %d\n', all(abs(x_u - x_u_backslash) < 1e-10));

% --- 8.2 無限多解 (Infinite Solutions) ---
fprintf('\n--- 8.2 無限多解 (Infinite Solutions) ---\n');
fprintf('方程組: x+y+z=6, 2x+2y+2z=12 (第二式為第一式的 2 倍)\n');
A_inf = [1 1 1; 2 2 2];
b_inf = [6; 12];
[rref_inf, type_inf, x_inf] = gaussian_elimination_manual(A_inf, b_inf, false);
disp('RREF [A|b] =')
disp(rref_inf)
fprintf('解的類型: %s\n', type_inf);
fprintf('說明: 只有 1 個主元但有 3 個未知數，故有 2 個自由變數，解集合為一平面。\n');
fprintf('可用內建 rref() 驗證: \n');
disp(rref([A_inf, b_inf]))

% --- 8.3 無解 (No Solution) ---
fprintf('\n--- 8.3 無解 (No Solution) ---\n');
fprintf('方程組: x+y=2, x+y=5 (與第一式矛盾)\n');
A_none = [1 1; 1 1];
b_none = [2; 5];
[rref_none, type_none, x_none] = gaussian_elimination_manual(A_none, b_none, false);
disp('RREF [A|b] =')
disp(rref_none)
fprintf('解的類型: %s\n', type_none);
fprintf('說明: 化簡後出現 [0 0 | c], c~=0 的矛盾列，無解。\n');

fprintf('\n');
disp(repmat('=', 1, 60))
disp('全部範例執行完畢。')
disp(repmat('=', 1, 60))

end % ch04_linear_systems（主函式結束）


%% ====================================================================
%  區域函式 (local function)：手刻高斯消去法
%  ====================================================================
function [M, solution_type, x] = gaussian_elimination_manual(A, b, verbose)
    % GAUSSIAN_ELIMINATION_MANUAL 以高斯消去法求解 Ax = b（教學用途）
    %
    % 輸入:
    %   A       : (n x n) 或 (m x n) 係數矩陣
    %   b       : (n x 1) 或 (m x 1) 常數向量
    %   verbose : 是否印出每一步的矩陣狀態 (true/false)
    %
    % 輸出:
    %   M              : 化簡後的增廣矩陣 (RREF)
    %   solution_type  : 'unique' | 'infinite' | 'none'
    %   x              : 若為唯一解，回傳解向量 (n x 1)；否則回傳 []

    if nargin < 3
        verbose = false;
    end

    A = double(A);
    b = double(b(:));  % 確保為欄向量
    M = [A, b];
    [n_rows, n_cols] = size(M);
    n_vars = n_cols - 1;

    pivot_row = 1;
    pivot_cols = [];

    % --- 前向消去：化為列梯形式 (REF)，使用部分主元法增加穩定性 ---
    for col = 1:n_vars
        if pivot_row > n_rows
            break
        end

        % 在目前欄位（自 pivot_row 以下）尋找絕對值最大的列作為主元列
        [max_val, rel_idx] = max(abs(M(pivot_row:n_rows, col)));
        max_row = pivot_row + rel_idx - 1;

        if abs(max_val) < 1e-10
            continue  % 此欄沒有可用主元，跳到下一欄（表示自由變數）
        end

        if max_row ~= pivot_row
            temp = M(pivot_row, :);
            M(pivot_row, :) = M(max_row, :);
            M(max_row, :) = temp;
            if verbose
                fprintf('交換 R%d <-> R%d:\n', pivot_row, max_row);
                disp(M)
            end
        end

        % 將主元化為 1
        M(pivot_row, :) = M(pivot_row, :) / M(pivot_row, col);

        % 消去該欄其他所有列的元素
        for r = 1:n_rows
            if r ~= pivot_row && abs(M(r, col)) > 1e-10
                M(r, :) = M(r, :) - M(r, col) * M(pivot_row, :);
            end
        end

        if verbose
            fprintf('以 R%d 為主元列，消去第 %d 欄其餘元素:\n', pivot_row, col);
            disp(M)
        end

        pivot_cols(end + 1) = col; %#ok<AGROW>
        pivot_row = pivot_row + 1;
    end

    rank_A = numel(pivot_cols);

    % --- 判斷解的類型 ---
    % 無解：存在一列形如 [0, 0, ..., 0 | c]，c ~= 0
    inconsistent = false;
    for r = 1:n_rows
        if all(abs(M(r, 1:n_vars)) < 1e-10) && abs(M(r, end)) > 1e-10
            inconsistent = true;
            break
        end
    end

    if inconsistent
        solution_type = 'none';
        x = [];
    elseif rank_A < n_vars
        solution_type = 'infinite';
        x = [];
    else
        solution_type = 'unique';
        x = zeros(n_vars, 1);
        for i = 1:numel(pivot_cols)
            col = pivot_cols(i);
            x(col) = M(i, end);
        end
    end
end
