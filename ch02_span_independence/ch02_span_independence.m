% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 2 章：線性組合、生成空間與線性獨立
% ==========================================
%
% 本腳本示範：
%   1. 線性組合 (linear combination) 的計算
%   2. 生成空間 (span) 的幾何意義（2D 平面示意圖）
%   3. 線性獨立 / 線性相依的判斷
%      - 方法一：手刻函式，解齊次方程組 c1*v1 + c2*v2 + ... = 0，檢查是否只有零解
%      - 方法二：用 rank() 判斷矩陣的秩
%   4. 秩 (rank) 的直觀意義
%   5. 幾何直觀圖示：三個線性相依向量（其中一個是另兩個的線性組合）
%
% 執行方式：在 MATLAB 中開啟本檔案所在資料夾，執行
%   run('ch02_span_independence.m')
%
% 註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch02_span_independence()

clear; clc; close all;

% 圖片輸出資料夾：與本腳本同一層
outDir = fileparts(mfilename('fullpath'));

%% ========================================================================
%% 1. 線性組合 (Linear Combination)
%% ========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('1. 線性組合 (Linear Combination)\n');
fprintf('%s\n', repmat('=', 1, 60));

v1 = [1; 0];
v2 = [0; 1];
c1 = 3;
c2 = -2;

combo = c1 * v1 + c2 * v2;
fprintf('v1 = [%g, %g], v2 = [%g, %g]\n', v1(1), v1(2), v2(1), v2(2));
fprintf('c1 = %g, c2 = %g\n', c1, c2);
fprintf('線性組合 c1*v1 + c2*v2 = [%g, %g]\n', combo(1), combo(2));

% 判斷目標向量 b 是否可以用 v1, v2 的線性組合表示
% 也就是要解 A c = b，其中 A 的欄向量是 v1, v2
b = [5; -1];
A = [v1, v2];
coeffs = A \ b;  % 解線性方程組 A*coeffs = b
residual = A * coeffs - b;
fprintf('\n目標向量 b = [%g, %g]\n', b(1), b(2));
fprintf('解出的係數 (c1, c2) = [%g, %g]\n', coeffs(1), coeffs(2));
fprintf('驗證 A*coeffs - b = [%g, %g] (應接近 0 向量，代表 b 確實在 span{v1, v2} 中)\n', ...
    residual(1), residual(2));

%% ========================================================================
%% 2. 生成空間 (Span) 的幾何意義
%% ========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('2. 生成空間 (Span) 的幾何意義\n');
fprintf('%s\n', repmat('=', 1, 60));

v1 = [2; 1];
v2 = [-1; 1];
fprintf('v1 = [%g, %g], v2 = [%g, %g] 彼此不平行，兩者可以生成 (span) 整個 2D 平面 R^2。\n', ...
    v1(1), v1(2), v2(1), v2(2));
fprintf('這代表：對任何 2D 向量 b，都能找到 c1, c2 使得 c1*v1 + c2*v2 = b。\n');

% 隨機取樣許多係數組合，畫出它們的線性組合，直觀呈現 span 幾乎覆蓋整個平面
rng(42);  % 固定隨機種子，方便重現結果
numSamples = 400;
coeffSamples = (rand(numSamples, 2) * 4) - 2;  % 均勻分布於 [-2, 2]
points = coeffSamples * [v1, v2]';  % 每一列是一個線性組合的結果點

fig1 = figure('Position', [100, 100, 600, 600]);
hold on;
scatter(points(:, 1), points(:, 2), 10, [0.27 0.51 0.71], 'filled', ...
    'MarkerFaceAlpha', 0.4, 'DisplayName', 'c1*v1 + c2*v2 的隨機取樣點');
quiver(0, 0, v1(1), v1(2), 0, 'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
    'DisplayName', 'v1');
quiver(0, 0, v2(1), v2(2), 0, 'Color', 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
    'DisplayName', 'v2');
plot([-4.5, 4.5], [0, 0], 'Color', [0.5 0.5 0.5]);  % x 軸參考線（用 plot 取代 xline，Octave 相容）
plot([0, 0], [-4.5, 4.5], 'Color', [0.5 0.5 0.5]);  % y 軸參考線（用 plot 取代 yline，Octave 相容）
xlim([-4.5, 4.5]);
ylim([-4.5, 4.5]);
axis equal;
title('span\{v1, v2\}：兩個不平行向量生成整個 2D 平面');
legend('Location', 'northeast');
grid on;
hold off;

outPath1 = fullfile(outDir, 'span_demo_matlab.png');
saveas(fig1, outPath1);
fprintf('圖片已存至: %s\n', outPath1);

%% ========================================================================
%% 3. 線性獨立 / 線性相依的判斷
%% ========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('3. 線性獨立 / 線性相依的判斷\n');
fprintf('%s\n', repmat('=', 1, 60));

% 範例 A：兩個不平行的 2D 向量 -> 線性獨立
v1a = [1; 2];
v2a = [3; 1];
Amat = [v1a, v2a];
fprintf('範例 A: v1 = [%g, %g], v2 = [%g, %g]\n', v1a(1), v1a(2), v2a(1), v2a(2));
isIndepA = isIndependentByHomogeneousSystem(Amat);
fprintf('  手刻函式判斷 (解齊次方程組是否只有零解): %s\n', boolToLabel(isIndepA));
fprintf('  rank(A) = %d, 向量個數 = %d -> %s\n', rank(Amat), size(Amat, 2), ...
    boolToLabel(rank(Amat) == size(Amat, 2)));

% 範例 B：三個 2D 向量，其中 v3 = v1 + 2*v2，必定線性相依（2D 空間最多只能有 2 個獨立向量）
v1b = [1; 0];
v2b = [0; 1];
v3b = v1b + 2 * v2b;
Bmat = [v1b, v2b, v3b];
fprintf('\n範例 B: v1 = [%g, %g], v2 = [%g, %g], v3 = v1 + 2*v2 = [%g, %g]\n', ...
    v1b(1), v1b(2), v2b(1), v2b(2), v3b(1), v3b(2));
isIndepB = isIndependentByHomogeneousSystem(Bmat);
fprintf('  手刻函式判斷: %s\n', boolToLabel(isIndepB));
fprintf('  rank(B) = %d, 向量個數 = %d -> %s\n', rank(Bmat), size(Bmat, 2), ...
    boolToLabel(rank(Bmat) == size(Bmat, 2)));
fprintf('  說明：因為 v3 = 1*v1 + 2*v2，所以 1*v1 + 2*v2 + (-1)*v3 = 0 是一組非零解，代表這三個向量線性相依。\n');

% 範例 C：3D 空間中三個線性獨立的向量（標準基底）
Cmat = eye(3);
fprintf('\n範例 C (3D 標準基底):\n');
disp(Cmat);
fprintf('  rank(C) = %d, 向量個數 = %d -> %s\n', rank(Cmat), size(Cmat, 2), ...
    boolToLabel(rank(Cmat) == size(Cmat, 2)));

%% ========================================================================
%% 4. 秩 (Rank) 的直觀意義
%% ========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('4. 秩 (Rank) 的直觀意義（第 7 章會有更完整的討論）\n');
fprintf('%s\n', repmat('=', 1, 60));

M1 = [1 3; 2 1];
M2 = [1 2; 2 4];
M3 = [1 0 1; 0 1 2];

fprintf('滿秩方陣 (2x2 獨立):\n');
disp(M1);
fprintf('  rank = %d  (矩陣形狀 %dx%d)\n\n', rank(M1), size(M1, 1), size(M1, 2));

fprintf('秩為 1 的方陣 (兩列成比例):\n');
disp(M2);
fprintf('  rank = %d  (矩陣形狀 %dx%d)\n\n', rank(M2), size(M2, 1), size(M2, 2));

fprintf('3 個向量在 2D 空間 (最多秩 2):\n');
disp(M3);
fprintf('  rank = %d  (矩陣形狀 %dx%d)\n\n', rank(M3), size(M3, 1), size(M3, 2));

fprintf('直觀意義：秩代表這組向量「實際能撐出的空間維度」。\n');
fprintf('- 秩 = 向量個數 -> 線性獨立，沒有「浪費」的方向\n');
fprintf('- 秩 < 向量個數 -> 線性相依，至少有一個向量是其他向量的線性組合，方向重複了\n');

%% ========================================================================
%% 5. 幾何直觀圖示：三個線性相依向量
%% ========================================================================
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('5. 幾何直觀圖示：2D 平面中三個線性相依向量\n');
fprintf('%s\n', repmat('=', 1, 60));

v1c = [2; 1];
v2c = [-1; 1];
v3c = v1c + v2c;  % v3 是 v1, v2 的線性組合，因此三者線性相依
fprintf('v1 = [%g, %g], v2 = [%g, %g], v3 = v1 + v2 = [%g, %g]\n', ...
    v1c(1), v1c(2), v2c(1), v2c(2), v3c(1), v3c(2));
fprintf('由於 v3 可以表示成 v1 與 v2 的線性組合，這三個向量落在同一個 2D 平面內，彼此線性相依\n');
fprintf('（v1*1 + v2*1 + v3*(-1) = 0 是非零解）。\n');

fig2 = figure('Position', [100, 100, 600, 600]);
hold on;
quiver(0, 0, v1c(1), v1c(2), 0, 'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
    'DisplayName', 'v1 = [2, 1]');
quiver(0, 0, v2c(1), v2c(2), 0, 'Color', 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
    'DisplayName', 'v2 = [-1, 1]');
quiver(0, 0, v3c(1), v3c(2), 0, 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MaxHeadSize', 0.5, ...
    'DisplayName', 'v3 = [1, 2]');

% 用虛線畫出平行四邊形法則，顯示 v3 = v1 + v2
plot([v1c(1), v3c(1)], [v1c(2), v3c(2)], 'k--');
plot([v2c(1), v3c(1)], [v2c(2), v3c(2)], 'k--');

plot([-2, 4], [0, 0], 'Color', [0.5 0.5 0.5]);  % x 軸參考線（用 plot 取代 xline，Octave 相容）
plot([0, 0], [-1, 3], 'Color', [0.5 0.5 0.5]);  % y 軸參考線（用 plot 取代 yline，Octave 相容）
xlim([-2, 4]);
ylim([-1, 3]);
axis equal;
title('線性相依示意圖：v3 = v1 + v2（三者共平面，v3 沒有提供新方向）');
legend('Location', 'northwest');
grid on;
hold off;

outPath2 = fullfile(outDir, 'dependent_vectors_demo_matlab.png');
saveas(fig2, outPath2);
fprintf('圖片已存至: %s\n', outPath2);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('完成\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('本章所有示範已執行完畢，圖片已存至 ch02_span_independence/ 資料夾中。\n');

end % ch02_span_independence（主函式結束）


%% ========================================================================
%% 本地函式 (Local Functions)
%% ========================================================================

function isIndep = isIndependentByHomogeneousSystem(V)
    % 手刻函式：判斷一組向量是否線性獨立。
    %
    % 輸入 V 是一個矩陣，每一欄是一個向量。
    % 做法：把 V 當作齊次方程組 V*c = 0 的係數矩陣，用高斯消去法
    % （部分主元法）化簡為列梯形式，計算主元 (pivot) 的數量。
    % - 若主元數量 = 向量個數 (欄數)，代表齊次方程組只有零解 c = 0
    %   -> 線性獨立
    % - 若主元數量 < 向量個數，代表存在非零解 -> 線性相依
    %
    % 這裡手動實作高斯消去法，用來展示「解齊次方程組」背後的邏輯，
    % 而不是直接呼叫 rank()。

    tol = 1e-10;
    M = V;
    [nRows, nCols] = size(M);

    pivotRow = 1;
    numPivots = 0;

    for col = 1:nCols
        if pivotRow > nRows
            break;
        end
        % 找此欄中，從 pivotRow 開始絕對值最大的列，作為主元（部分主元法）
        [maxVal, relIdx] = max(abs(M(pivotRow:nRows, col)));
        maxRow = pivotRow + relIdx - 1;

        if maxVal < tol
            continue;  % 此欄沒有主元，跳到下一欄
        end

        % 交換列
        temp = M(pivotRow, :);
        M(pivotRow, :) = M(maxRow, :);
        M(maxRow, :) = temp;

        % 消去 pivotRow 以下的列，使該欄下方元素變成 0
        for r = (pivotRow + 1):nRows
            factor = M(r, col) / M(pivotRow, col);
            M(r, :) = M(r, :) - factor * M(pivotRow, :);
        end

        numPivots = numPivots + 1;
        pivotRow = pivotRow + 1;
    end

    isIndep = (numPivots == nCols);
end

function label = boolToLabel(tf)
    % 將布林值轉換為中文標籤，方便輸出訊息
    if tf
        label = '線性獨立';
    else
        label = '線性相依';
    end
end
