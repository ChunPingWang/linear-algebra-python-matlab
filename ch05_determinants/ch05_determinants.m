% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 5 章：行列式（MATLAB 實作）
%
% 本腳本示範：
%   1. 2x2 行列式公式（ad - bc）
%   2. 3x3 行列式：手刻餘因子展開（cofactor expansion）並與內建 det() 比對驗證
%   3. 行列式的性質：列交換變號、某列乘常數、列的倍加不影響行列式、轉置行列式不變
%   4. det(A*B) = det(A) * det(B)
%   5. 行列式與可逆性的關係（det(A) = 0 <=> A 不可逆）
%   6. 行列式的幾何意義：2D 面積、3D 體積（含圖示）
%
% 執行方式：
%   run('ch05_determinants.m')
%
% 註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch05_determinants()

clear; clc; close all;

% 圖片輸出資料夾（本章資料夾，與此 .m 檔案同層）
out_dir = fileparts(mfilename('fullpath'));

%% ------------------------------------------------------------------
%% 1. 2x2 行列式：ad - bc
%% ------------------------------------------------------------------
fprintf('%s\n', repmat('=', 1, 60));
fprintf('1. 2x2 行列式：ad - bc\n');
fprintf('%s\n', repmat('=', 1, 60));

A2 = [2 3; 1 4];
disp('矩陣 A2 =');
disp(A2);

a = A2(1,1); b = A2(1,2);
c = A2(2,1); d = A2(2,2);
det_manual_2x2 = a*d - b*c;
det_builtin_2x2 = det(A2);

fprintf('手算：det(A2) = a*d - b*c = %g*%g - %g*%g = %g\n', a, d, b, c, det_manual_2x2);
fprintf('MATLAB det(A2) = %g\n', det_builtin_2x2);
fprintf('兩者是否一致 (abs 差 < 1e-9): %d\n', abs(det_manual_2x2 - det_builtin_2x2) < 1e-9);

%% ------------------------------------------------------------------
%% 2. 3x3 行列式：手刻餘因子展開（cofactor expansion）
%% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('2. 3x3 行列式：手刻餘因子展開（沿第一列展開）\n');
fprintf('%s\n', repmat('=', 1, 60));

A3 = [1 2 3; 0 4 5; 1 0 6];
disp('矩陣 A3 =');
disp(A3);

fprintf('\n沿第 1 列展開：\n');
det_manual_3x3 = cofactor_expansion_3x3(A3, 1);
fprintf('手算 det(A3) = %g\n', det_manual_3x3);

det_builtin_3x3 = det(A3);
fprintf('MATLAB det(A3) = %g\n', det_builtin_3x3);
fprintf('兩者是否一致 (abs 差 < 1e-9): %d\n', abs(det_manual_3x3 - det_builtin_3x3) < 1e-9);

% 驗證：沿第 2 行（column）展開也應得到相同結果
fprintf('\n驗證：沿第 2 行（column j=2）展開（對轉置矩陣沿第 2 列展開，等價於原矩陣沿第 2 行展開）：\n');
det_manual_3x3_col = cofactor_expansion_3x3(A3.', 2);
fprintf('沿第 2 行展開結果 = %g\n', det_manual_3x3_col);
fprintf('與 MATLAB det(A3) 是否一致 (abs 差 < 1e-9): %d\n', abs(det_manual_3x3_col - det_builtin_3x3) < 1e-9);

% 用遞迴版本交叉驗證 4x4 隨機整數矩陣
fprintf('\n以遞迴版餘因子展開交叉驗證 4x4 矩陣：\n');
rng(42);  % 固定亂數種子，方便重現
A4 = randi([-3, 3], 4, 4);
disp('矩陣 A4 =');
disp(A4);
det_manual_4x4 = cofactor_expansion_general(A4);
det_builtin_4x4 = det(A4);
fprintf('遞迴餘因子展開結果 = %g\n', det_manual_4x4);
fprintf('MATLAB det(A4) = %g\n', det_builtin_4x4);
fprintf('兩者是否一致 (abs 差 < 1e-6): %d\n', abs(det_manual_4x4 - det_builtin_4x4) < 1e-6);

%% ------------------------------------------------------------------
%% 3. 行列式的性質：列運算的影響
%% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('3. 行列式的性質：列運算對行列式的影響\n');
fprintf('%s\n', repmat('=', 1, 60));

B = [2 1 0; 1 3 1; 0 1 2];
disp('矩陣 B =');
disp(B);
det_B = det(B);
fprintf('det(B) = %g\n', det_B);

% 性質 1：交換兩列，行列式變號
fprintf('\n性質 1：交換兩列 -> 行列式變號\n');
B_swap = B([2 1 3], :);  % 交換第 1、2 列
disp('交換第 1、2 列後的矩陣 =');
disp(B_swap);
det_B_swap = det(B_swap);
fprintf('det(B_swap) = %g\n', det_B_swap);
fprintf('det(B_swap) 是否等於 -det(B) (abs 差 < 1e-9): %d\n', abs(det_B_swap - (-det_B)) < 1e-9);

% 性質 2：某一列乘上常數 k，行列式也乘上 k
fprintf('\n性質 2：某一列乘上常數 k -> 行列式乘上 k\n');
k = 3;
B_scaled = B;
B_scaled(1, :) = B_scaled(1, :) * k;  % 只將第 1 列乘上 k
fprintf('將第 1 列乘上 k=%g 後的矩陣 =\n', k);
disp(B_scaled);
det_B_scaled = det(B_scaled);
fprintf('det(B_scaled) = %g\n', det_B_scaled);
fprintf('det(B_scaled) 是否等於 %g*det(B) (abs 差 < 1e-9): %d\n', k, abs(det_B_scaled - k*det_B) < 1e-9);

% 性質 3：某列加上另一列的倍數（列的倍加），行列式不變
fprintf('\n性質 3：列的倍加（row replacement）-> 行列式不變\n');
c2 = 5;
B_replace = B;
B_replace(2, :) = B_replace(2, :) + c2 * B_replace(1, :);  % 第 2 列 += c2 * 第 1 列
fprintf('將第 2 列加上 %g 倍第 1 列後的矩陣 =\n', c2);
disp(B_replace);
det_B_replace = det(B_replace);
fprintf('det(B_replace) = %g\n', det_B_replace);
fprintf('det(B_replace) 是否等於 det(B) (abs 差 < 1e-9): %d\n', abs(det_B_replace - det_B) < 1e-9);

% 性質 4：轉置矩陣的行列式不變
fprintf('\n性質 4：det(A^T) = det(A)\n');
det_B_T = det(B.');
fprintf('det(B^T) = %g\n', det_B_T);
fprintf('det(B^T) 是否等於 det(B) (abs 差 < 1e-9): %d\n', abs(det_B_T - det_B) < 1e-9);

% 性質 5：det(AB) = det(A) * det(B)
fprintf('\n性質 5：det(A*B) = det(A) * det(B)\n');
C = [1 0 2; 2 1 0; 0 1 1];
disp('矩陣 C =');
disp(C);
det_C = det(C);
fprintf('det(C) = %g\n', det_C);

BC = B * C;
det_BC = det(BC);
fprintf('det(B*C) = %g\n', det_BC);
fprintf('det(B) * det(C) = %g\n', det_B * det_C);
fprintf('det(BC) 是否等於 det(B)*det(C) (abs 差 < 1e-9): %d\n', abs(det_BC - det_B*det_C) < 1e-9);

%% ------------------------------------------------------------------
%% 4. 行列式與矩陣可逆性
%% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('4. 行列式與矩陣可逆性：det(A)=0 <=> A 不可逆 <=> 各列線性相依\n');
fprintf('%s\n', repmat('=', 1, 60));

% 可逆矩陣範例
A_invertible = [2 1; 1 1];
det_invertible = det(A_invertible);
disp('矩陣 A_invertible =');
disp(A_invertible);
fprintf('det(A_invertible) = %g\n', det_invertible);
fprintf('是否可逆 (det ~= 0): %d\n', abs(det_invertible) > 1e-9);

% 不可逆矩陣範例：第二列是第一列的倍數（線性相依）
A_singular = [2 1; 4 2];  % 第 2 列 = 2 * 第 1 列
det_singular = det(A_singular);
fprintf('\n矩陣 A_singular（第 2 列 = 2 * 第 1 列，線性相依）=\n');
disp(A_singular);
fprintf('det(A_singular) = %g\n', det_singular);
fprintf('是否可逆 (det ~= 0): %d\n', abs(det_singular) > 1e-9);

% 嘗試對奇異矩陣求逆，MATLAB 會發出警告並回傳 Inf/NaN 元素（不會拋出例外）
fprintf('嘗試 inv(A_singular)（預期 MATLAB 會發出 "Matrix is singular" 警告）：\n');
inv_singular = inv(A_singular); %#ok<MINV>
disp(inv_singular);

%% ------------------------------------------------------------------
%% 5. 行列式的幾何意義：2D 面積
%% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('5. 行列式的幾何意義：2D 平行四邊形面積\n');
fprintf('%s\n', repmat('=', 1, 60));

u = [3, 1];
v = [1, 2];
M_area = [u; v];  % 以 u, v 為列向量組成的矩陣

area_val = abs(det(M_area));
fprintf('向量 u = [%g, %g]\n', u(1), u(2));
fprintf('向量 v = [%g, %g]\n', v(1), v(2));
disp('以 u, v 為列的矩陣 M =');
disp(M_area);
fprintf('平行四邊形面積 = |det(M)| = %g\n', area_val);

fig1 = figure('Position', [100, 100, 600, 500]);
hold on;
origin = [0, 0];
parallelogram_x = [origin(1), u(1), u(1)+v(1), v(1), origin(1)];
parallelogram_y = [origin(2), u(2), u(2)+v(2), v(2), origin(2)];
fill(parallelogram_x, parallelogram_y, [0.3 0.5 0.9], 'FaceAlpha', 0.3);
quiver(origin(1), origin(2), u(1), u(2), 0, 'Color', 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(origin(1), origin(2), v(1), v(2), 0, 'Color', [0.9 0.5 0.1], 'LineWidth', 2, 'MaxHeadSize', 0.5);
xlim([-1, 5]);
ylim([-1, 4]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title(sprintf('行列式的幾何意義：面積 = |det(M)| = %.2f', area_val));
legend({'平行四邊形', 'u = (3, 1)', 'v = (1, 2)'}, 'Location', 'northwest');
hold off;

fig1_path = fullfile(out_dir, 'det_area_matlab.png');
saveas(fig1, fig1_path);
fprintf('已儲存 2D 面積示意圖至: %s\n', fig1_path);
close(fig1);

%% ------------------------------------------------------------------
%% 6. 行列式的幾何意義：3D 體積
%% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('6. 行列式的幾何意義：3D 平行六面體體積\n');
fprintf('%s\n', repmat('=', 1, 60));

p = [2, 0, 0];
q = [0, 2, 0];
r = [1, 1, 2];
M_volume = [p; q; r];

volume_val = abs(det(M_volume));
fprintf('向量 p = [%g, %g, %g]\n', p(1), p(2), p(3));
fprintf('向量 q = [%g, %g, %g]\n', q(1), q(2), q(3));
fprintf('向量 r = [%g, %g, %g]\n', r(1), r(2), r(3));
disp('以 p, q, r 為列的矩陣 M =');
disp(M_volume);
fprintf('平行六面體體積 = |det(M)| = %g\n', volume_val);

fig2 = figure('Position', [100, 100, 600, 500]);
hold on;
o = [0, 0, 0];
verts = [o; p; q; r; p+q; p+r; q+r; p+q+r];
% 依序連接平行六面體的 12 條邊
edges = [
    1 2; 1 3; 1 4;
    2 5; 2 6;
    3 5; 3 7;
    4 6; 4 7;
    5 8; 6 8; 7 8
];
for e = 1:size(edges, 1)
    pa = verts(edges(e,1), :);
    pb = verts(edges(e,2), :);
    plot3([pa(1) pb(1)], [pa(2) pb(2)], [pa(3) pb(3)], 'Color', 'b');
end
quiver3(o(1), o(2), o(3), p(1), p(2), p(3), 0, 'Color', 'r', 'LineWidth', 2);
quiver3(o(1), o(2), o(3), q(1), q(2), q(3), 0, 'Color', 'g', 'LineWidth', 2);
quiver3(o(1), o(2), o(3), r(1), r(2), r(3), 0, 'Color', [0.5 0 0.5], 'LineWidth', 2);
xlabel('x');
ylabel('y');
zlabel('z');
title(sprintf('行列式的幾何意義：體積 = |det(M)| = %.2f', volume_val));
grid on;
view(3);
hold off;

fig2_path = fullfile(out_dir, 'det_volume_matlab.png');
saveas(fig2, fig2_path);
fprintf('已儲存 3D 體積示意圖至: %s\n', fig2_path);
close(fig2);

fprintf('\n全部範例執行完畢。\n');

end % ch05_determinants（主函式結束）


%% ==================================================================
%% 本地函式（local functions）
%% ==================================================================

function d = cofactor_expansion_3x3(M, expand_row)
    % 手刻 3x3 行列式的餘因子展開（cofactor expansion）。
    %
    % 公式：沿第 i 列展開時，
    %     det(M) = sum_j M(i, j) * (-1)^(i+j) * det(minor(M, i, j))
    %
    % 本函式僅示範沿「某一列」展開（expand_row，MATLAB 為 1-based 索引），
    % 對於 3x3 矩陣，minor 是 2x2，可直接用 ad - bc 公式計算。
    i = expand_row;
    total = 0;
    for j = 1:3
        sub = minor_matrix(M, i, j);
        sub_det = sub(1,1)*sub(2,2) - sub(1,2)*sub(2,1);
        cofactor_sign = (-1)^(i + j);
        term = M(i, j) * cofactor_sign * sub_det;
        fprintf('  展開項 j=%d: M(%d,%d)=%g, 符號=(-1)^(%d+%d)=%+d, 子行列式 det(minor)=%g, 貢獻=%g\n', ...
            j, i, j, M(i,j), i, j, cofactor_sign, sub_det, term);
        total = total + term;
    end
    d = total;
end

function d = cofactor_expansion_general(M)
    % 遞迴版本的餘因子展開，可處理任意 n x n 矩陣（沿第 1 列展開）。
    n = size(M, 1);
    if n == 1
        d = M(1,1);
        return;
    end
    if n == 2
        d = M(1,1)*M(2,2) - M(1,2)*M(2,1);
        return;
    end
    total = 0;
    for j = 1:n
        sub = minor_matrix(M, 1, j);
        cofactor_sign = (-1)^(1 + j);
        total = total + M(1, j) * cofactor_sign * cofactor_expansion_general(sub);
    end
    d = total;
end

function sub = minor_matrix(M, i, j)
    % 回傳矩陣 M 刪除第 i 列、第 j 行後的子矩陣（minor 的基礎）。
    rows = true(1, size(M, 1));
    cols = true(1, size(M, 2));
    rows(i) = false;
    cols(j) = false;
    sub = M(rows, cols);
end
