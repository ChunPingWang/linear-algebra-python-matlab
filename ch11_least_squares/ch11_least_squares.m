% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 11 章：最小二乘法與線性迴歸（MATLAB 實作）
%
% 本腳本示範：
%   1. 超定方程組 (overdetermined system) 範例
%   2. 正規方程式 (normal equation) 手動計算：A'*A*x = A'*b
%   3. QR 分解求解最小二乘（數值上更穩定）
%   4. 直接用左除運算子 A\b 求解（MATLAB 對非方陣自動使用最小二乘法）
%   5. 簡單線性迴歸範例：模擬資料 y = 2x + 1 + noise（資料為固定數值，
%      與本章 Python 腳本使用 np.random.seed(42) 產生的結果相同，
%      方便跨語言對照，不依賴 MATLAB 與 numpy 不同步的亂數種子）
%   6. 比較三種解法的結果應一致
%   7. 繪製資料點與擬合直線
%
% 執行方式：於 MATLAB 中執行 run('ch11_least_squares.m')

clear; clc; close all;

%% 1. 超定方程組 (overdetermined system) 範例
disp('============================================================')
disp('1. 超定方程組 (overdetermined system) 範例')
disp('============================================================')

% 3 個方程式、2 個未知數，一般而言沒有精確解
A_demo = [1.0 1.0;
          1.0 2.0;
          1.0 3.0];
b_demo = [2.0; 3.0; 5.0];

disp('矩陣 A_demo (3x2, 3 個方程式、2 個未知數) =')
disp(A_demo)
disp('向量 b_demo =')
disp(b_demo)
fprintf('因為方程式數量 (3) 大於未知數數量 (2)，一般而言 Ax=b 無精確解，只能求最小二乘近似解。\n');

x_demo = A_demo \ b_demo;   % 左除運算子對非方陣自動使用最小二乘法
fprintf('最小二乘解 x_demo = [%.6f, %.6f]\n', x_demo(1), x_demo(2));
residual_demo = A_demo * x_demo - b_demo;
fprintf('殘差平方和 ||Ax-b||^2 = %.6f\n', sum(residual_demo.^2));

%% 2. 模擬資料：y = 2x + 1 + noise（固定數值，對應 Python np.random.seed(42) 的輸出）
disp('============================================================')
disp('2. 模擬資料：y = 2x + 1 + noise（固定數值，與 Python 腳本結果相同）')
disp('============================================================')

x_data = [0.0000000000, 0.3448275862, 0.6896551724, 1.0344827586, 1.3793103448, ...
          1.7241379310, 2.0689655172, 2.4137931034, 2.7586206897, 3.1034482759, ...
          3.4482758621, 3.7931034483, 4.1379310345, 4.4827586207, 4.8275862069, ...
          5.1724137931, 5.5172413793, 5.8620689655, 6.2068965517, 6.5517241379, ...
          6.8965517241, 7.2413793103, 7.5862068966, 7.9310344828, 8.2758620690, ...
          8.6206896552, 8.9655172414, 9.3103448276, 9.6551724138, 10.0000000000];

y_data = [1.7450712295, 1.4822587207, 3.3508431520, 5.3535103019, 3.4073906276, ...
          4.0970704266, 7.5067502577, 6.9787383006, 5.8130298004, 8.0207366171, ...
          7.2014251849, 7.8876122662, 9.6388054763, 7.0955968744, 8.0677956650, ...
          10.5013962923, 10.5152360781, 13.1955089299, 12.0517569902, 11.9849927239, ...
          16.9915766017, 15.1440941700, 16.2737061001, 14.7249466862, 16.7351500511, ...
          18.4077631949, 17.2045441166, 20.1842366827, 19.4093867927, 20.5624593753];

n_points = length(x_data);
true_slope = 2.0;
true_intercept = 1.0;

fprintf('資料點數量: %d\n', n_points);
fprintf('真實斜率 (true slope) = %.1f, 真實截距 (true intercept) = %.1f\n', true_slope, true_intercept);
fprintf('前 5 筆 (x, y):\n');
for i = 1:5
    fprintf('  (%.4f, %.4f)\n', x_data(i), y_data(i));
end

% 將問題寫成矩陣形式 A x = b
% A 的欄為 [x, 1]，x = [斜率 m; 截距 c]，b = y
A = [x_data(:), ones(n_points, 1)];
b = y_data(:);

fprintf('\n矩陣形式 A x = b：\n');
fprintf('A 的大小 = [%d, %d]（每列是 [x_i, 1]），b 的大小 = [%d, 1]\n', size(A,1), size(A,2), length(b));
disp('A 的前 5 列 =')
disp(A(1:5, :))

%% 3. 方法一：正規方程式 (normal equation) 手動計算
disp('============================================================')
disp('3. 方法一：正規方程式 (normal equation) 手動計算')
disp('============================================================')

AtA = A' * A;
Atb = A' * b;
disp('A''*A =')
disp(AtA)
disp('A''*b =')
disp(Atb)

x_normal = AtA \ Atb;   % 解正規方程式 A'A x = A'b（AtA 是方陣，可直接用左除）
slope_normal = x_normal(1);
intercept_normal = x_normal(2);
fprintf('正規方程式解得：斜率 m = %.6f, 截距 c = %.6f\n', slope_normal, intercept_normal);

%% 4. 方法二：QR 分解求解最小二乘
disp('============================================================')
disp('4. 方法二：QR 分解求解最小二乘')
disp('============================================================')

[Q, R] = qr(A, 0);   % 精簡 QR 分解 (economy-size)：Q 為 30x2，R 為 2x2 上三角
fprintf('Q 的大小 = [%d, %d], R 的大小 = [%d, %d]\n', size(Q,1), size(Q,2), size(R,1), size(R,2));
disp('R (上三角矩陣) =')
disp(R)

Qtb = Q' * b;
x_qr = R \ Qtb;   % R 是上三角矩陣，左除運算子會直接用回代法求解
slope_qr = x_qr(1);
intercept_qr = x_qr(2);
fprintf('QR 分解解得：斜率 m = %.6f, 截距 c = %.6f\n', slope_qr, intercept_qr);

%% 5. 方法三：左除運算子 A\b（MATLAB 內建最小二乘）
disp('============================================================')
disp('5. 方法三：左除運算子 A\b（MATLAB 內建最小二乘）')
disp('============================================================')

x_backslash = A \ b;
slope_backslash = x_backslash(1);
intercept_backslash = x_backslash(2);
fprintf('A\\b 解得：斜率 m = %.6f, 截距 c = %.6f\n', slope_backslash, intercept_backslash);
fprintf('rank(A) = %d\n', rank(A));

%% 6. 交叉驗證：三種方法結果應一致
disp('============================================================')
disp('6. 交叉驗證：正規方程式 vs QR 分解 vs A\b')
disp('============================================================')

fprintf('正規方程式解 x = [%.6f, %.6f]\n', x_normal(1), x_normal(2));
fprintf('QR 分解解     x = [%.6f, %.6f]\n', x_qr(1), x_qr(2));
fprintf('A\\b 解        x = [%.6f, %.6f]\n', x_backslash(1), x_backslash(2));

tol = 1e-8;
match_normal_qr = all(abs(x_normal - x_qr) < tol);
match_normal_backslash = all(abs(x_normal - x_backslash) < tol);
match_qr_backslash = all(abs(x_qr - x_backslash) < tol);

fprintf('正規方程式 與 QR 分解 是否一致: %d\n', match_normal_qr);
fprintf('正規方程式 與 A\\b     是否一致: %d\n', match_normal_backslash);
fprintf('QR 分解     與 A\\b     是否一致: %d\n', match_qr_backslash);

if match_normal_qr && match_normal_backslash && match_qr_backslash
    disp('驗證通過：三種方法求出的最小二乘解完全一致。')
else
    error('三種方法結果不一致！');
end

fprintf('\n最終擬合直線: y = %.4f * x + %.4f\n', slope_backslash, intercept_backslash);
fprintf('（真實直線為 y = %.1f * x + %.1f，因雜訊影響，估計值會有些微誤差）\n', true_slope, true_intercept);

residual_sum_sq = sum((A * x_backslash - b).^2);
fprintf('殘差平方和 ||Ax-b||^2 = %.6f\n', residual_sum_sq);

%% 7. 繪圖：資料點散佈圖 + 擬合直線
disp('============================================================')
disp('7. 繪圖：資料點與最佳擬合直線')
disp('============================================================')

figure('Position', [100, 100, 700, 550]);
hold on;

scatter(x_data, y_data, 40, [0 0.447 0.741], 'filled', 'DisplayName', '模擬資料點 (帶雜訊)');

x_line = linspace(min(x_data), max(x_data), 100);
y_fit = slope_backslash * x_line + intercept_backslash;
y_true = true_slope * x_line + true_intercept;

plot(x_line, y_fit, 'Color', [0.85 0.325 0.098], 'LineWidth', 2, ...
     'DisplayName', sprintf('最小二乘擬合: y = %.3fx + %.3f', slope_backslash, intercept_backslash));
plot(x_line, y_true, '--', 'Color', [0.466 0.674 0.188], 'LineWidth', 1.5, ...
     'DisplayName', sprintf('真實直線: y = %.1fx + %.1f', true_slope, true_intercept));

xlabel('x');
ylabel('y');
title('最小二乘法線性迴歸：資料點與最佳擬合直線');
legend('Location', 'northwest');
grid on;

exportgraphics(gca, 'regression_fit_matlab.png', 'ContentType', 'image');
fprintf('已儲存資料擬合圖至: regression_fit_matlab.png\n');

fprintf('\n全部範例執行完畢。\n');
