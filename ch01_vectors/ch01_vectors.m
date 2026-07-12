% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 1 章：向量基礎（MATLAB 實作）
%
% 本腳本示範：
%   1. 純量與向量的建立（行向量／列向量）
%   2. 向量加法、純量乘法（含幾何圖示）
%   3. 向量的範數（L1、L2、L∞）
%   4. 內積、夾角、正交性判斷
%   5. 單位向量（normalize）
%   6. 向量投影（projection）
%
% 執行方式：於 MATLAB 中執行 run('ch01_vectors.m')

clear; clc; close all;

%% 1. 純量與向量
disp('============================================================')
disp('1. 純量與向量')
disp('============================================================')

s = 3.0;
fprintf('純量 s = %.1f\n', s);

% 行向量（column vector）：MATLAB 中用分號分隔各元素
v_col = [2; 3];
disp('行向量 v (column vector):')
disp(v_col)

% 列向量（row vector）：MATLAB 中用逗號（或空白）分隔各元素
v_row = [2, 3];
disp('列向量 v (row vector):')
disp(v_row)

%% 2. 向量加法與純量乘法
disp('============================================================')
disp('2. 向量加法與純量乘法')
disp('============================================================')

a = [2, 1];
b = [1, 3];

a_plus_b = a + b;
fprintf('向量 a = [%d, %d]\n', a(1), a(2));
fprintf('向量 b = [%d, %d]\n', b(1), b(2));
fprintf('向量 a + b = [%d, %d]\n', a_plus_b(1), a_plus_b(2));

k = 2;
k_times_a = k * a;
fprintf('純量乘法 %d * a = [%d, %d]\n', k, k_times_a(1), k_times_a(2));

%% 3. 向量的長度／範數（L1、L2、L∞）
disp('============================================================')
disp('3. 向量的長度／範數')
disp('============================================================')

v = [3, -4];
fprintf('向量 v = [%d, %d]\n', v(1), v(2));

l1_norm = norm(v, 1);
l2_norm = norm(v, 2);
linf_norm = norm(v, Inf);

fprintf('L1 範數 ||v||_1 = %.4f\n', l1_norm);
fprintf('L2 範數 ||v||_2 = %.4f\n', l2_norm);
fprintf('L∞ 範數 ||v||_inf = %.4f\n', linf_norm);

%% 4. 內積、夾角與正交性
disp('============================================================')
disp('4. 內積、夾角與正交性')
disp('============================================================')

u = [1, 2];
w = [3, -1];

dot_uw = dot(u, w);
fprintf('向量 u = [%d, %d]\n', u(1), u(2));
fprintf('向量 w = [%d, %d]\n', w(1), w(2));
fprintf('內積 u . w = %d\n', dot_uw);

norm_u = norm(u);
norm_w = norm(w);
cos_theta = dot_uw / (norm_u * norm_w);
theta_rad = acos(cos_theta);
theta_deg = rad2deg(theta_rad);

fprintf('|u| = %.6f\n', norm_u);
fprintf('|w| = %.6f\n', norm_w);
fprintf('cos(theta) = %.6f\n', cos_theta);
fprintf('theta (degrees) = %.6f\n', theta_deg);

% 正交性判斷：內積是否為 0
p = [1, 0];
q = [0, 1];
dot_pq = dot(p, q);
fprintf('\n向量 p = [%d, %d]、向量 q = [%d, %d]\n', p(1), p(2), q(1), q(2));
if abs(dot_pq) < 1e-10
    fprintf('p . q = %d -> p 與 q 正交\n', dot_pq);
else
    fprintf('p . q = %d -> p 與 q 不正交\n', dot_pq);
end

%% 5. 單位向量（normalize）
disp('============================================================')
disp('5. 單位向量（normalize）')
disp('============================================================')

x = [3, 4];
x_norm = norm(x);
x_unit = x / x_norm;

fprintf('向量 x = [%d, %d]\n', x(1), x(2));
fprintf('|x| = %.4f\n', x_norm);
fprintf('單位向量 x_hat = x / |x| = [%.4f, %.4f]\n', x_unit(1), x_unit(2));
fprintf('驗證 |x_hat| = %.4f\n', norm(x_unit));

%% 6. 向量投影（projection of a onto b）
disp('============================================================')
disp('6. 向量投影（projection of a onto b）')
disp('============================================================')

a2 = [3, 2];
b2 = [4, 0];

scalar_proj = dot(a2, b2) / norm(b2);
proj_vec = (dot(a2, b2) / dot(b2, b2)) * b2;

fprintf('向量 a = [%d, %d]\n', a2(1), a2(2));
fprintf('向量 b = [%d, %d]\n', b2(1), b2(2));
fprintf('純量投影 (scalar projection) = %.4f\n', scalar_proj);
fprintf('向量投影 proj_b(a) = [%.4f, %.4f]\n', proj_vec(1), proj_vec(2));

%% 7. 幾何圖示：向量加法
disp('============================================================')
disp('7. 繪製向量加法幾何圖')
disp('============================================================')

figure('Position', [100, 100, 500, 500]);
hold on;
quiver(0, 0, a(1), a(2), 0, 'Color', [0 0.447 0.741], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(a(1), a(2), b(1), b(2), 0, 'Color', [0.85 0.325 0.098], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, a_plus_b(1), a_plus_b(2), 0, 'Color', [0.466 0.674 0.188], 'LineWidth', 2, 'MaxHeadSize', 0.5);
xlim([-1, 5]);
ylim([-1, 5]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('向量加法：a + b（平行四邊形法則）');
legend('a = (2, 1)', 'b（從 a 端點平移畫出）', 'a + b = (3, 4)', 'Location', 'northwest');
saveas(gcf, 'vector_addition_matlab.png');
fprintf('已儲存向量加法示意圖至: vector_addition_matlab.png\n');

%% 8. 幾何圖示：向量投影
figure('Position', [100, 100, 500, 500]);
hold on;
quiver(0, 0, b2(1), b2(2), 0, 'Color', [0.85 0.325 0.098], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, a2(1), a2(2), 0, 'Color', [0 0.447 0.741], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, proj_vec(1), proj_vec(2), 0, 'Color', [0.466 0.674 0.188], 'LineWidth', 2, 'MaxHeadSize', 0.5);
plot([a2(1), proj_vec(1)], [a2(2), proj_vec(2)], '--', 'Color', [0.5 0.5 0.5]);
xlim([-1, 5]);
ylim([-1, 4]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('向量投影：a 投影到 b 上');
legend('b = (4, 0)', 'a = (3, 2)', 'proj_b(a)', '垂直於 b 的分量', 'Location', 'northwest');
saveas(gcf, 'vector_projection_matlab.png');
fprintf('已儲存向量投影示意圖至: vector_projection_matlab.png\n');

fprintf('\n全部範例執行完畢。\n');
