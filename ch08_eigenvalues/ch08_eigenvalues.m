% 注意：本檔案已使用 GNU Octave 10.2 實際執行驗證通過（輸出數值與同章 Python 版本一致）。
% 由於作者端目前沒有正版 MATLAB 授權，尚未在正式 MATLAB 環境中執行過；
% 語法均為標準 MATLAB 語法，理論上可直接執行，但仍建議你在 MATLAB 中重新執行一次以完全確認。
%
% 第 8 章：特徵值與特徵向量（MATLAB 實作）
%
% 本腳本示範：
%   1. 特徵值/特徵向量的定義 Av = lambda*v，並用 eig() 求解
%   2. 驗證 A*v ≈ lambda*v
%   3. 特徵值的性質：所有特徵值之和 = trace(A)，所有特徵值之積 = det(A)
%   4. 對角化 A = P*D*P^-1，並驗證 P*D*inv(P) ≈ A
%   5. 幾何意義：畫出特徵向量方向在矩陣變換下不改變、只被縮放
%
% 執行方式：於 MATLAB 中執行 run('ch08_eigenvalues.m')

clear; clc; close all;

%% 1. 特徵值與特徵向量的定義：Av = lambda * v
disp('============================================================')
disp('1. 特徵值與特徵向量的定義：Av = lambda * v')
disp('============================================================')

A = [4 1; 2 3];
disp('矩陣 A =')
disp(A)

% [V, D] = eig(A)：V 的每一行 column 是一個特徵向量，D 為特徵值組成的對角矩陣
[V, D] = eig(A);
fprintf('\n用 eig(A) 求得：\n');
disp('特徵值對角矩陣 D =')
disp(D)
disp('特徵向量矩陣 V（每一行 column 對應一個特徵向量）=')
disp(V)

eigenvalues = diag(D);   % 取出對角線上的特徵值，變成向量

%% 2. 驗證 A*v ≈ lambda*v
disp('============================================================')
disp('2. 驗證 A*v ≈ lambda*v')
disp('============================================================')

n = size(A, 1);
for i = 1:n
    lam = eigenvalues(i);
    v = V(:, i);
    lhs = A * v;         % A v
    rhs = lam * v;       % lambda v
    ok = all(abs(lhs - rhs) < 1e-10);
    fprintf('\n第 %d 組: lambda_%d = %.6f\n', i, i, lam);
    fprintf('  v_%d = [%.6f, %.6f]\n', i, v(1), v(2));
    fprintf('  A * v_%d       = [%.6f, %.6f]\n', i, lhs(1), lhs(2));
    fprintf('  lambda_%d * v_%d = [%.6f, %.6f]\n', i, i, rhs(1), rhs(2));
    if ok
        fprintf('  驗證通過：A*v 與 lambda*v 相符\n');
    else
        fprintf('  驗證失敗\n');
    end
end

%% 3. 手算範例對照：2x2 矩陣完整解特徵方程式
disp('============================================================')
disp('3. 手算範例對照：det(A - lambda*I) = 0')
disp('============================================================')

% 手算過程（見 .md 教學文件）得到：
%   特徵方程式: lambda^2 - 7*lambda + 10 = 0 -> (lambda-5)(lambda-2)=0
%   lambda_1 = 5, 對應特徵向量方向 (1, 1)
%   lambda_2 = 2, 對應特徵向量方向 (1, -2)
hand_eigenvalues = [5; 2];
hand_v1 = [1; 1];
hand_v2 = [1; -2];

fprintf('手算特徵值（由大到小）： [%.1f, %.1f]\n', hand_eigenvalues(1), hand_eigenvalues(2));
sorted_eig = sort(eigenvalues, 'descend');
fprintf('MATLAB eig 求得特徵值（由大到小）： [%.1f, %.1f]\n', sorted_eig(1), sorted_eig(2));
fprintf('兩者是否一致： %d\n', all(abs(sorted_eig - hand_eigenvalues) < 1e-10));

fprintf('\n驗證手算特徵向量 v1=(1,1) 對應 lambda=5:\n');
lhs1 = A * hand_v1;
rhs1 = 5 * hand_v1;
fprintf('  A * v1 = [%.4f, %.4f] ; 5 * v1 = [%.4f, %.4f]\n', lhs1(1), lhs1(2), rhs1(1), rhs1(2));
fprintf('  是否相等: %d\n', all(abs(lhs1 - rhs1) < 1e-10));

fprintf('\n驗證手算特徵向量 v2=(1,-2) 對應 lambda=2:\n');
lhs2 = A * hand_v2;
rhs2 = 2 * hand_v2;
fprintf('  A * v2 = [%.4f, %.4f] ; 2 * v2 = [%.4f, %.4f]\n', lhs2(1), lhs2(2), rhs2(1), rhs2(2));
fprintf('  是否相等: %d\n', all(abs(lhs2 - rhs2) < 1e-10));

%% 4. 特徵值的性質：和 = trace(A)，積 = det(A)
disp('============================================================')
disp('4. 特徵值的性質：sum(eigenvalues) = trace(A), prod(eigenvalues) = det(A)')
disp('============================================================')

trace_A = trace(A);
det_A = det(A);
sum_eig = sum(eigenvalues);
prod_eig = prod(eigenvalues);

fprintf('trace(A) = %.6f\n', trace_A);
fprintf('sum(特徵值) = %.6f\n', sum_eig);
fprintf('兩者是否相等: %d\n', abs(trace_A - sum_eig) < 1e-10);

fprintf('\ndet(A) = %.6f\n', det_A);
fprintf('prod(特徵值) = %.6f\n', prod_eig);
fprintf('兩者是否相等: %d\n', abs(det_A - prod_eig) < 1e-10);

%% 5. 對角化 A = P D P^-1
disp('============================================================')
disp('5. 對角化 A = P D P^-1')
disp('============================================================')

P = V;
Dmat = D;
P_inv = inv(P);

A_reconstructed = P * Dmat * P_inv;

disp('P（特徵向量組成的矩陣）=')
disp(P)
disp('D（特徵值對角矩陣）=')
disp(Dmat)
disp('P^-1 =')
disp(P_inv)

fprintf('\n重建 P * D * P^-1 =\n');
disp(A_reconstructed)
fprintf('原矩陣 A =\n');
disp(A)

diff_norm = max(max(abs(A_reconstructed - A)));
is_diagonalizable_check = diff_norm < 1e-10;
fprintf('\nP*D*P^-1 與 A 是否相符: %d\n', is_diagonalizable_check);
if is_diagonalizable_check
    disp('-> 對角化驗證通過')
else
    disp('-> 對角化驗證失敗')
end

% 說明何時矩陣可對角化：n 個線性獨立的特徵向量（等價於 P 為可逆矩陣）
rank_P = rank(P);
fprintf('\nP 的秩 (rank) = %d，矩陣大小 n = %d\n', rank_P, n);
disp('秩 = n，代表特徵向量線性獨立，P 可逆，因此 A 可對角化。')

%% 6. 反例：不可對角化的矩陣（重根但特徵向量不足）
disp('============================================================')
disp('6. 反例：不可對角化的矩陣（重根但特徵向量不足）')
disp('============================================================')

B = [2 1; 0 2];
[V_B, D_B] = eig(B);
rank_V_B = rank(V_B);

disp('矩陣 B =')
disp(B)
fprintf('特徵值 = [%.1f, %.1f] (重根 lambda=2)\n', D_B(1,1), D_B(2,2));
fprintf('特徵向量矩陣 P_B 的秩 = %d（< 2，只有 1 個線性獨立的特徵向量）\n', rank_V_B);
disp('因此 B 不可對角化（defective matrix）。')

%% 7. 幾何意義：畫出矩陣變換下，特徵向量方向不變、只被縮放
disp('============================================================')
disp('7. 繪製幾何意義示意圖')
disp('============================================================')

figure('Position', [100, 100, 1100, 550]);

% 左圖：一般向量在 A 變換下方向改變
subplot(1, 2, 1);
hold on;
general_vectors = {[1;0], [0;1], [1;1]};
colors_gen = {[0 0.447 0.741], [0.85 0.325 0.098], [0.494 0.184 0.556]};
for k = 1:numel(general_vectors)
    v = general_vectors{k};
    Av = A * v;
    c = colors_gen{k};
    quiver(0, 0, v(1), v(2), 0, 'Color', c, 'LineWidth', 1, 'MaxHeadSize', 0.5, 'LineStyle', ':');
    quiver(0, 0, Av(1), Av(2), 0, 'Color', c, 'LineWidth', 2, 'MaxHeadSize', 0.5);
end
xlim([-2, 6]);
ylim([-2, 6]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('一般向量：經 A 變換後方向改變（虛線=原向量，實線=變換後）');

% 右圖：特徵向量在 A 變換下方向不變，只被縮放
subplot(1, 2, 2);
hold on;
v1_unit = hand_v1 / norm(hand_v1);
v2_unit = hand_v2 / norm(hand_v2);
eig_dirs = {v1_unit, v2_unit};
eig_lams = [5, 2];
colors_eig = {[0.466 0.674 0.188], [0.635 0.078 0.184]};
for k = 1:numel(eig_dirs)
    v = eig_dirs{k};
    lam = eig_lams(k);
    Av = A * v;
    c = colors_eig{k};
    quiver(0, 0, v(1), v(2), 0, 'Color', c, 'LineWidth', 1, 'MaxHeadSize', 0.5, 'LineStyle', ':');
    quiver(0, 0, Av(1), Av(2), 0, 'Color', c, 'LineWidth', 2, 'MaxHeadSize', 0.5);
end
xlim([-3, 6]);
ylim([-3, 6]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('特徵向量：經 A 變換後方向不變，只被縮放 lambda 倍');
legend('v (lambda=5方向)', 'A v = 5 v', 'v (lambda=2方向)', 'A v = 2 v', 'Location', 'northwest');

saveas(gcf, 'eigenvectors_geometry_matlab.png');
fprintf('已儲存特徵向量幾何意義示意圖至: eigenvectors_geometry_matlab.png\n');

fprintf('\n全部範例執行完畢。\n');
