% 注意：本檔案已使用 GNU Octave 10.2 實際執行驗證通過（輸出數值與同章 Python 版本一致）。
% 由於作者端目前沒有正版 MATLAB 授權，尚未在正式 MATLAB 環境中執行過；
% 語法均為標準 MATLAB 語法，理論上可直接執行，但仍建議你在 MATLAB 中重新執行一次以完全確認。
%
% 第 10 章：奇異值分解（SVD）（MATLAB 實作）
%
% 本腳本示範：
%   1. 用 svd() 對矩陣做奇異值分解，並驗證 U * Sigma * V' ≈ A
%   2. SVD 與 A'*A、A*A' 特徵分解的關係
%   3. 用 SVD 判斷矩陣的秩（rank）
%   4. Moore-Penrose 偽逆（pseudo-inverse）：用 SVD 手算並與 pinv() 比較
%   5. 幾何意義：2x2 矩陣如何把單位圓映射成橢圓（旋轉 -> 縮放 -> 旋轉）
%
% 執行方式：於 MATLAB 中執行 run('ch10_svd.m')

clear; clc; close all;

format short

%% 1. SVD 的定義：A = U * Sigma * V'
disp('============================================================')
disp('1. SVD 的定義：A = U * Sigma * V''')
disp('============================================================')

% 用一個 3x2（長方形）矩陣示範一般情況
A = [3 0; 4 5; 0 0];
disp('矩陣 A (3x2) =')
disp(A)

% [U, S, V] = svd(A) 預設回傳完整版 SVD：U 為 m x m、S 為 m x n、V 為 n x n
[U, S, V] = svd(A);
disp('U (3x3, 正交矩陣) =')
disp(U)
disp('Sigma (3x2, 對角矩陣) =')
disp(S)
disp('V (2x2, 正交矩陣) =')
disp(V)

% 驗證 A = U * Sigma * V'
A_reconstructed = U * S * V';
disp('重建 U * Sigma * V'' =')
disp(A_reconstructed)
err_recon = norm(A_reconstructed - A, 'fro');
fprintf('重建誤差 (Frobenius norm) = %.10f\n', err_recon);
if err_recon < 1e-10
    disp('[通過] U * Sigma * V'' 與原矩陣 A 一致。')
else
    disp('[失敗] 重建結果與 A 不一致，請檢查！')
end

% 驗證 U、V 皆為正交矩陣：U'*U = I、V'*V = I
[m, n] = size(A);
fprintf('驗證 U 為正交矩陣 (U''*U ≈ I)，誤差 = %.10f\n', norm(U' * U - eye(m), 'fro'));
fprintf('驗證 V 為正交矩陣 (V''*V ≈ I)，誤差 = %.10f\n', norm(V' * V - eye(n), 'fro'));

%% 2. SVD 與特徵值分解的關係
disp('============================================================')
disp('2. SVD 與特徵值分解的關係')
disp('============================================================')

% A'*A 是 n x n 的對稱矩陣，其特徵值就是奇異值的平方，特徵向量就是 V 的欄
AtA = A' * A;
disp('A''*A =')
disp(AtA)

% eig() 對對稱矩陣回傳實數特徵值，預設由小到大排序，這裡改為由大到小
[eigvecs_AtA, eigvals_AtA_diag] = eig(AtA);
eigvals_AtA = diag(eigvals_AtA_diag);
[eigvals_AtA, order] = sort(eigvals_AtA, 'descend');
eigvecs_AtA = eigvecs_AtA(:, order);

s = diag(S);   % S 的對角線元素（奇異值），長度為 min(m, n)
disp('A''*A 的特徵值 (由大到小) =')
disp(eigvals_AtA)
disp('奇異值的平方 s.^2 =')
disp(s .^ 2)
fprintf('驗證：特徵值 ≈ 奇異值平方，誤差 = %.10f\n', ...
    norm(eigvals_AtA(1:length(s)) - s .^ 2));

% A*A' 是 m x m 的對稱矩陣，非零特徵值同樣是奇異值的平方，特徵向量就是 U 的欄
AAt = A * A';
[eigvecs_AAt, eigvals_AAt_diag] = eig(AAt);
eigvals_AAt = diag(eigvals_AAt_diag);
[eigvals_AAt, order2] = sort(eigvals_AAt, 'descend');
eigvecs_AAt = eigvecs_AAt(:, order2);

disp('A*A'' 的特徵值 (由大到小) =')
disp(eigvals_AAt)
disp('（前 length(s) 個應與奇異值平方一致，其餘應接近 0，因為 rank(A) <= min(m, n)）')

%% 3. 手算範例：2x2 矩陣完整算出 U, Sigma, V
disp('============================================================')
disp('3. 手算範例：2x2 矩陣的 SVD')
disp('============================================================')

% 這個矩陣的 B'*B 特徵值恰為簡單數字，適合手算
B = [1 1; 0 1];
disp('矩陣 B (2x2) =')
disp(B)

BtB = B' * B;
disp('B''*B =')
disp(BtB)

[eigvecs_B, eigvals_B_diag] = eig(BtB);
eigvals_B = diag(eigvals_B_diag);
[eigvals_B, order_B] = sort(eigvals_B, 'descend');
eigvecs_B = eigvecs_B(:, order_B);
singular_values_B = sqrt(eigvals_B);

disp('B''*B 的特徵值 (由大到小) =')
disp(eigvals_B)
disp('手算奇異值 = sqrt(特徵值) =')
disp(singular_values_B)

[U_B, S_B, V_B] = svd(B);
s_B = diag(S_B);
disp('MATLAB svd() 計算的奇異值 =')
disp(s_B)
fprintf('驗證手算奇異值與 svd() 結果一致，誤差 = %.10f\n', ...
    norm(singular_values_B - s_B));

%% 4. 用 SVD 計算矩陣的秩
disp('============================================================')
disp('4. 用 SVD 計算矩陣的秩 (rank)')
disp('============================================================')

% 建立一個明顯秩不足的矩陣：第二列是第一列的 2 倍
C = [1 2 3; 2 4 6; 1 0 1];
disp('矩陣 C (3x3) =')
disp(C)

s_C = svd(C);
disp('奇異值 s =')
disp(s_C)

tol = 1e-10;
rank_from_svd = sum(s_C > tol);
rank_builtin = rank(C);

fprintf('非零奇異值個數（容忍誤差 %.0e）= %d\n', tol, rank_from_svd);
fprintf('rank(C) (MATLAB 內建函式) = %d\n', rank_builtin);
if rank_from_svd == rank_builtin
    fprintf('[通過] 用 SVD 求出的秩與內建函式一致，rank(C) = %d\n', rank_builtin);
else
    disp('[失敗] 秩的計算不一致，請檢查！')
end

%% 5. Moore-Penrose 偽逆 (pseudo-inverse)
disp('============================================================')
disp('5. Moore-Penrose 偽逆 (pseudo-inverse)')
disp('============================================================')

disp('矩陣 A (3x2) =')
disp(A)

% 用 SVD 手算偽逆：A^+ = V * Sigma^+ * U'
% Sigma^+ 是把 Sigma 的非零元素取倒數後轉置 (n x m)
Sigma_pinv = zeros(n, m);
for i = 1:length(s)
    if s(i) > tol
        Sigma_pinv(i, i) = 1 / s(i);
    end
end

A_pinv_manual = V * Sigma_pinv * U';
disp('手算偽逆 A^+ = V * Sigma^+ * U'' =')
disp(A_pinv_manual)

A_pinv_builtin = pinv(A);
disp('pinv(A) (MATLAB 內建函式) =')
disp(A_pinv_builtin)

err_pinv = norm(A_pinv_manual - A_pinv_builtin, 'fro');
fprintf('驗證手算偽逆與 pinv() 誤差 = %.10f\n', err_pinv);
if err_pinv < 1e-10
    disp('[通過] 手算（SVD）偽逆與 pinv() 結果一致。')
else
    disp('[失敗] 偽逆計算不一致，請檢查！')
end

% 偽逆的用途：非方陣或不可逆矩陣時求「最小二乘解」 x = A^+ * b（第 11 章會深入介紹）
b = [1; 2; 3];
x_lstsq = A_pinv_builtin * b;
disp('以 b = [1; 2; 3] 為例，最小二乘解 x = A^+ * b =')
disp(x_lstsq)

x_check = A \ b;   % MATLAB 的反斜線運算子對長方形矩陣會自動求最小二乘解
disp('A \ b（MATLAB 反斜線運算子）的解 =')
disp(x_check)
fprintf('兩者誤差 = %.10f\n', norm(x_lstsq - x_check));

% 也驗證偽逆滿足四個 Moore-Penrose 條件之一：A * A^+ * A = A
fprintf('驗證 A * A^+ * A ≈ A，誤差 = %.10f\n', norm(A * A_pinv_builtin * A - A, 'fro'));

%% 6. 幾何意義：2x2 矩陣如何把單位圓映射成橢圓
disp('============================================================')
disp('6. 幾何意義：單位圓 -> 橢圓（旋轉 -> 縮放 -> 旋轉）')
disp('============================================================')

% 選一個容易觀察「旋轉+縮放」效果的 2x2 矩陣
M = [3 1; 1 2];
disp('矩陣 M (2x2) =')
disp(M)

[U_M, S_M, V_M] = svd(M);
s_M = diag(S_M);
disp('U =')
disp(U_M)
disp('奇異值 s =')
disp(s_M)
disp('V =')
disp(V_M)

% 產生單位圓上的點
theta = linspace(0, 2 * pi, 200);
circle = [cos(theta); sin(theta)];   % 2 x 200

% 映射：每個圓上的點 x 經過 M 變成 Mx，畫出來會是橢圓
ellipse = M * circle;

figure('Position', [100, 100, 1000, 500]);

% 左圖：原本的單位圓，並畫出 V 的兩個正交方向（輸入端的主軸）
subplot(1, 2, 1);
hold on;
plot(circle(1, :), circle(2, :), 'Color', [0 0.447 0.741], 'LineWidth', 1.5);
quiver(0, 0, V_M(1,1), V_M(2,1), 0, 'Color', [0.85 0.325 0.098], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, V_M(1,2), V_M(2,2), 0, 'Color', [0.466 0.674 0.188], 'LineWidth', 2, 'MaxHeadSize', 0.5);
xlim([-2, 2]);
ylim([-2, 2]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('映射前：單位圓與 V 的正交方向');
legend('單位圓', 'v_1 方向 (V 第 1 欄)', 'v_2 方向 (V 第 2 欄)', 'Location', 'northwest');

% 右圖：映射後的橢圓，並畫出 U 的方向乘上對應奇異值（輸出端的主軸半長就是奇異值）
subplot(1, 2, 2);
hold on;
plot(ellipse(1, :), ellipse(2, :), 'Color', [0 0.447 0.741], 'LineWidth', 1.5);
u1_scaled = U_M(:, 1) * s_M(1);
u2_scaled = U_M(:, 2) * s_M(2);
quiver(0, 0, u1_scaled(1), u1_scaled(2), 0, 'Color', [0.85 0.325 0.098], 'LineWidth', 2, 'MaxHeadSize', 0.5);
quiver(0, 0, u2_scaled(1), u2_scaled(2), 0, 'Color', [0.466 0.674 0.188], 'LineWidth', 2, 'MaxHeadSize', 0.5);
xlim([-4, 4]);
ylim([-4, 4]);
axis equal;
grid on;
xlabel('x');
ylabel('y');
title('映射後：橢圓與主軸（長度 = 奇異值）');
legend('M 映射後的橢圓', ...
    sprintf('sigma_1 * u_1 (奇異值=%.3f)', s_M(1)), ...
    sprintf('sigma_2 * u_2 (奇異值=%.3f)', s_M(2)), ...
    'Location', 'northwest');

% 用 annotation textbox 取代 sgtitle（sgtitle 在 Octave 中未實作，此法可跨 MATLAB/Octave 相容）
annotation('textbox', [0, 0.94, 1, 0.05], 'String', ...
    'SVD 的幾何意義：旋轉 -> 縮放 -> 旋轉（單位圓變橢圓）', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, 'svd_circle_to_ellipse_matlab.png');
fprintf('已儲存單位圓映射成橢圓示意圖至: svd_circle_to_ellipse_matlab.png\n');

fprintf('\n全部範例執行完畢。\n');
