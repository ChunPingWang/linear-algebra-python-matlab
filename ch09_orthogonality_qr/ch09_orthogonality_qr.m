% 注意：本檔案已使用 GNU Octave 10.2 實際執行驗證通過（輸出數值與同章 Python 版本一致）。
% 由於作者端目前沒有正版 MATLAB 授權，尚未在正式 MATLAB 環境中執行過；
% 語法均為標準 MATLAB 語法，理論上可直接執行，但仍建議你在 MATLAB 中重新執行一次以完全確認。
%
% 第 9 章：正交性、Gram-Schmidt 與 QR 分解（MATLAB 實作）
%
% 本腳本示範：
%   1. 正交向量的判斷（內積是否為 0）
%   2. 正交矩陣的性質：Q'*Q = Q*Q' = I，並驗證保長度變換
%   3. 手刻 Gram-Schmidt 正交化流程（含逐步顯示每個正交化向量）
%   4. 用內建 qr() 函式驗證手刻結果一致
%   5. 驗證 Q*R ≈ A 與 Q'*Q ≈ I
%
% 執行方式：於 MATLAB 中執行 run('ch09_orthogonality_qr.m')
%
% 註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch09_orthogonality_qr()

clear; clc; close all;

%% 1. 正交向量的判斷
disp('============================================================')
disp('1. 正交向量的判斷')
disp('============================================================')

u = [1, 0];
v = [0, 1];
dot_uv = dot(u, v);
fprintf('u = [%d, %d], v = [%d, %d]\n', u(1), u(2), v(1), v(2));
if abs(dot_uv) < 1e-10
    fprintf('u . v = %d -> 正交\n', dot_uv);
else
    fprintf('u . v = %d -> 不正交\n', dot_uv);
end

a = [1, 2];
b = [3, -1];
dot_ab = dot(a, b);
fprintf('\na = [%d, %d], b = [%d, %d]\n', a(1), a(2), b(1), b(2));
if abs(dot_ab) < 1e-10
    fprintf('a . b = %d -> 正交\n', dot_ab);
else
    fprintf('a . b = %d -> 不正交\n', dot_ab);
end

%% 2. 正交矩陣的性質
disp('============================================================')
disp('2. 正交矩陣的性質：Q''*Q = Q*Q'' = I，且保持向量長度')
disp('============================================================')

theta = pi / 3;  % 60 度旋轉矩陣
Q_rot = [cos(theta), -sin(theta); sin(theta), cos(theta)];
disp('旋轉矩陣 Q (theta=60°):')
disp(Q_rot)

disp('Q''*Q:')
disp(Q_rot' * Q_rot)
disp('Q*Q'':')
disp(Q_rot * Q_rot')

if norm(Q_rot' * Q_rot - eye(2)) < 1e-10
    disp('Q''*Q ≈ I ? true')
else
    disp('Q''*Q ≈ I ? false')
end

x = [3.0, 4.0];
Qx = (Q_rot * x')';
fprintf('\n原向量 x = [%.4f, %.4f], |x| = %.4f\n', x(1), x(2), norm(x));
fprintf('變換後 Qx = [%.4f, %.4f], |Qx| = %.4f\n', Qx(1), Qx(2), norm(Qx));
if abs(norm(x) - norm(Qx)) < 1e-10
    disp('長度是否保持不變 ? true')
else
    disp('長度是否保持不變 ? false')
end

%% 3. 手刻 Gram-Schmidt 正交化流程
disp('============================================================')
disp('3. 手刻 Gram-Schmidt 正交化流程')
disp('============================================================')

% 使用 .md 教學文件中的手算範例
a1 = [1; 1; 0];
a2 = [2; 0; 2];
a3 = [3; 3; 3];
A = [a1, a2, a3];

disp('原始向量:')
fprintf('a1 = [%d, %d, %d]\n', a1(1), a1(2), a1(3));
fprintf('a2 = [%d, %d, %d]\n', a2(1), a2(2), a2(3));
fprintf('a3 = [%d, %d, %d]\n', a3(1), a3(2), a3(3));
disp('矩陣 A = [a1 a2 a3]:')
disp(A)

[Q_manual, R_manual, U_manual] = gram_schmidt_manual(A, true);

disp('正交化後（尚未標準化）的向量:')
fprintf('u1 = [%.4f, %.4f, %.4f]\n', U_manual(1,1), U_manual(2,1), U_manual(3,1));
fprintf('u2 = [%.4f, %.4f, %.4f]\n', U_manual(1,2), U_manual(2,2), U_manual(3,2));
fprintf('u3 = [%.4f, %.4f, %.4f]\n', U_manual(1,3), U_manual(2,3), U_manual(3,3));

disp('驗證正交性（內積應為 0）:')
fprintf('u1 . u2 = %.10f\n', dot(U_manual(:,1), U_manual(:,2)));
fprintf('u1 . u3 = %.10f\n', dot(U_manual(:,1), U_manual(:,3)));
fprintf('u2 . u3 = %.10f\n', dot(U_manual(:,2), U_manual(:,3)));

disp('標準化後的標準正交基底 Q（各行向量）:')
fprintf('e1 = [%.4f, %.4f, %.4f], ||e1|| = %.4f\n', Q_manual(1,1), Q_manual(2,1), Q_manual(3,1), norm(Q_manual(:,1)));
fprintf('e2 = [%.4f, %.4f, %.4f], ||e2|| = %.4f\n', Q_manual(1,2), Q_manual(2,2), Q_manual(3,2), norm(Q_manual(:,2)));
fprintf('e3 = [%.4f, %.4f, %.4f], ||e3|| = %.4f\n', Q_manual(1,3), Q_manual(2,3), Q_manual(3,3), norm(Q_manual(:,3)));

disp('手刻 Gram-Schmidt 得到的 R (上三角矩陣):')
disp(R_manual)

%% 4. 用內建 qr() 函式驗證手刻結果一致
disp('============================================================')
disp('4. 用內建 qr() 函式驗證手刻 Gram-Schmidt 結果')
disp('============================================================')

[Q_mat, R_mat] = qr(A, 0);  % 'economy size' QR，Q_mat 為 3x3，R_mat 為 3x3 上三角

disp('MATLAB qr() 得到的 Q:')
disp(Q_mat)
disp('MATLAB qr() 得到的 R:')
disp(R_mat)

% qr() 與手刻結果的 Q 每一行可能差一個正負號（正交基底不唯一，
% 但張成的子空間相同），因此逐行比較「絕對值後」是否一致
same_up_to_sign = max(max(abs(abs(Q_manual) - abs(Q_mat)))) < 1e-10;
fprintf('\n手刻 Q 與 qr() 的 Q 是否只差正負號（逐行絕對值比較）: %d\n', same_up_to_sign);

disp('手刻 Q 的行向量兩兩正交 (Q_manual''*Q_manual ≈ I):')
gram_manual = Q_manual' * Q_manual;
disp(gram_manual)
fprintf('  -> %d\n', max(max(abs(gram_manual - eye(3)))) < 1e-10);

disp('qr() 的 Q 的行向量兩兩正交 (Q_mat''*Q_mat ≈ I):')
gram_mat = Q_mat' * Q_mat;
disp(gram_mat)
fprintf('  -> %d\n', max(max(abs(gram_mat - eye(3)))) < 1e-10);

%% 5. 最終驗證：Q*R ≈ A 與 Q'*Q ≈ I
disp('============================================================')
disp('5. 最終驗證：Q*R ≈ A 與 Q''*Q ≈ I')
disp('============================================================')

disp('手刻版本:')
disp('  Q_manual * R_manual =')
disp(Q_manual * R_manual)
check1 = max(max(abs(Q_manual * R_manual - A))) < 1e-10;
fprintf('  是否等於原始 A ? %d\n', check1);
check2 = max(max(abs(Q_manual' * Q_manual - eye(3)))) < 1e-10;
fprintf('  Q_manual''*Q_manual ≈ I ? %d\n', check2);

disp('qr() 版本:')
disp('  Q_mat * R_mat =')
disp(Q_mat * R_mat)
check3 = max(max(abs(Q_mat * R_mat - A))) < 1e-10;
fprintf('  是否等於原始 A ? %d\n', check3);
check4 = max(max(abs(Q_mat' * Q_mat - eye(3)))) < 1e-10;
fprintf('  Q_mat''*Q_mat ≈ I ? %d\n', check4);

all_checks_passed = check1 && check2 && check3 && check4 && same_up_to_sign;

disp('============================================================')
if all_checks_passed
    disp('所有驗證通過：手刻 Gram-Schmidt 與 qr() 結果一致，')
    disp('且 QR = A、Q''Q = I 皆成立。')
else
    disp('警告：部分驗證未通過，請檢查程式碼。')
end
disp('============================================================')

fprintf('\n全部範例執行完畢。\n')

end % ch09_orthogonality_qr（主函式結束）


%% 本機函式：手刻 Gram-Schmidt 正交化
function [Q, R, U] = gram_schmidt_manual(A, verbose)
    % 對矩陣 A（行向量為欲正交化的向量組）做 Gram-Schmidt 正交化與標準化。
    %
    % 輸入:
    %   A       - m x n 矩陣，n 個線性獨立的 m 維行向量
    %   verbose - 是否顯示每一步的中間結果（邏輯值）
    %
    % 輸出:
    %   Q - m x n 矩陣，行向量為標準正交基底
    %   R - n x n 上三角矩陣，滿足 A = Q * R
    %   U - m x n 矩陣，尚未標準化的正交向量 u_1, ..., u_n

    if nargin < 2
        verbose = false;
    end

    [m, n] = size(A);
    U = zeros(m, n);
    Q = zeros(m, n);
    R = zeros(n, n);

    for j = 1:n
        a_j = A(:, j);
        u_j = a_j;
        for i = 1:(j-1)
            coeff = Q(:, i)' * a_j;   % 投影係數 = e_i . a_j
            R(i, j) = coeff;
            u_j = u_j - coeff * Q(:, i);
            if verbose
                fprintf('  a_%d 在 e_%d 方向上的投影係數 = %.4f\n', j, i, coeff);
            end
        end
        norm_u = norm(u_j);
        R(j, j) = norm_u;
        U(:, j) = u_j;
        Q(:, j) = u_j / norm_u;
        if verbose
            fprintf('  u_%d (尚未標準化) = [%.4f, %.4f, %.4f]\n', j, u_j(1), u_j(2), u_j(3));
            fprintf('  ||u_%d|| = %.4f\n', j, norm_u);
            fprintf('  e_%d (標準化後) = [%.4f, %.4f, %.4f]\n', j, Q(1,j), Q(2,j), Q(3,j));
            fprintf('\n');
        end
    end
end
