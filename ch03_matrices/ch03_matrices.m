% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 3 章：矩陣與矩陣運算
% ========================
% 對應 ch03_matrices.md 的教學內容，示範：
%   - 矩陣的定義與形狀 (shape)
%   - 矩陣加法、純量乘法
%   - 矩陣乘法，並驗證手算範例
%   - 矩陣乘法不滿足交換律 (AB ~= BA)
%   - 轉置矩陣
%   - 特殊矩陣：零矩陣、單位矩陣、對角矩陣、對稱矩陣、上/下三角矩陣
%   - 跡 (trace) 及其性質
%
% 執行方式：於 MATLAB 指令視窗中執行
%   run('ch03_matrices/ch03_matrices.m')

clear; clc;

%% 1. 矩陣的定義與形狀 (shape)
disp('============================================================')
disp('1. 矩陣的定義與形狀 (shape)')
disp('============================================================')

A_demo = [1 2 3; 4 5 6];   % 2x3 矩陣（2 列、3 行）
disp('A_demo =')
disp(A_demo)
fprintf('A_demo 的形狀 (size): %d x %d  (2 列 x 3 行)\n', size(A_demo, 1), size(A_demo, 2))
fprintf('a_12 (第1列第2行) = %d\n', A_demo(1, 2))
fprintf('a_23 (第2列第3行) = %d\n', A_demo(2, 3))

%% 2. 矩陣加法與純量乘法
disp(' ')
disp('============================================================')
disp('2. 矩陣加法與純量乘法')
disp('============================================================')

M1 = [1 2; 3 4];
M2 = [5 6; 7 8];

M_sum = M1 + M2;
M_scalar = 2 * M1;

disp('M1 ='); disp(M1)
disp('M2 ='); disp(M2)
disp('M1 + M2 ='); disp(M_sum)
disp('預期結果 [6 8; 10 12]，驗證是否相符：')
disp(isequal(M_sum, [6 8; 10 12]))

disp('2 * M1 ='); disp(M_scalar)
disp('預期結果 [2 4; 6 8]，驗證是否相符：')
disp(isequal(M_scalar, [2 4; 6 8]))

%% 3. 矩陣乘法：定義、維度規則、手算範例驗證
disp(' ')
disp('============================================================')
disp('3. 矩陣乘法')
disp('============================================================')

A = [1 2; 3 4];
B = [5 6; 7 8];

disp('A ='); disp(A)
disp('B ='); disp(B)
fprintf('A 的形狀: %d x %d,  B 的形狀: %d x %d\n', size(A,1), size(A,2), size(B,1), size(B,2))

AB = A * B;   % 矩陣乘法用 *
disp(' ')
disp('AB = A * B ='); disp(AB)

expected_AB = [19 22; 43 50];
disp('手算範例結果應為 [19 22; 43 50]')
disp('程式計算結果與手算結果一致：')
disp(isequal(AB, expected_AB))

% 逐步印出手算過程對照 (c_ij = A 第 i 列與 B 第 j 行的內積)
disp(' ')
disp('逐步驗證每個元素:')
for i = 1:2
    for j = 1:2
        row = A(i, :);
        col = B(:, j);
        value = row * col;
        fprintf('  c_%d%d = A第%d列 . B第%d行 = [%d %d] . [%d; %d] = %d\n', ...
            i, j, i, j, row(1), row(2), col(1), col(2), value)
    end
end

%% 3.1 矩陣乘法不滿足交換律
disp(' ')
disp('============================================================')
disp('3.1 矩陣乘法不滿足交換律 (AB ~= BA)')
disp('============================================================')

BA = B * A;
expected_BA = [23 34; 31 46];

disp('BA = B * A ='); disp(BA)
disp('手算範例結果應為 [23 34; 31 46]')
disp('程式計算結果與手算結果一致：')
disp(isequal(BA, expected_BA))

disp('AB ='); disp(AB)
disp('BA ='); disp(BA)
fprintf('AB == BA ? %d  -> 證實矩陣乘法一般不滿足交換律\n', isequal(AB, BA))

%% 4. 轉置矩陣
disp(' ')
disp('============================================================')
disp('4. 轉置矩陣')
disp('============================================================')

A_wide = [1 2 3; 4 5 6];
A_wide_T = A_wide';   % 轉置用 '

disp('A_wide (2x3) ='); disp(A_wide)
disp('A_wide'' (3x2) ='); disp(A_wide_T)
disp('預期轉置 [1 4; 2 5; 3 6]，驗證是否相符：')
disp(isequal(A_wide_T, [1 4; 2 5; 3 6]))

% 驗證 (A^T)^T = A
fprintf('(A_wide'')'' == A_wide ? %d\n', isequal(A_wide_T', A_wide))

% 驗證 (AB)^T = B^T A^T
lhs = (A * B)';
rhs = B' * A';
disp(' ')
disp('驗證 (AB)'' = B''A'':')
disp('(AB)'' ='); disp(lhs)
disp('B''A'' ='); disp(rhs)
fprintf('兩者相等: %d\n', isequal(lhs, rhs))

%% 5. 特殊矩陣
disp(' ')
disp('============================================================')
disp('5. 特殊矩陣')
disp('============================================================')

% 零矩陣
zero_mat = zeros(2, 3);
disp('零矩陣 (2x3):'); disp(zero_mat)

% 單位矩陣
I3 = eye(3);
disp('單位矩陣 I3:'); disp(I3)

% 驗證 A*I = I*A = A（用 3x3 方陣示範）
A3 = [1 2 3; 0 1 4; 5 6 0];
disp('A3 ='); disp(A3)
disp('A3 * I3 ='); disp(A3 * I3)
disp('I3 * A3 ='); disp(I3 * A3)
fprintf('A3*I3 == A3 == I3*A3 ? %d\n', isequal(A3*I3, A3) && isequal(I3*A3, A3))

% 對角矩陣
D = diag([2 5 -1]);
disp(' ')
disp('對角矩陣 D = diag(2, 5, -1):'); disp(D)

% 對稱矩陣
S = [1 2; 2 3];
disp('對稱矩陣 S ='); disp(S)
disp('S'' ='); disp(S')
fprintf('S 是否對稱 (S == S'') ? %d\n', isequal(S, S'))

% 上三角矩陣
U = [1 2 3; 0 5 6; 0 0 9];
disp(' ')
disp('上三角矩陣 U ='); disp(U)
disp('用 triu 從任意矩陣取上三角部分:'); disp(triu(A3))

% 下三角矩陣
L = [1 0 0; 4 5 0; 7 8 9];
disp('下三角矩陣 L ='); disp(L)
disp('用 tril 從任意矩陣取下三角部分:'); disp(tril(A3))

%% 6. 跡 (trace)
disp(' ')
disp('============================================================')
disp('6. 跡 (trace)')
disp('============================================================')

disp('A ='); disp(A)
trace_A = trace(A);
fprintf('trace(A) = %d  (預期 1+4=5)\n', trace_A)

% 跡的線性
C = [2 0; 1 3];
disp(' ')
disp('C ='); disp(C)
fprintf('trace(A) + trace(C) = %d\n', trace(A) + trace(C))
fprintf('trace(A + C) = %d\n', trace(A + C))
fprintf('線性驗證 (trace(A+C) == trace(A)+trace(C)): %d\n', ...
    trace(A + C) == trace(A) + trace(C))

k = 3;
fprintf('\ntrace(%d*A) = %d\n', k, trace(k * A))
fprintf('%d * trace(A) = %d\n', k, k * trace(A))
fprintf('純量倍數驗證: %d\n', trace(k * A) == k * trace(A))

% trace(AB) == trace(BA)
trace_AB = trace(A * B);
trace_BA = trace(B * A);
fprintf('\ntrace(AB) = %d  (預期 19+50=69)\n', trace_AB)
fprintf('trace(BA) = %d  (預期 23+46=69)\n', trace_BA)
fprintf('trace(AB) == trace(BA) ? %d  -> 即使 AB ~= BA，兩者的跡仍相等\n', ...
    trace_AB == trace_BA)

disp(' ')
disp('============================================================')
disp('全部驗證完成')
disp('============================================================')
disp('本章所有數值計算應與教學文件 (ch03_matrices.md) 的手算範例比對一致。')
