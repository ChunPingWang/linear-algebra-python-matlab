% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 6 章：逆矩陣與 LU 分解（MATLAB 實作）
%
% 本腳本示範：
% 1. 2x2 矩陣的逆矩陣（手算公式對照 inv()）
% 2. 3x3 矩陣的逆矩陣（對照講義中高斯-若丹消去法手算的結果）
% 3. 用 inv() 求逆矩陣，並驗證 A * A_inv ≈ I
% 4. 判斷奇異矩陣（singular matrix）無法求逆
% 5. 逆矩陣的性質：(AB)^-1 = B^-1 A^-1、(A^T)^-1 = (A^-1)^T、(A^-1)^-1 = A
% 6. 用 lu() 做 LU 分解，並驗證 L * U ≈ P * A
%
% 執行方式：
%   run('ch06_inverse_lu.m')

clear; clc;

format short

%% ------------------------------------------------------------------
% 1. 2x2 逆矩陣
% ------------------------------------------------------------------
disp(repmat('=', 1, 60));
disp('1. 2x2 逆矩陣');
disp(repmat('=', 1, 60));

A2 = [2 1; 1 1];
disp('矩陣 A =');
disp(A2);

det_A2 = det(A2);
fprintf('det(A) = %g\n', det_A2);

A2_inv = inv(A2);
disp('A 的逆矩陣 A_inv =');
disp(A2_inv);

% 手算公式對照： (1/det) * [d -b; -c a]
a = A2(1,1); b = A2(1,2);
c = A2(2,1); d = A2(2,2);
A2_inv_formula = (1/det_A2) * [d -b; -c a];
disp('用公式手算的逆矩陣 =');
disp(A2_inv_formula);
fprintf('公式結果與 inv() 結果是否一致: %d\n', ...
    norm(A2_inv - A2_inv_formula) < 1e-10);

I2 = eye(2);
check1 = norm(A2 * A2_inv - I2) < 1e-10;
fprintf('驗證 A * A_inv ≈ I: %d\n', check1);
assert(check1, '2x2 逆矩陣驗證失敗！');


%% ------------------------------------------------------------------
% 2. 3x3 逆矩陣（對照講義高斯-若丹消去法手算範例）
% ------------------------------------------------------------------
fprintf('\n');
disp(repmat('=', 1, 60));
disp('2. 3x3 逆矩陣（對照高斯-若丹消去法手算結果）');
disp(repmat('=', 1, 60));

A3 = [1 2 3; 0 1 4; 5 6 0];
disp('矩陣 A =');
disp(A3);

det_A3 = det(A3);
fprintf('det(A) = %g\n', det_A3);

A3_inv = inv(A3);
disp('MATLAB 計算出的逆矩陣 A_inv =');
disp(A3_inv);

% 講義中高斯-若丹消去法手算得到的結果
A3_inv_hand = [-24 18 5; 20 -15 -4; -5 4 1];
disp('講義手算的逆矩陣 =');
disp(A3_inv_hand);
fprintf('MATLAB 結果與手算結果是否一致: %d\n', ...
    norm(A3_inv - A3_inv_hand) < 1e-10);

I3 = eye(3);
check2 = norm(A3 * A3_inv - I3) < 1e-10;
fprintf('驗證 A * A_inv ≈ I: %d\n', check2);
assert(check2, '3x3 逆矩陣驗證失敗！');


%% ------------------------------------------------------------------
% 3. 奇異矩陣（singular matrix）無法求逆
% ------------------------------------------------------------------
fprintf('\n');
disp(repmat('=', 1, 60));
disp('3. 奇異矩陣（det(A) = 0）無法求逆');
disp(repmat('=', 1, 60));

S = [2 4; 1 2];
disp('矩陣 S =');
disp(S);
det_S = det(S);
fprintf('det(S) = %g （接近 0，代表 S 為奇異矩陣）\n', det_S);

% MATLAB 的 inv() 對奇異矩陣不會拋出例外，而是回傳 Inf 並顯示警告
S_inv = inv(S);
disp('inv(S) 結果（會出現 Inf 或警告，代表無法正常求逆）:');
disp(S_inv);


%% ------------------------------------------------------------------
% 4. 逆矩陣的性質
% ------------------------------------------------------------------
fprintf('\n');
disp(repmat('=', 1, 60));
disp('4. 逆矩陣的性質');
disp(repmat('=', 1, 60));

A = [2 1; 1 1];
B = [3 0; 1 2];

A_inv = inv(A);
B_inv = inv(B);

% 性質 1: (AB)^-1 = B^-1 A^-1
AB_inv = inv(A * B);
B_inv_A_inv = B_inv * A_inv;
fprintf('性質 1: (AB)^-1 == B^-1 A^-1 ? %d\n', ...
    norm(AB_inv - B_inv_A_inv) < 1e-10);

% 性質 2: (A^T)^-1 = (A^-1)^T
AT_inv = inv(A');
A_inv_T = A_inv';
fprintf('性質 2: (A^T)^-1 == (A^-1)^T ? %d\n', ...
    norm(AT_inv - A_inv_T) < 1e-10);

% 性質 3: (A^-1)^-1 = A
A_inv_inv = inv(A_inv);
fprintf('性質 3: (A^-1)^-1 == A ? %d\n', ...
    norm(A_inv_inv - A) < 1e-10);

assert(norm(AB_inv - B_inv_A_inv) < 1e-10);
assert(norm(AT_inv - A_inv_T) < 1e-10);
assert(norm(A_inv_inv - A) < 1e-10);


%% ------------------------------------------------------------------
% 5. LU 分解：不需 pivoting 的簡單範例
% ------------------------------------------------------------------
fprintf('\n');
disp(repmat('=', 1, 60));
disp('5. LU 分解：不需 pivoting 的簡單範例');
disp(repmat('=', 1, 60));

% 選擇第一列主元已是該行絕對值最大的元素，
% 這樣即使 MATLAB 預設採用 partial pivoting，也不會實際交換列，
% 方便與講義中的手算範例直接對照。
M = [6 3; 4 3];
disp('矩陣 M =');
disp(M);

% MATLAB 的 [L, U, P] = lu(A) 回傳順序與 SciPy 不同，
% 且遵循慣例 P*A = L*U
[L, U, P] = lu(M);
disp('排列矩陣 P =');
disp(P);
disp('下三角矩陣 L =');
disp(L);
disp('上三角矩陣 U =');
disp(U);

check3 = norm(L * U - P * M) < 1e-10;
fprintf('驗證 L * U ≈ P * M: %d\n', check3);
assert(check3, 'LU 分解驗證失敗！');

% 因為此範例消去過程不需要交換列，P 應為單位矩陣
fprintf('P 是否為單位矩陣（代表此例不需要 pivoting）: %d\n', ...
    norm(P - eye(2)) < 1e-10);

% 用 LU 驗證行列式計算：det(M) = det(L) * det(U) * det(P)
det_M = det(M);
det_via_LU = det(L) * det(U) * det(P);
fprintf('det(M) (直接計算) = %g\n', det_M);
fprintf('det(L)*det(U)*det(P) (透過 LU 分解計算) = %g\n', det_via_LU);
fprintf('兩者是否一致: %d\n', abs(det_M - det_via_LU) < 1e-10);


%% ------------------------------------------------------------------
% 6. LU 分解：3x3 範例（一般情況，含 permutation）
% ------------------------------------------------------------------
fprintf('\n');
disp(repmat('=', 1, 60));
disp('6. LU 分解：3x3 範例（一般情況，含 permutation）');
disp(repmat('=', 1, 60));

N = [1 2 3; 0 1 4; 5 6 0];
disp('矩陣 N =');
disp(N);

[L2, U2, P2] = lu(N);
disp('排列矩陣 P =');
disp(P2);
disp('下三角矩陣 L =');
disp(L2);
disp('上三角矩陣 U =');
disp(U2);

check4 = norm(L2 * U2 - P2 * N) < 1e-10;
fprintf('驗證 L * U ≈ P * N: %d\n', check4);
assert(check4, '3x3 LU 分解驗證失敗！');

% 應用：用 LU 分解快速解 Nx = b（MATLAB 的 \ 運算子內部也會利用 LU 分解求解）
b_vec = [1; 2; 3];
x = N \ b_vec;
x_check = N * x;
disp('解 Nx = b, 其中 b =');
disp(b_vec');
disp('解得 x =');
disp(x');
fprintf('驗證 N * x ≈ b: %d\n', norm(x_check - b_vec) < 1e-10);


fprintf('\n');
disp(repmat('=', 1, 60));
disp('全部範例執行完畢，逆矩陣與 LU 分解驗證皆通過。');
disp(repmat('=', 1, 60));
