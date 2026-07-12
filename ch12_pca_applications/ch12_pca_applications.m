% 注意：本檔案已在 MATLAB R2025a 驗證通過（輸出數值與同章 Python 版本一致），亦相容於 GNU Octave 10.2。
%
% 第 12 章：PCA 與 SVD 實務應用（MATLAB 實作）
%
% 本腳本示範：
%   Part A - 主成分分析 (PCA)
%     1. 建立一組具相關性的 2D 資料（固定數值，避免跨語言亂數種子不同步）
%     2. 資料中心化（減去平均值）
%     3. 計算共變異數矩陣，並用 eig() 對其做特徵分解求主成分方向
%     4. 用 svd() 對中心化資料直接求主成分方向，並與特徵分解結果交叉驗證
%     5. 繪製原始資料點與主成分方向、投影到第一主成分的降維結果
%
%   Part B - SVD 影像/矩陣壓縮
%     1. 用外積組合出一個有明顯圖案的簡單「圖像」矩陣（純公式生成，不依賴外部檔案）
%     2. 用 svd() 對其做奇異值分解
%     3. 用前 k 個奇異值做低秩近似重建，比較不同 k 值的重建誤差與壓縮率
%     4. 繪製原始矩陣與不同 k 值重建結果的視覺化比較
%
% 執行方式：於 MATLAB 中執行 run('ch12_pca_applications.m')
%
% 註：本章刻意不使用 Statistics and Machine Learning Toolbox 的 pca() 函式，
%     而是用 eig() / svd() 手動實作，以維持基礎相容性並凸顯 PCA 的數學原理。
%
% 另註：本檔案採用「function 檔案 + local function」寫法（函式名稱與檔名相同），
% 而非 MATLAB R2016b+ 才支援的「script 檔案內含 local function」寫法，
% 目的是同時相容於 MATLAB 與 GNU Octave。

function ch12_pca_applications()

clear; clc; close all;

%% =======================================================================
%% Part A：主成分分析 (PCA)
%% =======================================================================

%% 1. 建立一組具相關性的 2D 資料
disp('============================================================')
disp('Part A - 1. 建立模擬資料（兩個變數具明顯相關性）')
disp('============================================================')

% 固定寫死的資料點（共 24 筆），x1、x2 兩個變數之間有明顯的正相關，
% 使用固定數值可避免 MATLAB 與 Python 亂數產生器不同步的問題。
X = [
    5.6000, 2.7728;
    7.0471, 3.2368;
    7.6274, 3.5127;
    6.8821, 2.2571;
    3.3609, 0.7150;
    1.7821, 0.6880;
    3.2814, 0.8254;
    4.7303, 2.7144;
    7.1046, 3.4280;
    8.5089, 3.2749;
    5.9077, 2.4288;
    3.3600, 0.5832;
    2.6558, 0.6829;
    2.3770, 0.9734;
    4.8425, 2.5348;
    8.0092, 3.6595;
    7.5516, 3.0885;
    5.9410, 2.4717;
    4.1797, 0.6181;
    1.6943, 0.5246;
    2.5070, 1.2158;
    5.7471, 2.3490;
    7.0794, 3.8068;
    7.6527, 3.0501
];

fprintf('資料矩陣 X 的大小 (n_samples x n_features) = %d x %d\n', size(X, 1), size(X, 2));
disp('X 的前 5 筆樣本：')
disp(X(1:5, :))
fprintf('X 的平均值（中心化前）= [%.4f, %.4f]\n', mean(X, 1));

%% 2. 資料中心化 (centering)
disp('============================================================')
disp('Part A - 2. 資料中心化 (centering)')
disp('============================================================')

X_mean = mean(X, 1);
X_centered = X - X_mean;

fprintf('平均值 mean(X) = [%.4f, %.4f]\n', X_mean);
fprintf('中心化後 X_centered 的平均值（應接近 0）= [%.2e, %.2e]\n', mean(X_centered, 1));

% 註：是否要再除以標準差（做「標準化」）取決於各變數尺度是否相近。
% 本範例兩個變數尺度相近，這裡只做中心化，不做標準差縮放。

%% 3. 共變異數矩陣與特徵分解
disp('============================================================')
disp('Part A - 3. 共變異數矩陣的特徵分解')
disp('============================================================')

n = size(X_centered, 1);
cov_matrix = (X_centered' * X_centered) / (n - 1);
disp('共變異數矩陣 C =')
disp(cov_matrix)

% 用內建 cov() 驗證手算的共變異數矩陣是否正確
cov_matrix_builtin = cov(X_centered);
fprintf('與內建 cov() 計算結果是否一致：%d\n', norm(cov_matrix - cov_matrix_builtin) < 1e-10);

% 對共變異數矩陣做特徵分解：C v = lambda v
% eig() 對「對稱矩陣」回傳的特徵值預設由小到大排序
[eigvecs, eigvals_diag] = eig(cov_matrix);
eigvals = diag(eigvals_diag);

% 依特徵值由大到小重新排序（PCA 習慣把變異量最大的方向排在第一個）
[eigvals_sorted, sort_idx] = sort(eigvals, 'descend');
eigvecs_sorted = eigvecs(:, sort_idx);

fprintf('\n特徵值（由大到小排序）= [%.6f, %.6f]\n', eigvals_sorted);
disp('特徵向量（每一行 column 對應一個主成分方向）=')
disp(eigvecs_sorted)

pc1_eig = eigvecs_sorted(:, 1);
pc2_eig = eigvecs_sorted(:, 2);
fprintf('第一主成分方向 PC1（特徵分解法）= [%.4f, %.4f]\n', pc1_eig);
fprintf('第二主成分方向 PC2（特徵分解法）= [%.4f, %.4f]\n', pc2_eig);

explained_ratio_eig = eigvals_sorted / sum(eigvals_sorted);
fprintf('各主成分解釋的變異量比例 = [%.4f, %.4f]\n', explained_ratio_eig);

%% 4. 用 SVD 重做一次 PCA，並與特徵分解結果交叉驗證
disp('============================================================')
disp('Part A - 4. 用 SVD 重做 PCA，並驗證與特徵分解結果一致')
disp('============================================================')

% 對中心化後的資料矩陣直接做 SVD：X_centered = U * S * V'
[U, S, V] = svd(X_centered, 'econ');
singular_values = diag(S);

% 關鍵關係：共變異數矩陣 C = X_centered' * X_centered / (n-1) = V * diag(S.^2/(n-1)) * V'
% 所以 SVD 的右奇異向量矩陣 V 的每一行 column，就是共變異數矩陣的特徵向量，
% 對應的特徵值 = 奇異值平方 / (n-1)
eigvals_from_svd = (singular_values .^ 2) / (n - 1);

fprintf('SVD 奇異值 S = [%.6f, %.6f]\n', singular_values);
fprintf('由奇異值換算出的特徵值 S^2/(n-1) = [%.6f, %.6f]\n', eigvals_from_svd);
fprintf('與特徵分解求得的特徵值比較       = [%.6f, %.6f]\n', eigvals_sorted);
fprintf('兩者是否一致：%d\n', norm(eigvals_from_svd - eigvals_sorted) < 1e-8);

pc1_svd = V(:, 1);
pc2_svd = V(:, 2);
fprintf('\n第一主成分方向 PC1（SVD 法，未校正正負號）= [%.4f, %.4f]\n', pc1_svd);
fprintf('第一主成分方向 PC1（特徵分解法）           = [%.4f, %.4f]\n', pc1_eig);

% 特徵向量（或奇異向量）的方向在數學上不唯一，可能相差一個正負號
% （v 與 -v 都是同一個特徵值對應的合法特徵向量）。
% 用內積正負號來對齊 v_target 與 v_ref 的方向，方便比較數值是否一致。
if dot(pc1_eig, pc1_svd) < 0
    pc1_svd_aligned = -pc1_svd;
else
    pc1_svd_aligned = pc1_svd;
end
if dot(pc2_eig, pc2_svd) < 0
    pc2_svd_aligned = -pc2_svd;
else
    pc2_svd_aligned = pc2_svd;
end

fprintf('\n校正正負號後的 PC1（SVD 法）= [%.4f, %.4f]\n', pc1_svd_aligned);
fprintf('校正正負號後的 PC2（SVD 法）= [%.4f, %.4f]\n', pc2_svd_aligned);

pc1_match = norm(pc1_eig - pc1_svd_aligned) < 1e-8;
pc2_match = norm(pc2_eig - pc2_svd_aligned) < 1e-8;
fprintf('\nPC1 特徵分解 vs SVD（校正正負號後）是否一致：%d\n', pc1_match);
fprintf('PC2 特徵分解 vs SVD（校正正負號後）是否一致：%d\n', pc2_match);

if pc1_match && pc2_match
    disp('>>> 驗證通過：特徵分解與 SVD 兩種方法求出的主成分方向一致。')
else
    disp('>>> 驗證失敗：兩種方法結果不一致，請檢查計算過程。')
end

%% 5. 投影到第一主成分：資料降維
disp('============================================================')
disp('Part A - 5. 投影到第一主成分（2D -> 1D 降維）')
disp('============================================================')

% 用特徵分解得到的 PC1 方向做投影：z = X_centered * pc1
z_1d = X_centered * pc1_eig;
disp('投影到 PC1 後的 1D 座標 z（前 5 筆）=')
disp(z_1d(1:5))

% 從 1D 座標還原回 2D（近似重建，因為只保留第一主成分，會有資訊損失）
X_reconstructed_1pc = z_1d * pc1_eig' + X_mean;
reconstruction_error = norm(X - X_reconstructed_1pc, 'fro');
fprintf('只用第一主成分重建的 Frobenius 誤差 = %.6f\n', reconstruction_error);
fprintf('原始資料的變異量中，被保留（第一主成分解釋）的比例 = %.4f\n', explained_ratio_eig(1));

%% 6. 繪圖：原始資料點 + 主成分方向 + 投影結果
disp('============================================================')
disp('Part A - 6. 繪製 PCA 圖示')
disp('============================================================')

figure('Position', [100, 100, 1100, 500]);

% 左圖：原始資料點 + 主成分方向
subplot(1, 2, 1);
hold on;
scatter(X(:, 1), X(:, 2), 25, [0 0.447 0.741], 'filled', 'MarkerFaceAlpha', 0.6);

scale = 2.0;
pc1_end = X_mean + scale * sqrt(eigvals_sorted(1)) * pc1_eig';
pc2_end = X_mean + scale * sqrt(eigvals_sorted(2)) * pc2_eig';

quiver(X_mean(1), X_mean(2), pc1_end(1) - X_mean(1), pc1_end(2) - X_mean(2), 0, ...
    'Color', [0.85 0.1 0.1], 'LineWidth', 2.5, 'MaxHeadSize', 0.5);
quiver(X_mean(1), X_mean(2), pc2_end(1) - X_mean(1), pc2_end(2) - X_mean(2), 0, ...
    'Color', [0.1 0.6 0.1], 'LineWidth', 2.5, 'MaxHeadSize', 0.5);
plot(X_mean(1), X_mean(2), 'kx', 'MarkerSize', 12, 'LineWidth', 2.5);

axis equal;
grid on;
xlabel('x1');
ylabel('x2');
title('原始資料點與主成分方向');
legend('原始資料點', 'PC1（第一主成分）', 'PC2（第二主成分）', '資料平均值', 'Location', 'best');

% 右圖：投影到第一主成分後的降維結果（1D 座標，用 0 當作 y 軸方便視覺化）
subplot(1, 2, 2);
hold on;
scatter(z_1d, zeros(size(z_1d)), 25, [0.85 0.1 0.1], 'filled', 'MarkerFaceAlpha', 0.6);
xl_ref = xlim();
plot(xl_ref, [0, 0], 'Color', [0.5 0.5 0.5]);  % x 軸參考線（用 plot 取代 yline，Octave 相容）
grid on;
xlabel('PC1 座標 z');
set(gca, 'YTick', []);
title('投影到第一主成分後的 1D 降維結果');

saveas(gcf, 'pca_projection_matlab.png');
fprintf('已儲存 PCA 圖示至: pca_projection_matlab.png\n');

%% =======================================================================
%% Part B：SVD 影像/矩陣壓縮
%% =======================================================================

%% 1. 用外積組合出一個有明顯圖案的簡單「圖像」矩陣
disp('============================================================')
disp('Part B - 1. 生成有結構的簡單圖像矩陣')
disp('============================================================')

img_size = 40;
coords = linspace(-3, 3, img_size);

% 用幾個「外積」疊加出一個有明顯條紋 + 漸層圖案的矩陣，
% 這種做法本質上就是「低秩結構」：矩陣主要是由少數幾個 rank-1 外積成分組成，
% 因此很適合用來示範 SVD 低秩近似的效果。
pattern_1 = sin(coords)' * cos(coords);                 % 棋盤狀條紋
pattern_2 = exp(-coords.^2 / 4)' * ones(1, img_size);    % 中央亮、邊緣暗的漸層
pattern_3 = ones(img_size, 1) * linspace(-1, 1, img_size); % 水平方向線性漸層
pattern_4 = cos(2 * coords)' * sin(3 * coords);          % 較高頻的細節條紋

% 用固定公式產生「類雜訊」擾動（不用隨機數，確保跨環境結果一致）
[ii, jj] = ndgrid(0:(img_size - 1), 0:(img_size - 1));
pseudo_noise = 0.05 * sin(12.9898 * ii + 78.233 * jj);

image_mat = 3.0 * pattern_1 + 2.0 * pattern_2 + 1.0 * pattern_3 + 0.5 * pattern_4 + pseudo_noise;

fprintf('圖像矩陣形狀 = %d x %d\n', size(image_mat, 1), size(image_mat, 2));
fprintf('圖像矩陣的秩 = %d\n', rank(image_mat));

%% 2. 對圖像矩陣做 SVD
disp('============================================================')
disp('Part B - 2. 對圖像矩陣做 SVD')
disp('============================================================')

[U_img, S_img, V_img] = svd(image_mat, 'econ');
singular_values_img = diag(S_img);

fprintf('U 形狀 = %d x %d; S 對角線長度 = %d; V 形狀 = %d x %d\n', ...
    size(U_img, 1), size(U_img, 2), length(singular_values_img), size(V_img, 1), size(V_img, 2));
disp('前 10 個奇異值 =')
disp(singular_values_img(1:10)')
disp('奇異值快速遞減，代表大部分「能量」集中在前幾個成分中，適合做低秩近似。')

%% 3. 用前 k 個奇異值做低秩近似重建，比較誤差與壓縮率
disp('============================================================')
disp('Part B - 3. 低秩近似重建：比較不同 k 值的誤差與壓縮率')
disp('============================================================')

k_values = [1, 2, 3, 5, 10, 20];
[m, n_cols] = size(image_mat);
full_rank = min(m, n_cols);

errors = zeros(length(k_values), 1);
compression_ratios = zeros(length(k_values), 1);

fprintf('%4s | %16s | %10s | %10s\n', 'k', 'Frobenius誤差', '相對誤差', '壓縮率');
disp(repmat('-', 1, 52))
for idx = 1:length(k_values)
    k = k_values(idx);
    A_k = low_rank_approx(U_img, singular_values_img, V_img, k);
    err = norm(image_mat - A_k, 'fro');
    rel_err = err / norm(image_mat, 'fro');

    % 壓縮率：儲存低秩近似只需存 U_k (m x k)、S_k (k)、V_k (n x k)，
    % 相對於原始 m x n 個元素的比例
    storage_k = k * (m + n_cols + 1);
    storage_full = m * n_cols;
    compression_ratio = storage_k / storage_full;

    errors(idx) = err;
    compression_ratios(idx) = compression_ratio;
    fprintf('%4d | %16.6f | %9.4f%% | %9.4f%%\n', k, err, rel_err * 100, compression_ratio * 100);
end

% 驗證誤差是否隨 k 遞增而單調遞減
is_monotonic_decreasing = all(diff(errors) <= 1e-10);
fprintf('\n');
if is_monotonic_decreasing
    disp('>>> 驗證通過：重建誤差（Frobenius norm）隨 k 增加而單調遞減（或持平）。')
else
    disp('>>> 驗證失敗：重建誤差沒有隨 k 增加而遞減，請檢查計算過程。')
end

% 額外驗證：k = full_rank 時應完全重建（誤差趨近於 0）
A_full = low_rank_approx(U_img, singular_values_img, V_img, full_rank);
full_rank_error = norm(image_mat - A_full, 'fro');
fprintf('k = full_rank (%d) 時的重建誤差 = %.2e（應接近 0）\n', full_rank, full_rank_error);

%% 4. 繪圖：原始矩陣 vs 不同 k 值的重建結果
disp('============================================================')
disp('Part B - 4. 繪製原始矩陣與低秩近似重建結果')
disp('============================================================')

k_to_plot = [1, 3, 5, 20];
vmin = min(image_mat(:));
vmax = max(image_mat(:));

figure('Position', [100, 100, 1400, 320]);
subplot(1, length(k_to_plot) + 1, 1);
imagesc(image_mat, [vmin, vmax]);
axis image off;
colormap(gca, 'gray');
title('原始矩陣（滿秩）');

for idx = 1:length(k_to_plot)
    k = k_to_plot(idx);
    A_k = low_rank_approx(U_img, singular_values_img, V_img, k);
    err = norm(image_mat - A_k, 'fro');

    subplot(1, length(k_to_plot) + 1, idx + 1);
    imagesc(A_k, [vmin, vmax]);
    axis image off;
    colormap(gca, 'gray');
    title(sprintf('k = %d, 誤差 = %.3f', k, err));
end

saveas(gcf, 'svd_compression_matlab.png');
fprintf('已儲存 SVD 低秩近似比較圖至: svd_compression_matlab.png\n');

% 額外繪製「誤差 vs k」的折線圖，直觀呈現遞減趨勢
figure('Position', [100, 100, 600, 450]);
plot(k_values, errors, '-o', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.5, 'MarkerFaceColor', [0.49 0.18 0.56]);
grid on;
xlabel('保留的奇異值個數 k');
ylabel('Frobenius 誤差 ||A - A_k||_F');
title('重建誤差（Frobenius norm）隨 k 增加而遞減');
saveas(gcf, 'svd_error_curve_matlab.png');
fprintf('已儲存誤差遞減曲線圖至: svd_error_curve_matlab.png\n');

fprintf('\n全部範例執行完畢。\n');

end % ch12_pca_applications（主函式結束）

%% =======================================================================
%% 本地函式定義
%% =======================================================================

function A_k = low_rank_approx(U, s, V, k)
    % 用前 k 個奇異值/奇異向量重建矩陣 A_k = U_k * diag(s_k) * V_k'
    A_k = U(:, 1:k) * diag(s(1:k)) * V(:, 1:k)';
end
