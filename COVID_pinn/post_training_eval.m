%% ===== Post-training evaluation =====
Nx = 60; Ny = 60; Nt = 45;

% Normalization functions (same as in make_PINN_colloc_points)
normX = @(idx) (idx - 1) / (Nx - 1);
normY = @(idx) (idx - 1) / (Ny - 1);
normT = @(idx) (idx - 1) / (Nt - 1);

S_pred_total = zeros(Nt,1);
I_pred_total = zeros(Nt,1);
R_pred_total = zeros(Nt,1);
% Initial condition
X0 = dlarray(ICmask','CB');
Y0 = forward(net, X0);
S0_hat = Y0(1,:);
% L0_hat = Y0(2,:);
I0_hat = Y0(3,:);
R0_hat = Y0(4,:);

sumS = sum(S0_hat);
sumI = sum(I0_hat);
sumR = sum(R0_hat);

for k = 1:Nt
    % Normalized time
    t_k = single(normT(k));

    % Normalized grid
    [xg, yg] = ndgrid(1:Nx, 1:Ny);
    x_norm = single(normX(xg(:)));
    y_norm = single(normY(yg(:)));
    t_norm = t_k * ones(numel(x_norm), 1, 'single');

    % Convert to dlarray
    Xdl = dlarray([x_norm'; y_norm'; t_norm'], 'CB');

    % Forward pass (no gradients)
    Yall = predict(net, Xdl);   % or forward(net, Xdl)

    % Extract predictions
    S_hat = reshape(extractdata(Yall(1,:)), Nx, Ny);
    I_hat = reshape(extractdata(Yall(2,:)), Nx, Ny);
    R_hat = reshape(extractdata(Yall(3,:)), Nx, Ny);

    % Sum spatially
    S_pred_total(k) = sum(S_hat(:));
    I_pred_total(k) = sum(I_hat(:));
    R_pred_total(k) = sum(R_hat(:));
end

% Append initial condition
S_pred_total = [extractdata(sumS); S_pred_total];
I_pred_total = [extractdata(sumI); I_pred_total];
R_pred_total = [extractdata(sumR); R_pred_total];

t_axis = 0:Nt;  % days

%% ===== True totals =====
S_true_total = squeeze(sum(sum(S_True, 1), 2));
I_true_total = squeeze(sum(sum(I_True, 1), 2));
R_true_total = squeeze(sum(sum(R_True, 1), 2));

S_true_total = [extractdata(sumS); S_true_total(:)];
I_true_total = [extractdata(sumI); I_true_total(:)];
R_true_total = [extractdata(sumR); R_true_total(:)];

%% ===== Plot =====
figure;
plot(t_axis, S_true_total, 'b-', 'LineWidth', 2); hold on;
plot(t_axis, S_pred_total, 'b--', 'LineWidth', 2);

plot(t_axis, I_true_total, 'r-', 'LineWidth', 2);
plot(t_axis, I_pred_total, 'r--', 'LineWidth', 2);

plot(t_axis, R_true_total, 'g-', 'LineWidth', 2);
plot(t_axis, R_pred_total, 'g--', 'LineWidth', 2);

legend({'S true','S pred','I true','I pred','R true','R pred'}, 'Location','best');
xlabel('Day (t)');
ylabel('Total population (summed over grid)');
title('PINN Predicted vs True SIR Dynamics');
grid on;