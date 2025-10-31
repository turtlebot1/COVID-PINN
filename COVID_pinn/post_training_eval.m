S_pred_total = zeros(Nt,1);
L_pred_total = zeros(Nt,1);
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
    dayPts = getCollocPointsForDay(collocPts, k, Nt);
    x = dlarray(single(dayPts(:,1)'), 'CB');
    y = dlarray(single(dayPts(:,2)'), 'CB');
    t = dlarray(single(dayPts(:,3)'), 'CB');
    
    Xday = [x; y; t];
    Yall = predict(net, Xday);

    % % Extract predictions
    % S_hat = reshape(extractdata(Yall(1,:)), Nx, Ny);
    % I_hat = reshape(extractdata(Yall(2,:)), Nx, Ny);
    % R_hat = reshape(extractdata(Yall(3,:)), Nx, Ny);
    % 
    % % Sum spatially
    % S_pred_total(k) = sum(S_hat(:));
    % I_pred_total(k) = sum(I_hat(:));
    % R_pred_total(k) = sum(R_hat(:));
    % ---- Extract predicted fields ----
    S_hat = extractdata(Yall(1,:));
    L_hat = extractdata(Yall(2,:));
    I_hat = extractdata(Yall(3,:));
    R_hat = extractdata(Yall(4,:));
    
    % ---- Reshape to 2D grid (Nx × Ny) ----
    try
        S_hat = reshape(S_hat, Nx, Ny);
        L_hat = reshape(L_hat, Nx, Ny);
        I_hat = reshape(I_hat, Nx, Ny);
        R_hat = reshape(R_hat, Nx, Ny);
    catch
        % fallback if interior points < full grid size
        warning('Day %d: cannot reshape to full Nx×Ny; using partial region only.', k);
    end

    % ---- Compute total sums for this day ----
    S_pred_total(k) = sum(S_hat(:), 'omitnan');
    L_pred_total(k) = sum(L_hat(:), 'omitnan');
    I_pred_total(k) = sum(I_hat(:), 'omitnan');
    R_pred_total(k) = sum(R_hat(:), 'omitnan');
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

function dayPoints = getCollocPointsForDay(collocPts, dayIdx, Nt)
    % Convert day index (1...Nt) to normalized time
    normT = (dayIdx - 1) / (Nt - 1);
    
    % Logical mask to select only points for this day
    mask = abs(collocPts(:,3) - normT) < 1e-6;  % numerical tolerance
    dayPoints = collocPts(mask, :);
end