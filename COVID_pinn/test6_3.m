%% PINN for 2D Reaction–Diffusion SIR PDE with comparisons
clear; clc;

%% Parameters
param.beta = 0.1; 
param.gamma = 0.1;
param.D_S = 0.05; 
param.D_I = 0.5; 
param.D_R = 0.05;

Nx = 5; 
Ny = 5; 
Nt = 10;
linspc.x = linspace(0,Nx,Nx+1);
linspc.y = linspace(0,Ny,Ny+1);
linspc.t = linspace(0,Nt,Nt+1);

%% Collocation points
[X,Y,T] = ndgrid(linspc.x,linspc.y,linspc.t);
points = [T(:), X(:), Y(:)];

isBoundary = (X==0 | X==1 | Y==0 | Y==1);
isInitial  = (T==0);
isInterior = ~(isBoundary | isInitial);

colloc.Int = points(isInterior(:),:);
colloc.Bd  = points(isBoundary(:),:);
colloc.IC  = points(isInitial(:),:);

%% Define network
layers = [
    featureInputLayer(3,"Normalization","none")
    fullyConnectedLayer(64); tanhLayer
    fullyConnectedLayer(64); tanhLayer
    fullyConnectedLayer(64); tanhLayer
    fullyConnectedLayer(3)   % outputs [S,I,R]
];
net = dlnetwork(layers);

%% Training setup
numEpochs = 5;
learnRate = 3e-3;
avgGrad = []; avgSqGrad = [];

%% Finite-difference baseline (true simulation)
true.S = 1*ones(Nt,Nx,Ny); 
% enforce zero Dirichlet BCs for all SIR compartments
true.S(:,1,:)=0; true.S(:,end,:)=0; true.S(:,:,1)=0; true.S(:,:,end)=0;
true.I = zeros(Nt,Nx,Ny); 
true.R = zeros(Nt,Nx,Ny);

true.I(1,round(Nx/2),round(Ny/2)) = 1; % infection bump at center
dt = 0.1;
for k = 2:Nt
    S_prev = squeeze(true.S(k-1,:,:));
    I_prev = squeeze(true.I(k-1,:,:));
    R_prev = squeeze(true.R(k-1,:,:));

    Lap = @(U) circshift(U,[1,0])+circshift(U,[-1,0])+circshift(U,[0,1])+circshift(U,[0,-1])-4*U;

    S_next = S_prev + dt*(param.D_S*Lap(S_prev) - param.beta*S_prev.*I_prev);
    I_next = I_prev + dt*(param.D_I*Lap(I_prev) + param.beta*S_prev.*I_prev - param.gamma*I_prev);
    R_next = R_prev + dt*(param.D_R*Lap(R_prev) + param.gamma*I_prev);

    % boundary: I=0
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;
    % enforce zero Dirichlet BCs for all SIR compartments
    S_next(1,:)=0; S_next(end,:)=0; S_next(:,1)=0; S_next(:,end)=0;
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;
    R_next(1,:)=0; R_next(end,:)=0; R_next(:,1)=0; R_next(:,end)=0;

    true.S(k,:,:) = S_next;
    true.I(k,:,:) = I_next;
    true.R(k,:,:) = R_next;
end
%% clear redundant variables
clear S_next I_next R_next S_prev I_prev R_prev

%% Data prep for data loss
% Pick 3 anchor slices (early, mid, late)
tIdx = round([1, Nt/2, Nt]);   % e.g., [1, 5, 10] if Nt=10
[Xg, Yg, Tg] = ndgrid(1:Nx, 1:Ny, tIdx);   % full spatial grid at those times
Nd = numel(Xg);                            % number of anchor points
% Flatten to [Nx*Ny*Nt, 1]
S_flat = reshape(permute(true.S, [2 3 1]), [], Nt);  % (Nx*Ny) × Nt
I_flat = reshape(permute(true.I, [2 3 1]), [], Nt);
R_flat = reshape(permute(true.R, [2 3 1]), [], Nt);
% Linear indices into the flattened arrays
idxT = [tIdx];  % selected times

Sdata_vals = [];
Idata_vals = [];
Rdata_vals = [];
for k = 1:length(tIdx)
    Sdata_vals = [Sdata_vals; S_flat(:, idxT(k))];
    Idata_vals = [Idata_vals; I_flat(:, idxT(k))];
    Rdata_vals = [Rdata_vals; R_flat(:, idxT(k))];
end
xD = (Xg(:)-1)/(Nx-1);   % Nx → [0,1]
yD = (Yg(:)-1)/(Ny-1);
tD = (Tg(:)-1)/(Nt-1);   % Nt → [0,1]
% Input coordinates for data anchors
Xdata = dlarray([tD'; xD'; yD'], 'CB');   % 3 × Nd

% True values as dlarrays
Sdata = dlarray(Sdata_vals', 'CB');   % 1 × Nd
Idata = dlarray(Idata_vals', 'CB');
Rdata = dlarray(Rdata_vals', 'CB');

wPDE  = dlarray(1.0);
wIC   = dlarray(1.0);
wBC   = dlarray(1.0);
wData = dlarray(1.0);
% Initialize training rate tracking GradNorm
initialLosses = struct('pde', NaN, 'ic', NaN, 'bc', NaN, 'data', NaN);

colloc.Int = dlarray(single(colloc.Int'), 'CB');
colloc.Bd  = dlarray(single(colloc.Bd'),  'CB');
colloc.IC  = dlarray(single(colloc.IC'),  'CB');

%% Training loop
for epoch = 1:numEpochs
    disp(epoch);
    % Wrap collocation points as dlarray before dlfeval
    % tInt = dlarray(colloc.Int(:,1)','CB');
    % xInt = dlarray(colloc.Int(:,2)','CB');
    % yInt = dlarray(colloc.Int(:,3)','CB');
    Xint = colloc.Int; 
    tInt = Xint(1,:);         % traced
    xInt = Xint(2,:);
    yInt = Xint(3,:);

    [losses, grads, Gnorms, Ybd, Yic, S0] = dlfeval(@modelLossGradNorm,net, ...
                                        tInt,xInt,yInt,colloc, ...
                                        param,Xdata,Sdata,Idata,Rdata, ...
                                        wPDE,wIC,wBC,wData);

    % Update network
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
    
    % ---------------- GradNorm weight update ----------------
    if epoch == 1
        totalInit = double(losses.pde + losses.ic + losses.bc + losses.data);
        initialLosses.pde  = double(losses.pde)/ totalInit;
        initialLosses.ic   = double(losses.ic)/ totalInit;
        initialLosses.bc   = double(losses.bc)/ totalInit;
        initialLosses.data = double(losses.data)/ totalInit;
    end
    % Compute training rate ratios
    rPDE  = double(losses.pde)  / (initialLosses.pde  + 1e-6);
    rIC   = double(losses.ic)   / (initialLosses.ic   + 1e-6);
    rBC   = double(losses.bc)   / (initialLosses.bc   + 1e-6);
    rData = double(losses.data) / (initialLosses.data + 1e-6);
    % Compute mean gradient norm
    Gmean = mean([Gnorms.pde, Gnorms.ic, Gnorms.bc, Gnorms.data]);
    % Update each weight using GradNorm update rule
    alpha = 0.12;
    
    targetPDE  = Gmean * (rPDE  ^ alpha);
    targetIC   = Gmean * (rIC   ^ alpha);
    targetBC   = Gmean * (rBC   ^ alpha);
    targetData = Gmean * (rData ^ alpha);
    
    % GradNorm pseudo-gradients
    eta = 0.0025;
    Gnorms.data;
    targetData;
    grad_wPDE  = Gnorms.pde  - targetPDE;
    grad_wIC   = Gnorms.ic   - targetIC;
    grad_wBC   = Gnorms.bc   - targetBC;
    grad_wData = Gnorms.data - targetData;
    
    % Update task weights
    wPDE  = wPDE  - eta * grad_wPDE;  1e-3;
    wIC   = min(wIC   - eta * grad_wIC,   1);
    wBC   = wBC   - eta * grad_wBC;   1e-3;
    wData = wData - eta * grad_wData; 1e-3;
    
    % Normalize
    wsum  = wPDE + wIC + wBC + wData + 1e-6;
    wPDE  = wPDE  * 4 / wsum;
    wIC   = wIC   * 4 / wsum;
    % wIC = 2; 
    wBC   = wBC   * 4 / wsum;
    wData = wData * 4 / wsum;
    % wData = 2;
    % fprintf('Epoch %d: wPDE = %.3f, wIC = %.3f, wBC = %.3f, wData = %.3f\n', ...
    %     epoch, wPDE, wIC, wBC, wData);
end

%% PINN Prediction
% [Tg,Xg,Yg] = ndgrid(linspc.t, linspc.x, linspc.y);
% inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
% out = predict(net,inp);
% out = extractdata(out); % [3 × N]
% 
% S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
% I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
% R_pred = reshape(out(3,:),[Nt,Nx,Ny]);
% 
% %% True totals
% S_true_sum = squeeze(sum(true.S,[2 3]));
% I_true_sum = squeeze(sum(true.I,[2 3]));
% R_true_sum = squeeze(sum(true.R,[2 3]));
% 
% % Predicted totals
% S_pred_sum = squeeze(sum(S_pred,[2 3]));
% I_pred_sum = squeeze(sum(I_pred,[2 3]));
% R_pred_sum = squeeze(sum(R_pred,[2 3]));

%% ---------------- Loss function with GradNorm ----------------
function [losses, grads, Gnorms, Ybd, Yic, S0] = modelLossGradNorm(net,t,x,y,colloc, ...
                                  param, Xdata,Sdata,Idata,Rdata, ...
                                  wPDE,wIC,wBC,wData)
    lossPDE = 0;
    for day = 0:45
        % Select collocation points at the current day
        mask = (colloc.Int(:,1) == day);
        % icDay = colloc.Int(mask,:);
        
        if ~any(mask)
            continue  % No points for this day, skip
        end
        
        % Prepare dlarray input for the network (shape [3, N])
        % Xday = dlarray(single(icDay)', 'CB'); % Inputs: [t; x; y]
        Xday = colloc.Int(:,mask);

        % Forward pass
        Yday = forward(net, Xday);

        % Extract outputs
        S = Yday(1,:);
        I = Yday(2,:);
        R = Yday(3,:);

        % Compute first derivatives wrt inputs (t,x,y)
        dS_dX = dlgradient(sum(S), Xday, 'EnableHigherDerivatives', true);
        dI_dX = dlgradient(sum(I), Xday, 'EnableHigherDerivatives', true);
        dR_dX = dlgradient(sum(R), Xday, 'EnableHigherDerivatives', true);

        % Extract partial derivatives wrt t, x, y:
        disp("varan")
        S_t = dS_dX(1, :)
        S_x = dS_dX(2, :);
        S_y = dS_dX(3, :);

        I_t = dI_dX(1, :);
        I_x = dI_dX(2, :);
        I_y = dI_dX(3, :);

        R_t = dR_dX(1, :);
        R_x = dR_dX(2, :);
        R_y = dR_dX(3, :);

        % Compute second derivatives wrt x and y:
        S_xx = dlgradient(sum(S_x), Xday(2, :));
        S_yy = dlgradient(sum(S_y), Xday(3, :));
        I_xx = dlgradient(sum(I_x), Xday(2, :));
        I_yy = dlgradient(sum(I_y), Xday(3, :));
        R_xx = dlgradient(sum(R_x), Xday(2, :));
        R_yy = dlgradient(sum(R_y), Xday(3, :));

        % PDE residuals
        fS = S_t - (param.D_S * (S_xx + S_yy) - param.beta * S .* I);
        fI = I_t - (param.D_I * (I_xx + I_yy) + param.beta * S .* I - param.gamma * I);
        fR = R_t - (param.D_R * (R_xx + R_yy) + param.gamma * I);

        % Accumulate PDE loss (mean squared error)
        lossDay = mse(fS, zeros(size(fS), 'like', fS)) + ...
                  mse(fI, zeros(size(fI), 'like', fI)) + ...
                  mse(fR, zeros(size(fR), 'like', fR));
        lossPDE = lossPDE + lossDay;
    end
    losses.pde = lossPDE;

    % ---------------- Boundary condition ----------------
    Xbd = colloc.Bd;
    Ybd = forward(net,Xbd);
    losses.bc = mse(Ybd(1,:), zeros(size(Ybd(1,:)),'like',Ybd)) + ...   % S = 0
         mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd)) + ...   % I = 0
         mse(Ybd(3,:), zeros(size(Ybd(3,:)),'like',Ybd));

    % ---------------- Initial condition ----------------
    Xic = colloc.IC;
    Yic = forward(net,Xic);
    S0 = 1 * ones(size(Yic(1,:)),'like',Yic);   % Susceptibles in thousands
    Nx = 5; Ny = 5;

    % Zero out the boundaries (as before)
    x_coords = Xic(2,:);   % x ∈ [0,1]
    y_coords = Xic(3,:);   % y ∈ [0,1]

    % Grid step
    dx = 1 / (Nx - 1);
    dy = 1 / (Ny - 1);

    % Boundary mask
    isBoundary = (x_coords == 0) | (x_coords == 1) | ...
                 (y_coords == 0) | (y_coords == 1);

    % Find center point and its immediate neighbors
    center_x = 0.5;
    center_y = 0.5;

    % Any point within ±dx of center in both x and y
    % isInfectedRegion = (abs(x_coords - center_x) <= dx) & ...
    %                    (abs(y_coords - center_y) <= dy);

    % Combine boundary and infected region
    isNotSusceptible = isBoundary;

    % Set S0 = 0 where people are not susceptible
    S0(isNotSusceptible) = 0;
    % Infected: 50 only at center cell
    x_coords = Xic(2,:);   % x ∈ [0,1]
    y_coords = Xic(3,:);   % y ∈ [0,1]

    % Find closest grid point to (0.5,0.5)
    [~,centerIdx] = min((x_coords-0.5).^2 + (y_coords-0.5).^2);

    I0 = zeros(size(Yic(2,:)),'like',Yic);
    I0(centerIdx) = 1;
    R0 = zeros(size(Yic(3,:)),'like',Yic);       % Initially recovered
    losses.ic = mse(Yic(1,:), S0) ...
       + mse(Yic(2,:), I0) ...
       + mse(Yic(3,:), R0);

    % ---------------- Data anchors ----------------
    Yd = forward(net,Xdata);
    Sd = Yd(1,:); Id = Yd(2,:); Rd = Yd(3,:);
    losses.data = mse(Sd,Sdata) + mse(Id,Idata) + mse(Rd,Rdata);

    % ---------------- Weighted total loss ----------------
    % fprintf('Losses: PDE=%.3f, IC=%.3f, BC=%.3f, Data=%.3f\n', ...
    %     double(losses.pde), double(losses.ic), double(losses.bc), double(losses.data));

    loss = wPDE*losses.pde + wBC*losses.bc + wIC*losses.ic + wData*losses.data;

    % Gradients wrt network
    grads = dlgradient(loss,net.Learnables);

    % ---------------- Gradient norms per task ----------------
    % Use one reference parameter (say first layer weights)
    refParam = net.Learnables.Value{1};

    % Example inside modelLossGradNorm
% Compute gradient norms for each task loss
g_pde  = dlgradient(losses.pde, net.Learnables, 'RetainData', true);
g_ic   = dlgradient(losses.ic,  net.Learnables, 'RetainData', true);
g_bc   = dlgradient(losses.bc,  net.Learnables, 'RetainData', true);
g_data = dlgradient(losses.data,net.Learnables);

% Collapse into scalars
Gnorms.pde  = gradNormFromCell(g_pde);
Gnorms.ic   = gradNormFromCell(g_ic);
Gnorms.bc   = gradNormFromCell(g_bc);
Gnorms.data = gradNormFromCell(g_data);
end

function gnorm = gradNormFromCell(grads)
    gnorm = 0;
    % grads is a table with a column "Value"
    for i = 1:height(grads)
        gcell = grads.Value{i};   % extract the dlarray from cell
        if ~isempty(gcell)
            gnorm = gnorm + sum(double(gcell).^2,'all');
        end
    end
    gnorm = sqrt(gnorm);
end