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
Nt = 5;
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

for epoch = 1:numEpochs
    disp(epoch);
    % Wrap collocation points as dlarray before dlfeval
    tInt = colloc.Int(:,1);
    xInt = colloc.Int(:,2);
    yInt = colloc.Int(:,3);
    % xInt = dlarray(collocInt(:,2)','CB');
    % yInt = dlarray(collocInt(:,3)','CB');

    [loss,grads] = dlfeval(@modelLoss,net,tInt,xInt,yInt, ...
                           colloc, param);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad, ...
    epoch,learnRate);

    % Update monitor
    % recordMetrics(monitor,epoch,Loss=double(loss));
    % updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    % monitor.Progress = 100*epoch/numEpochs;
    % 
    % % Allow early stopping from GUI
    % if monitor.Stop
    %     break
    % end
end

%% ---------------- Loss function ----------------
% spatial gradients in both dimensions
function [loss,grads] = modelLoss(net,t,x,y,colloc,param)

    Np = numel(t);

    % Storage
    S_t = zeros(1,Np,'like',t); 
    I_t = S_t; 
    R_t = S_t;

    S_x = S_t; S_y = S_t; S_xx = S_t; S_yy = S_t;
    I_x = S_t; I_y = S_t; I_xx = S_t; I_yy = S_t;
    R_x = S_t; R_y = S_t; R_xx = S_t; R_yy = S_t;
    S = zeros(1,Np,'like',t);
    I = S;
    R = S;

    % Loop over collocation points
    for k = 1:Np
        t_k = t(k); x_k = x(k); y_k = y(k);

        % Input for single point
        Xk = dlarray([t_k; x_k; y_k],'CB')

        % Forward pass
        Yk = forward(net,Xk);  % [3×1]
        Sk = Yk(1); Ik = Yk(2); Rk = Yk(3);

        % Save S,I,R for later use
        S(k) = Sk;
        I(k) = Ik;
        R(k) = Rk;

        % Time derivatives
        S_t = dlgradient(Sk, t_k, 'EnableHigherDerivatives', true);
        I_t = dlgradient(Ik, t_k, 'EnableHigherDerivatives', true);
        R_t = dlgradient(Rk, t_k, 'EnableHigherDerivatives', true);

        % Spatial derivatives
        sx = dlgradient(Sk, x_k, 'EnableHigherDerivatives', true);
        sy = dlgradient(Sk, y_k, 'EnableHigherDerivatives', true);
        ix = dlgradient(Ik, x_k, 'EnableHigherDerivatives', true);
        iy = dlgradient(Ik, y_k, 'EnableHigherDerivatives', true);
        rx = dlgradient(Rk, x_k, 'EnableHigherDerivatives', true);
        ry = dlgradient(Rk, y_k, 'EnableHigherDerivatives', true);

        sxx = dlgradient(sx, x_k);
        syy = dlgradient(sy, y_k);
        ixx = dlgradient(ix, x_k);
        iyy = dlgradient(iy, y_k);
        rxx = dlgradient(rx, x_k);
        ryy = dlgradient(ry, y_k);
        disp("Xk:")
        disp(size(Xk)); disp(dims(Xk));
        
        disp("Sk / Ik / Rk:")
        disp(size(Sk)); disp(dims(Sk));
        
        disp("S_t:")
        disp(size(S_t)); disp(dims(S_t));
        
        disp("S_xx:")
        disp(size(S_xx)); disp(dims(S_xx));

        % Assign
        S_x(k) = sx; S_y(k) = sy; S_xx(k) = sxx; S_yy(k) = syy;
        I_x(k) = ix; I_y(k) = iy; I_xx(k) = ixx; I_yy(k) = iyy;
        R_x(k) = rx; R_y(k) = ry; R_xx(k) = rxx; R_yy(k) = ryy;
    end

    % PDE residuals
    % Need forward pass again for all points (faster batch)
    % Xall = [t; x; y]
    % Yall = forward(net,Xall);
    % S = Yall(1,:); 
    % I = Yall(2,:); 
    % R = Yall(3,:);

    fS = S_t - (param.D_S*(S_xx+S_yy) - param.beta*S.*I);
    fI = I_t - (param.D_I*(I_xx+I_yy) + param.beta*S.*I - param.gamma*I);
    fR = R_t - (param.D_R*(R_xx+R_yy) + param.gamma*I);
    targetZeros = dlarray(zeros(size(fS),'like',extractdata(fS)),'CB');
    disp("fS:"); disp(size(fS)); disp(dims(fS));
    disp("fI:"); disp(size(fI)); disp(dims(fI));
    disp("fR:"); disp(size(fR)); disp(dims(fR));
    lossPDE = mse(fS, targetZeros) ...
        + mse(fI, targetZeros) ...
        + mse(fR, targetZeros);

    % Boundary condition: I=0
    Xbd = dlarray(single(colloc.Bd)','CB');
    Ybd = forward(net,Xbd);
    lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));

    % Initial condition
    Xic = dlarray(single(colloc.IC)','CB');
    Yic = forward(net,Xic);
    S0 = 1.0; 
    I0 = exp(-50*((Xic(2,:)-0.5).^2+(Xic(3,:)-0.5).^2));
    R0 = 0.0;
    % lossIC = mse(Yic(1,:), S0) ...
    %    + 5*mse(Yic(2,:), I0) ...
    %    + mse(Yic(3,:), R0);

    % Total loss
    loss = lossPDE + lossBC %+ lossIC;
    grads = dlgradient(loss,net.Learnables);
end

