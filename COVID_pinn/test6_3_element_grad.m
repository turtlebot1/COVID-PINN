%% PINN for 2D Reaction–Diffusion SIR PDE with comparisons
clear; clc;

%% Parameters
beta = 0.005; gamma = 0.1;
D_S = 0.05; D_I = 0.05; D_R = 0.05;

Nx = 5; Ny = 5; Nt = 5;
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Collocation points
[X,Y,T] = ndgrid(x,y,t);
points = [T(:), X(:), Y(:)];

isBoundary = (X==0 | X==1 | Y==0 | Y==1);
isInitial  = (T==0);
isInterior = ~(isBoundary | isInitial);

collocInt = points(isInterior(:),:);
collocBd  = points(isBoundary(:),:);
collocIC  = points(isInitial(:),:);

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
% tInt = dlarray(collocInt(:,1)','CB');
% xInt = dlarray(collocInt(:,2)','CB');
% yInt = dlarray(collocInt(:,3)','CB');

% Create training progress monitor
% monitor = trainingProgressMonitor( ...
%     Metrics="Loss", ...
%     Info=["Epoch","LearnRate"], ...
%     XLabel="Epoch");

%% Training loop
for epoch = 1:numEpochs
    disp(epoch);
    % Wrap collocation points as dlarray before dlfeval
    tInt = dlarray(collocInt(:,1)','CB');
    xInt = dlarray(collocInt(:,2)','CB');
    yInt = dlarray(collocInt(:,3)','CB');

    [loss,grads] = dlfeval(@modelLoss,net,tInt,xInt,yInt, ...
                           collocBd,collocIC, ...
                           beta,gamma,D_S,D_I,D_R);

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

%% PINN Prediction
[Tg,Xg,Yg] = ndgrid(t,x,y);
inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
out = predict(net,inp);
out = extractdata(out); % [3 × N]

S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
R_pred = reshape(out(3,:),[Nt,Nx,Ny]);

%% Finite-difference baseline (true simulation)
S_true = 50*ones(Nt,Nx,Ny); 
I_true = zeros(Nt,Nx,Ny); 
R_true = zeros(Nt,Nx,Ny);

I_true(1,round(Nx/2),round(Ny/2)) = 50; % infection bump at center
dt = 0.1;
for k = 2:Nt
    S_prev = squeeze(S_true(k-1,:,:));
    I_prev = squeeze(I_true(k-1,:,:));
    R_prev = squeeze(R_true(k-1,:,:));

    Lap = @(U) circshift(U,[1,0])+circshift(U,[-1,0])+circshift(U,[0,1])+circshift(U,[0,-1])-4*U;

    S_next = S_prev + dt*(D_S*Lap(S_prev) - beta*S_prev.*I_prev);
    I_next = I_prev + dt*(D_I*Lap(I_prev) + beta*S_prev.*I_prev - gamma*I_prev);
    R_next = R_prev + dt*(D_R*Lap(R_prev) + gamma*I_prev);

    % boundary: I=0
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;

    S_true(k,:,:) = S_next;
    I_true(k,:,:) = I_next;
    R_true(k,:,:) = R_next;
end

%% True totals
S_true_sum = squeeze(sum(S_true,[2 3]));
I_true_sum = squeeze(sum(I_true,[2 3]));
R_true_sum = squeeze(sum(R_true,[2 3]));

% Predicted totals
S_pred_sum = squeeze(sum(S_pred,[2 3]));
I_pred_sum = squeeze(sum(I_pred,[2 3]));
R_pred_sum = squeeze(sum(R_pred,[2 3]));

%% ---------------- Loss function ----------------
% spatial gradients in both dimensions
function [loss,grads] = modelLoss(net,t,x,y,collocBd,collocIC, ...  
                                  beta,gamma,D_S,D_I,D_R)

    Np = numel(t);

    % Storage
    S_t = zeros(1,Np,'like',t); 
    I_t = S_t; 
    R_t = S_t;

    S_x = S_t; S_y = S_t; S_xx = S_t; S_yy = S_t;
    I_x = S_t; I_y = S_t; I_xx = S_t; I_yy = S_t;
    R_x = S_t; R_y = S_t; R_xx = S_t; R_yy = S_t;

    % Loop over collocation points
    for k = 1:Np
        t_k = t(k); x_k = x(k); y_k = y(k);

        % Input for single point
        Xk = dlarray([t_k; x_k; y_k],'CB');

        % Forward pass
        Yk = forward(net,Xk);  % [3×1]
        Sk = Yk(1); Ik = Yk(2); Rk = Yk(3);

        % Time derivatives
        S_t(k) = dlgradient(Sk, t_k, 'EnableHigherDerivatives', true)
        I_t(k) = dlgradient(Ik, t_k, 'EnableHigherDerivatives', true);
        R_t(k) = dlgradient(Rk, t_k, 'EnableHigherDerivatives', true);

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

        % Assign
        S_x(k) = sx; S_y(k) = sy; S_xx(k) = sxx; S_yy(k) = syy;
        I_x(k) = ix; I_y(k) = iy; I_xx(k) = ixx; I_yy(k) = iyy;
        R_x(k) = rx; R_y(k) = ry; R_xx(k) = rxx; R_yy(k) = ryy;
    end

    % PDE residuals
    % Need forward pass again for all points (faster batch)
    Xall = [t; x; y];
    Yall = forward(net,Xall);
    S = Yall(1,:); 
    I = Yall(2,:); 
    R = Yall(3,:);

    fS = S_t - (D_S*(S_xx+S_yy) - beta*S.*I);
    fI = I_t - (D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
    fR = R_t - (D_R*(R_xx+R_yy) + gamma*I);
    lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
        + mse(fI, zeros(size(fI),'like',fI)) ...
        + mse(fR, zeros(size(fR),'like',fR));

    % Boundary condition: I=0
    Xbd = dlarray(single(collocBd)','CB');
    Ybd = forward(net,Xbd);
    lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));

    % Initial condition
    Xic = dlarray(single(collocIC)','CB');
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

