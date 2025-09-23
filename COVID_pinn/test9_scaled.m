%% PINN for 2D Reaction–Diffusion SIR PDE with comparisons
clear; clc;

%% Parameters
beta = 0.005; gamma = 0.1;
D_S = 0.05; D_I = 0.05; D_R = 0.05;

Nx = 3; Ny = 3; Nt = 3;
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

%% Build data anchors from FD baseline
% Choose anchor time indices (here: first, middle, last day)
tIdx = round([1, Nt/2, Nt]);   

% Build grid coordinates
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

[Xg,Yg,Tg] = ndgrid(x,y,t(tIdx));   % size Nx×Ny×numel(tIdx)

% Flatten into column vectors
xData = Xg(:);
yData = Yg(:);
tData = Tg(:);

% Collect true values at those points
Svals = S_true(tIdx,:,:);  Svals = Svals(:);
Ivals = I_true(tIdx,:,:);  Ivals = Ivals(:);
Rvals = R_true(tIdx,:,:);  Rvals = Rvals(:);

% Pack into dlarrays
Xdata = dlarray([tData'; xData'; yData'],'CB');  % 3×Nd
Sdata = dlarray(Svals','CB');   % 1×Nd
Idata = dlarray(Ivals','CB');   % 1×Nd
Rdata = dlarray(Rvals','CB');   % 1×Nd

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
numEpochs = 2;
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
% Scales (pick roughly the max magnitudes you expect)
Sscale = 50;   % per cell at t=0 (adjust if you use other units)
Iscale = 50;
Rscale = 50;

% Weights (tune a bit if needed)
wIC_I  = 50.0;   % infected IC heavier so the hotspot is learned
wData_I = 5.0;  % anchor I more strongly
wData_R = 2.0;  % anchor R moderately
wCons   = 1.0;  % conservation regularization
for epoch = 1:numEpochs
    disp(epoch);
    % Wrap collocation points as dlarray before dlfeval
    tInt = dlarray(collocInt(:,1)','CB');
    xInt = dlarray(collocInt(:,2)','CB');
    yInt = dlarray(collocInt(:,3)','CB');

    % [loss,grads] = dlfeval(@modelLoss,net,tInt,xInt,yInt, ...
    %                        collocBd,collocIC, ...
    %                        beta,gamma,D_S,D_I,D_R);
    [loss,grads] = dlfeval(@modelLoss, net, ...
      tInt,xInt,yInt,collocBd,collocIC, ...
      beta,gamma,D_S,D_I,D_R, ...
      Nt,Nx,Ny, ...
      Xdata,Sdata,Idata,Rdata, ...
      epoch, ...
      Sscale,Iscale,Rscale, ...
      wIC_I,wData_I,wData_R,wCons);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad, ...
    epoch,learnRate);

    % Update monitor
    % recordMetrics(monitor,epoch,Loss=double(loss));
    % updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    % monitor.Progress = 100*epoch/numEpochs;

    % Allow early stopping from GUI
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

%% True totals
S_true_sum = squeeze(sum(S_true,[2 3]));
I_true_sum = squeeze(sum(I_true,[2 3]));
R_true_sum = squeeze(sum(R_true,[2 3]));

% Predicted totals
S_pred_sum = squeeze(sum(S_pred,[2 3]));
I_pred_sum = squeeze(sum(I_pred,[2 3]));
R_pred_sum = squeeze(sum(R_pred,[2 3]));

%% Plot curves
days = 1:Nt;
figure;
subplot(3,1,1)
plot(days,S_true_sum,'b-',days,S_pred_sum,'r--','LineWidth',1.5)
legend('True','Predicted'); ylabel('S total')

subplot(3,1,2)
plot(days,I_true_sum,'b-',days,I_pred_sum,'r--','LineWidth',1.5)
legend('True','Predicted'); ylabel('I total')

subplot(3,1,3)
plot(days,R_true_sum,'b-',days,R_pred_sum,'r--','LineWidth',1.5)
legend('True','Predicted'); ylabel('R total'); xlabel('Day')

sgtitle('True vs PINN Predicted Epidemic Dynamics (spatial totals)')

%% Heatmap comparisons
selDays = [1, round(Nt/2), Nt];
for d = selDays
    figure;
    subplot(1,2,1)
    imagesc(x,y,squeeze(I_true(d,:,:))); axis equal tight; colorbar;
    title(['True I at day ',num2str(d)])

    subplot(1,2,2)
    imagesc(x,y,squeeze(I_pred(d,:,:))); axis equal tight; colorbar;
    title(['PINN Predicted I at day ',num2str(d)])
end

%% ---------------- Loss function ----------------

%% calculate gradients by summing over all points
% function [loss,grads] = modelLoss(net,t,x,y,collocBd,collocIC, ...
%                                   beta,gamma,D_S,D_I,D_R)
% 
%     % Pack all interior collocation points into one dlarray
%     Xall = dlarray([t; x; y],'CB');   % [3 × N]
%     Yall = forward(net,Xall);         % [3 × N]
% 
%     S = Yall(1,:); I = Yall(2,:); R = Yall(3,:);
% 
%     % --- Time derivatives (vectorized with sum-trick) ---
%     S_t = dlgradient(sum(S),t,EnableHigherDerivatives=true);
%     I_t = dlgradient(sum(I),t,EnableHigherDerivatives=true);
%     R_t = dlgradient(sum(R),t,EnableHigherDerivatives=true);
% 
%     % --- Spatial derivatives (vectorized with sum-trick) ---
%     S_x = dlgradient(sum(S),x,EnableHigherDerivatives=true);
%     S_y = dlgradient(sum(S),y,EnableHigherDerivatives=true);
%     I_x = dlgradient(sum(I),x,EnableHigherDerivatives=true);
%     I_y = dlgradient(sum(I),y,EnableHigherDerivatives=true);
%     R_x = dlgradient(sum(R),x,EnableHigherDerivatives=true);
%     R_y = dlgradient(sum(R),y,EnableHigherDerivatives=true);
% 
%     S_xx = dlgradient(sum(S_x),x);
%     S_yy = dlgradient(sum(S_y),y);
%     I_xx = dlgradient(sum(I_x),x);
%     I_yy = dlgradient(sum(I_y),y);
%     R_xx = dlgradient(sum(R_x),x);
%     R_yy = dlgradient(sum(R_y),y);
% 
%     % --- PDE residuals ---
%     dt = 0.1;
%     fS = S_t - dt * (D_S*(S_xx+S_yy) - beta*S.*I);
%     fI = I_t - dt * (D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
%     fR = R_t - dt * (D_R*(R_xx+R_yy) + gamma*I);
% 
%     lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
%         + mse(fI, zeros(size(fI),'like',fI)) ...
%         + mse(fR, zeros(size(fR),'like',fR));
% 
%     % --- Boundary condition: I=0 ---
%     Xbd = dlarray(single(collocBd)','CB');
%     Ybd = forward(net,Xbd);
%     lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));
% 
%     % --- Initial condition ---
%     Xic = dlarray(single(collocIC)','CB');
%     Yic = forward(net,Xic);
%     S0 = 50 * ones(size(Yic(1,:)),'like',Yic);   % Susceptibles in thousands
%     % Infected: 50 only at center cell
%     Nx = 10; Ny = 10;
%     x_coords = Xic(2,:);   % x ∈ [0,1]
%     y_coords = Xic(3,:);   % y ∈ [0,1]
% 
%     % Find closest grid point to (0.5,0.5)
%     [~,centerIdx] = min((x_coords-0.5).^2 + (y_coords-0.5).^2);
% 
%     I0 = zeros(size(Yic(2,:)),'like',Yic);
%     I0(centerIdx) = 50;
%        R0 = zeros(size(Yic(3,:)),'like',Yic);       % Initially recovered
%     % fprintf('true S: %f, pred S: %f', S0, Yic(1,1));
%     % fprintf('true I: %f, pred I: %f', I0, Yic(2,:));
%     % lossIC = mse(Yic(1,:), S0)/ mean(S0.^2) ...
%     %    + 5*mse(Yic(2,:), I0)/ mean(I0.^2) ...
%     %    + mse(Yic(3,:), R0)/ max(1, mean(R0.^2));
%     lossIC = mse(Yic(1,:), S0) ...
%        + 5*mse(Yic(2,:), I0) ...
%        + mse(Yic(3,:), R0);
% 
%     % --- Total loss ---
%     % disp("lossPDE", lossPDE, "lossBC", lossBC, "lossIC", lossIC)
%     fprintf('lossPDE: %f, lossBC: %f, lossIC: %f\n', lossPDE, lossBC, lossIC);
%     loss = lossPDE + lossBC + 10*lossIC;
%     disp(loss);
% 
%     % Gradients w.r.t. learnable params
%     grads = dlgradient(loss,net.Learnables);
% end

function [loss,grads] = modelLoss(net, ...
    t,x,y, ...                   % interior collocation coords (dlarray 1×N each, 'CB')
    collocBd,collocIC, ...       % boundary & IC collocation points (numeric N×3, from ndgrid selection)
    beta,gamma,D_S,D_I,D_R, ...  % PDE params (scalars)
    Nt,Nx,Ny, ...                % grid/time sizes (scalars)
    Xdata,Sdata,Idata,Rdata, ... % data anchors: Xdata is 3×Nd 'CB'; S/I/R are 1×Nd 'CB'
    epoch, ...                   % current epoch (scalar)
    Sscale,Iscale,Rscale, ...    % scaling constants for S/I/R (e.g., 50, 50, 50) 
    wIC_I, wData_I, wData_R, wCons) % weights: IC for I; data I, data R, conservation

% ---------------- Interior forward pass ----------------
Xall = dlarray([t; x; y],'CB');       % 3×N
Yall = forward(net,Xall);             % 3×N
S = Yall(1,:); I = Yall(2,:); R = Yall(3,:);

% ---------------- Derivatives (vectorized sum-trick) ----------------
S_t = dlgradient(sum(S),t,EnableHigherDerivatives=true);
I_t = dlgradient(sum(I),t,EnableHigherDerivatives=true);
R_t = dlgradient(sum(R),t,EnableHigherDerivatives=true);

S_x = dlgradient(sum(S),x,EnableHigherDerivatives=true);
S_y = dlgradient(sum(S),y,EnableHigherDerivatives=true);
I_x = dlgradient(sum(I),x,EnableHigherDerivatives=true);
I_y = dlgradient(sum(I),y,EnableHigherDerivatives=true);
R_x = dlgradient(sum(R),x,EnableHigherDerivatives=true);
R_y = dlgradient(sum(R),y,EnableHigherDerivatives=true);

S_xx = dlgradient(sum(S_x),x);
S_yy = dlgradient(sum(S_y),y);
I_xx = dlgradient(sum(I_x),x);
I_yy = dlgradient(sum(I_y),y);
R_xx = dlgradient(sum(R_x),x);
R_yy = dlgradient(sum(R_y),y);

% ---------------- PDE residuals (with dt balance) ----------------
dt = 1/(Nt-1);  % t normalized to [0,1]
fS = S_t - dt*( D_S*(S_xx+S_yy) - beta*S.*I );
fI = I_t - dt*( D_I*(I_xx+I_yy) + beta*S.*I - gamma*I );
fR = R_t - dt*( D_R*(R_xx+R_yy) + gamma*I );

lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
        + mse(fI, zeros(size(fI),'like',fI)) ...
        + mse(fR, zeros(size(fR),'like',fR));

% ---------------- Boundary condition: I=0 (all boundary points) ----------------
Xbd = dlarray(single(collocBd)','CB');   % 3×Nb
Ybd = forward(net,Xbd);
lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));

% ---------------- Initial condition (t=0):  S0 flat, I0 single-cell, R0=0 ----------------
Xic = dlarray(single(collocIC)','CB');   % 3×(Nx*Ny) at t=0
Yic = forward(net,Xic);                  % 3×(Nx*Ny)
S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

% Build true IC grids and flatten (order matches ndgrid + (:))
S0_grid = 50*ones(Nx,Ny,'like',extractdata(S_ic));   % per-cell susceptibles
I0_grid = zeros(Nx,Ny,'like',extractdata(I_ic));
R0_grid = zeros(Nx,Ny,'like',extractdata(R_ic));
% center cell:
i0 = round(0.5*(Nx-1))+1; j0 = round(0.5*(Ny-1))+1;
I0_grid(i0,j0) = 50;

% Flatten to row vectors and wrap as dlarray 1×(Nx*Ny) 'CB'
S0_ic = dlarray( reshape(S0_grid,1,[]), 'CB' );
I0_ic = dlarray( reshape(I0_grid,1,[]), 'CB' );
R0_ic = dlarray( reshape(R0_grid,1,[]), 'CB' );

% Scale in the loss so S doesn't dominate
lossIC = mse(S_ic./Sscale, S0_ic./Sscale) ...
       + wIC_I*mse(I_ic./Iscale, I0_ic./Iscale) ...
       + mse(R_ic./Rscale, R0_ic./Rscale);

% ---------------- Data anchors (supervised points across time) ----------------
% Xdata: 3×Nd 'CB', Sdata/Idata/Rdata: 1×Nd 'CB' (use same scaling in loss)
Yd = forward(net, Xdata);
Sd = Yd(1,:);  Id = Yd(2,:);  Rd = Yd(3,:);

% weights: S has 1.0 implicit; I and R passed in
lossData = mse(Sd./Sscale, Sdata./Sscale) ...
         + wData_I*mse(Id./Iscale, Idata./Iscale) ...
         + wData_R*mse(Rd./Rscale, Rdata./Rscale);

% ---------------- Conservation loss (global mass over interior batch) ----------------
% Use global average: keep mean(S+I+R) close to initial mean.
% Compute initial mean from IC vectors:
Nbar0 = mean( extractdata(S0_ic + I0_ic + R0_ic) ); % scalar
consResidual = mean(S + I + R) - Nbar0;             % scalar dlarray
lossCons = consResidual.^2;

% ---------------- Curriculum and total loss ----------------
wPDE = min(1.0, epoch/300);  % ramp PDE weight up during first ~300 epochs
loss = wPDE*lossPDE + lossBC + lossIC + lossData + wCons*lossCons;

% Optional: print monitor
fprintf('ep %4d | PDE %.3e | BC %.3e | IC %.3e | Data %.3e | Cons %.3e | Total %.3e\n', ...
    epoch, extractdata(lossPDE), extractdata(lossBC), extractdata(lossIC), ...
    extractdata(lossData), extractdata(lossCons), extractdata(loss));

% ---------------- Gradients ----------------
grads = dlgradient(loss, net.Learnables);
end
