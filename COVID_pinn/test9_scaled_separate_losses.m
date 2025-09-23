%% ================= PINN for 2D SIR PDE with Data Anchors =================
clear; clc;

%% Parameters
beta = 0.005; gamma = 0.1;
D_S = 0.05; D_I = 0.05; D_R = 0.05;

Nx = 3; Ny = 3; Nt = 10;        % grid size + time steps
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Collocation points
[X,Y,T] = ndgrid(x,y,t);
points = [T(:), X(:), Y(:)];

isBoundary = (X==0 | X==1 | Y==0 | Y==1);
isInitial  = (T==0);
isInterior = ~(isBoundary | isInitial);

collocIC  = points(isInitial(:),:);    % IC points (t=0)
collocDyn = points(isInterior(:),:);   % interior dynamics
collocBd  = points(isBoundary(:),:);   % boundary

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
tIdx = round([1, Nt/2, Nt]);   % choose anchor time indices
[Xg,Yg,Tg] = ndgrid(x,y,t(tIdx));

xData = Xg(:); yData = Yg(:); tData = Tg(:);

Svals = S_true(tIdx,:,:);  Svals = Svals(:);
Ivals = I_true(tIdx,:,:);  Ivals = Ivals(:);
Rvals = R_true(tIdx,:,:);  Rvals = Rvals(:);

Xdata = dlarray([tData'; xData'; yData'],'CB');  % 3×Nd
Sdata = dlarray(Svals','CB'); 
Idata = dlarray(Ivals','CB'); 
Rdata = dlarray(Rvals','CB');

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
numEpochsIC  = 500;
numEpochsDyn = 1500;
learnRate = 3e-3;
avgGrad = []; avgSqGrad = [];

Sscale = 50; Iscale = 50; Rscale = 50;
wIC_I  = 5.0;   % strong IC weight (infected hotspot)
wIC2   = 1.0;    % smaller IC weight in loop 2

%% ---------------- Loop 1: IC pretraining ----------------
for epoch = 1:numEpochsIC
    [lossIC,grads, S0, I0, R0] = dlfeval(@ICLoss,net,collocIC,Nx,Ny,Sscale,Iscale,Rscale,wIC_I);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
    if mod(epoch,50)==0
        fprintf('IC loop epoch %d | IC loss %.3e\n',epoch,extractdata(lossIC));
    end
end

%% ---------------- Loop 2: Full PINN training ----------------
for epoch = 1:numEpochsDyn
    % Convert collocation points into dlarrays
    tInt = dlarray(collocDyn(:,1)','CB');
    xInt = dlarray(collocDyn(:,2)','CB');
    yInt = dlarray(collocDyn(:,3)','CB');

    tBd = dlarray(collocBd(:,1)','CB');
    xBd = dlarray(collocBd(:,2)','CB');
    yBd = dlarray(collocBd(:,3)','CB');

    tIc = dlarray(collocIC(:,1)','CB');
    xIc = dlarray(collocIC(:,2)','CB');
    yIc = dlarray(collocIC(:,3)','CB');

    [loss,grads, S_ic2, I_ic2, R_ic2] = dlfeval(@FullLoss,net, ...
        tInt,xInt,yInt, ...         % interior
        tBd,xBd,yBd, ...            % boundary
        tIc,xIc,yIc, ...            % IC
        beta,gamma,D_S,D_I,D_R,Nt,Nx,Ny, ...
        Sscale,Iscale,Rscale,wIC2, ...
        Xdata,Sdata,Idata,Rdata);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);

    if mod(epoch,100)==0
        fprintf('Full loop epoch %d | Total loss %.3e\n',epoch,extractdata(loss));
    end
end
% 
%% PINN Prediction
[Tg,Xg,Yg] = ndgrid(t,x,y);
inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
out = predict(net,inp);
out = extractdata(out); % [3 × N]

S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
R_pred = reshape(out(3,:),[Nt,Nx,Ny]);

%% True and Predicted totals
S_true_sum = squeeze(sum(S_true,[2 3]));
I_true_sum = squeeze(sum(I_true,[2 3]));
R_true_sum = squeeze(sum(R_true,[2 3]));

S_pred_sum = squeeze(sum(S_pred,[2 3]));
I_pred_sum = squeeze(sum(I_pred,[2 3]));
R_pred_sum = squeeze(sum(R_pred,[2 3]));

%% Plot totals
days = 1:Nt;
figure;
subplot(3,1,1)
plot(days,S_true_sum,'b-',days,S_pred_sum,'r--','LineWidth',1.5)
legend('True','Pred'); ylabel('S total')

subplot(3,1,2)
plot(days,I_true_sum,'b-',days,I_pred_sum,'r--','LineWidth',1.5)
legend('True','Pred'); ylabel('I total')

subplot(3,1,3)
plot(days,R_true_sum,'b-',days,R_pred_sum,'r--','LineWidth',1.5)
legend('True','Pred'); ylabel('R total'); xlabel('Day')
sgtitle('True vs PINN Predicted Totals')

%% Heatmaps
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

%% ---------------- Loss Functions ----------------
function [lossIC,grads, S_ic, I_ic, R_ic] = ICLoss(net,collocIC,Nx,Ny,Sscale,Iscale,Rscale,wIC_I)
    Xic = dlarray(single(collocIC)','CB');
    Yic = forward(net,Xic);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    S0 = 50*ones(Nx,Ny);
    I0 = zeros(Nx,Ny); I0(round(Nx/2),round(Ny/2)) = 50;
    R0 = zeros(Nx,Ny);

    S0 = dlarray(reshape(S0,1,[]),'CB');
    I0 = dlarray(reshape(I0,1,[]),'CB');
    R0 = dlarray(reshape(R0,1,[]),'CB');

    lossIC = mse(S_ic./Sscale,S0./Sscale) ...
           + wIC_I*mse(I_ic./Iscale,I0./Iscale) ...
           + mse(R_ic./Rscale,R0./Rscale);

    grads = dlgradient(lossIC,net.Learnables);
end

function [loss,grads, S_ic, I_ic, R_ic] = FullLoss(net, ...
    tInt,xInt,yInt, ...        % interior dlarrays
    tBd,xBd,yBd, ...           % boundary dlarrays
    tIc,xIc,yIc, ...           % IC dlarrays
    beta,gamma,D_S,D_I,D_R,Nt,Nx,Ny, ...
    Sscale,Iscale,Rscale,wIC2, ...
    Xdata,Sdata,Idata,Rdata)

    %% Interior PDE
    Yall = forward(net,[tInt;xInt;yInt]); 
    S = Yall(1,:); I = Yall(2,:); R = Yall(3,:);

    S_t = dlgradient(sum(S),tInt,EnableHigherDerivatives=true);
    I_t = dlgradient(sum(I),tInt,EnableHigherDerivatives=true);
    R_t = dlgradient(sum(R),tInt,EnableHigherDerivatives=true);

    S_x = dlgradient(sum(S),xInt,EnableHigherDerivatives=true);
    S_y = dlgradient(sum(S),yInt,EnableHigherDerivatives=true);
    I_x = dlgradient(sum(I),xInt,EnableHigherDerivatives=true);
    I_y = dlgradient(sum(I),yInt,EnableHigherDerivatives=true);
    R_x = dlgradient(sum(R),xInt,EnableHigherDerivatives=true);
    R_y = dlgradient(sum(R),yInt,EnableHigherDerivatives=true);

    S_xx = dlgradient(sum(S_x),xInt); S_yy = dlgradient(sum(S_y),yInt);
    I_xx = dlgradient(sum(I_x),xInt); I_yy = dlgradient(sum(I_y),yInt);
    R_xx = dlgradient(sum(R_x),xInt); R_yy = dlgradient(sum(R_y),yInt);

    dt = 1/(Nt-1);
    fS = S_t - dt*(D_S*(S_xx+S_yy) - beta*S.*I);
    fI = I_t - dt*(D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
    fR = R_t - dt*(D_R*(R_xx+R_yy) + gamma*I);

    % lossPDE = mse(fS,0) + mse(fI,0) + mse(fR,0);
    lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
        + mse(fI, zeros(size(fI),'like',fI)) ...
        + mse(fR, zeros(size(fR),'like',fR));

    %% Boundary condition
    Ybd = forward(net,[tBd;xBd;yBd]);
    % lossBC = mse(Ybd(2,:),0);
    lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));

    %% Small IC penalty
    Yic = forward(net,[tIc;xIc;yIc]);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    S0 = 50*ones(Nx,Ny); 
    I0 = zeros(Nx,Ny); I0(round(Nx/2),round(Ny/2)) = 50; 
    R0 = zeros(Nx,Ny);

    S0 = dlarray(reshape(S0,1,[]),'CB');
    I0 = dlarray(reshape(I0,1,[]),'CB');
    R0 = dlarray(reshape(R0,1,[]),'CB');

    lossIC = mse(S_ic./Sscale,S0./Sscale) ...
           + mse(I_ic./Iscale,I0./Iscale) ...
           + mse(R_ic./Rscale,R0./Rscale);

    %% Data anchors
    Yd = forward(net,Xdata);
    Sd = Yd(1,:); Id = Yd(2,:); Rd = Yd(3,:);

    lossData = mse(Sd./Sscale,Sdata./Sscale) ...
             + 5*mse(Id./Iscale,Idata./Iscale) ...
             + 2*mse(Rd./Rscale,Rdata./Rscale);
    % lossIC = mse(S_ic./Sscale, S0_ic./Sscale) ...
    %    + wIC_I*mse(I_ic./Iscale, I0_ic./Iscale) ...
    %    + mse(R_ic./Rscale, R0_ic./Rscale);

    %% Total
    loss = lossPDE + lossBC + 0.1*lossIC + lossData;
    grads = dlgradient(loss,net.Learnables);
end
