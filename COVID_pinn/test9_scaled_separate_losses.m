% %% ================= PINN for 2D SIR PDE with Data Anchors =================
% clear; clc;
% 
% %% Parameters
% beta = 0.1; gamma = 0.1;
% D_S = 0.05; D_I = 0.05; D_R = 0.05;
% 
% Nx = 3; Ny = 3; Nt = 10;        % grid size + time steps
% x = linspace(0,1,Nx);
% y = linspace(0,1,Ny);
% t = linspace(0,1,Nt);
% 
% %% Collocation points
% [X,Y,T] = ndgrid(x,y,t);
% points = [T(:), X(:), Y(:)];
% 
% isBoundary = (X==0 | X==1 | Y==0 | Y==1);
% isInitial  = (T==0);
% isInterior = ~(isBoundary | isInitial);
% 
% collocIC  = points(isInitial(:),:);    % IC points (t=0)
% collocDyn = points(isInterior(:),:);   % interior dynamics
% collocBd  = points(isBoundary(:),:);   % boundary
% 
% %% Finite-difference baseline (true simulation)
% S_true = 50*ones(Nt,Nx,Ny); 
% I_true = zeros(Nt,Nx,Ny); 
% R_true = zeros(Nt,Nx,Ny);
% 
% I_true(1,round(Nx/2),round(Ny/2)) = 50; % infection bump at center
% dt = 0.1;
% for k = 2:Nt
%     S_prev = squeeze(S_true(k-1,:,:));
%     I_prev = squeeze(I_true(k-1,:,:));
%     R_prev = squeeze(R_true(k-1,:,:));
% 
%     Lap = @(U) circshift(U,[1,0])+circshift(U,[-1,0])+circshift(U,[0,1])+circshift(U,[0,-1])-4*U;
% 
%     S_next = S_prev + dt*(D_S*Lap(S_prev) - beta*S_prev.*I_prev);
%     I_next = I_prev + dt*(D_I*Lap(I_prev) + beta*S_prev.*I_prev - gamma*I_prev);
%     R_next = R_prev + dt*(D_R*Lap(R_prev) + gamma*I_prev);
% 
%     % boundary: I=0
%     I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;
% 
%     S_true(k,:,:) = S_next;
%     I_true(k,:,:) = I_next;
%     R_true(k,:,:) = R_next;
% end
% 
% %% Build data anchors from FD baseline
% tIdx = round([1, Nt/2, Nt]);   % choose anchor time indices
% [Xg,Yg,Tg] = ndgrid(x,y,t(tIdx));
% 
% xData = Xg(:); yData = Yg(:); tData = Tg(:);
% 
% Svals = S_true(tIdx,:,:);  Svals = Svals(:);
% Ivals = I_true(tIdx,:,:);  Ivals = Ivals(:);
% Rvals = R_true(tIdx,:,:);  Rvals = Rvals(:);
% 
% Xdata = dlarray([tData'; xData'; yData'],'CB');  % 3×Nd
% Sdata = dlarray(Svals','CB'); 
% Idata = dlarray(Ivals','CB'); 
% Rdata = dlarray(Rvals','CB');
% 
% %% Define network
% layers = [
%     featureInputLayer(3,"Normalization","none")
%     fullyConnectedLayer(64); tanhLayer
%     fullyConnectedLayer(64); tanhLayer
%     fullyConnectedLayer(64); tanhLayer
%     fullyConnectedLayer(3)   % outputs [S,I,R]
% ];
% net = dlnetwork(layers);
% 
% %% Training setup
% numEpochsIC  = 500;
% numEpochsDyn = 500;
% learnRate = 3e-3;
% avgGrad = []; avgSqGrad = [];
% 
% Sscale = 50; Iscale = 50; Rscale = 50;
% wIC_I  = 5.0;   % strong IC weight (infected hotspot)
% wIC2   = 5.0;    % smaller IC weight in loop 2
% 
% %% ---------------- Loop 1: IC pretraining ----------------
% for epoch = 1:numEpochsIC
%     [lossIC,grads, S0, I0, R0] = dlfeval(@ICLoss,net,collocIC,Nx,Ny,Sscale,Iscale,Rscale,wIC_I);
%     [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
%     if mod(epoch,50)==0
%         fprintf('IC loop epoch %d | IC loss %.3e\n',epoch,extractdata(lossIC));
%     end
% end
% 
% %% ---------------- Loop 2: Full PINN training ----------------
% for epoch = 1:numEpochsDyn
%     % Convert collocation points into dlarrays
%     tInt = dlarray(collocDyn(:,1)','CB');
%     xInt = dlarray(collocDyn(:,2)','CB');
%     yInt = dlarray(collocDyn(:,3)','CB');
% 
%     tBd = dlarray(collocBd(:,1)','CB');
%     xBd = dlarray(collocBd(:,2)','CB');
%     yBd = dlarray(collocBd(:,3)','CB');
% 
%     tIc = dlarray(collocIC(:,1)','CB');
%     xIc = dlarray(collocIC(:,2)','CB');
%     yIc = dlarray(collocIC(:,3)','CB');
% 
%     [loss,grads, S_ic2, I_ic2, R_ic2] = dlfeval(@FullLoss,net, ...
%         tInt,xInt,yInt, ...         % interior
%         tBd,xBd,yBd, ...            % boundary
%         tIc,xIc,yIc, ...            % IC
%         beta,gamma,D_S,D_I,D_R,Nt,Nx,Ny, ...
%         Sscale,Iscale,Rscale,wIC2, ...
%         Xdata,Sdata,Idata,Rdata);
% 
%     [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
% 
%     if mod(epoch,100)==0
%         fprintf('Full loop epoch %d | Total loss %.3e\n',epoch,extractdata(loss));
%     end
% end
% % 
% %% PINN Prediction
% [Tg,Xg,Yg] = ndgrid(t,x,y);
% inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
% out = predict(net,inp);
% out = extractdata(out); % [3 × N]
% 
% S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
% I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
% R_pred = reshape(out(3,:),[Nt,Nx,Ny]);
% 
% %% True and Predicted totals
% S_true_sum = squeeze(sum(S_true,[2 3]));
% I_true_sum = squeeze(sum(I_true,[2 3]));
% R_true_sum = squeeze(sum(R_true,[2 3]));
% 
% S_pred_sum = squeeze(sum(S_pred,[2 3]));
% I_pred_sum = squeeze(sum(I_pred,[2 3]));
% R_pred_sum = squeeze(sum(R_pred,[2 3]));
% 
% %% Plot totals
% days = 1:Nt;
% figure;
% subplot(3,1,1)
% plot(days,S_true_sum,'b-',days,S_pred_sum,'r--','LineWidth',1.5)
% legend('True','Pred'); ylabel('S total')
% 
% subplot(3,1,2)
% plot(days,I_true_sum,'b-',days,I_pred_sum,'r--','LineWidth',1.5)
% legend('True','Pred'); ylabel('I total')
% 
% subplot(3,1,3)
% plot(days,R_true_sum,'b-',days,R_pred_sum,'r--','LineWidth',1.5)
% legend('True','Pred'); ylabel('R total'); xlabel('Day')
% sgtitle('True vs PINN Predicted Totals')
% 
% %% Heatmaps
% selDays = [1, round(Nt/2), Nt];
% for d = selDays
%     figure;
%     subplot(1,2,1)
%     imagesc(x,y,squeeze(I_true(d,:,:))); axis equal tight; colorbar;
%     title(['True I at day ',num2str(d)])
%     subplot(1,2,2)
%     imagesc(x,y,squeeze(I_pred(d,:,:))); axis equal tight; colorbar;
%     title(['PINN Predicted I at day ',num2str(d)])
% end
% 
% %% ---------------- Loss Functions ----------------
% function [lossIC,grads, S_ic, I_ic, R_ic] = ICLoss(net,collocIC,Nx,Ny,Sscale,Iscale,Rscale,wIC_I)
%     Xic = dlarray(single(collocIC)','CB');
%     Yic = forward(net,Xic);
%     S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);
% 
%     S0 = 50*ones(Nx,Ny);
%     I0 = zeros(Nx,Ny); I0(round(Nx/2),round(Ny/2)) = 50;
%     R0 = zeros(Nx,Ny);
% 
%     S0 = dlarray(reshape(S0,1,[]),'CB');
%     I0 = dlarray(reshape(I0,1,[]),'CB');
%     R0 = dlarray(reshape(R0,1,[]),'CB');
% 
%     lossIC = mse(S_ic./Sscale,S0./Sscale) ...
%            + wIC_I*mse(I_ic./Iscale,I0./Iscale) ...
%            + mse(R_ic./Rscale,R0./Rscale);
% 
%     grads = dlgradient(lossIC,net.Learnables);
% end
% 
% function [loss,grads, S_ic, I_ic, R_ic] = FullLoss(net, ...
%     tInt,xInt,yInt, ...        % interior dlarrays
%     tBd,xBd,yBd, ...           % boundary dlarrays
%     tIc,xIc,yIc, ...           % IC dlarrays
%     beta,gamma,D_S,D_I,D_R,Nt,Nx,Ny, ...
%     Sscale,Iscale,Rscale,wIC2, ...
%     Xdata,Sdata,Idata,Rdata)
% 
%     %% Interior PDE
%     Yall = forward(net,[tInt;xInt;yInt]); 
%     S = Yall(1,:); 
%     I = Yall(2,:);
%     R = Yall(3,:);
% 
%     S_t = dlgradient(sum(S),tInt,EnableHigherDerivatives=true);
%     I_t = dlgradient(sum(I),tInt,EnableHigherDerivatives=true);
%     R_t = dlgradient(sum(R),tInt,EnableHigherDerivatives=true);
% 
%     S_x = dlgradient(sum(S),xInt,EnableHigherDerivatives=true);
%     S_y = dlgradient(sum(S),yInt,EnableHigherDerivatives=true);
%     I_x = dlgradient(sum(I),xInt,EnableHigherDerivatives=true);
%     I_y = dlgradient(sum(I),yInt,EnableHigherDerivatives=true);
%     R_x = dlgradient(sum(R),xInt,EnableHigherDerivatives=true);
%     R_y = dlgradient(sum(R),yInt,EnableHigherDerivatives=true);
% 
%     S_xx = dlgradient(sum(S_x),xInt); S_yy = dlgradient(sum(S_y),yInt);
%     I_xx = dlgradient(sum(I_x),xInt); I_yy = dlgradient(sum(I_y),yInt);
%     R_xx = dlgradient(sum(R_x),xInt); R_yy = dlgradient(sum(R_y),yInt);
% 
%     dt = 1/(Nt-1);
%     dt = 1;
%     fS = S_t - dt*(D_S*(S_xx+S_yy) - beta*S.*I);
%     fI = I_t - dt*(D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
%     fR = R_t - dt*(D_R*(R_xx+R_yy) + gamma*I);
% 
%     % lossPDE = mse(fS,0) + mse(fI,0) + mse(fR,0);
%     lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
%         + mse(fI, zeros(size(fI),'like',fI)) ...
%         + mse(fR, zeros(size(fR),'like',fR));
% 
%     %% Boundary condition
%     Ybd = forward(net,[tBd;xBd;yBd]);
%     % lossBC = mse(Ybd(2,:),0);
%     lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));
% 
%     %% Small IC penalty
%     Yic = forward(net,[tIc;xIc;yIc]);
%     S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);
% 
%     S0 = 50*ones(Nx,Ny); 
%     I0 = zeros(Nx,Ny); I0(round(Nx/2),round(Ny/2)) = 50; 
%     R0 = zeros(Nx,Ny);
% 
%     S0 = dlarray(reshape(S0,1,[]),'CB');
%     I0 = dlarray(reshape(I0,1,[]),'CB');
%     R0 = dlarray(reshape(R0,1,[]),'CB');
% 
%     lossIC = mse(S_ic./Sscale,S0./Sscale) ...
%            + mse(I_ic./Iscale,I0./Iscale) ...
%            + mse(R_ic./Rscale,R0./Rscale);
% 
%     %% Data anchors
%     Yd = forward(net,Xdata);
%     Sd = Yd(1,:); Id = Yd(2,:); Rd = Yd(3,:);
% 
%     lossData = mse(Sd./Sscale,Sdata./Sscale) ...
%              + 5*mse(Id./Iscale,Idata./Iscale) ...
%              + 2*mse(Rd./Rscale,Rdata./Rscale)
%     % lossIC = mse(S_ic./Sscale, S0_ic./Sscale) ...
%     %    + wIC_I*mse(I_ic./Iscale, I0_ic./Iscale) ...
%     %    + mse(R_ic./Rscale, R0_ic./Rscale);
% 
%     %% Total
%     loss = lossPDE + lossBC + 0.1*lossIC + lossData;
%     grads = dlgradient(loss,net.Learnables);
% end

%% ================= PINN for 2D SIR PDE with Data Anchors =================
clear; clc; rng(0);

%% Parameters (continuous PDE, no discrete dt inside PINN residuals)
beta  = 0.10; 
gamma = 0.10;
D_S   = 0.05; 
D_I   = 0.05; 
D_R   = 0.05;

Nx = 5; Ny = 5; Nt = 10;              % grid size + time steps
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Collocation points (t, x, y)
[X,Y,T] = ndgrid(x,y,t);
points = [T(:), X(:), Y(:)];

isBoundary = (X==0 | X==1 | Y==0 | Y==1);  % spatial boundary at any t
isInitial  = (T==0);                       % initial slice at t=0
isInterior = ~(isBoundary | isInitial);    % interior t>0 and not boundary

collocIC  = points(isInitial(:),:);    % IC points (t=0, all (x,y))
collocDyn = points(isInterior(:),:);   % interior dynamics
collocBd  = points(isBoundary(:),:);   % boundary (all t, boundary (x,y))

%% ---------- Finite-difference baseline (for sparse data anchors) ----------
S_true = 50*ones(Nt,Nx,Ny,'single'); 
I_true = zeros(Nt,Nx,Ny,'single'); 
R_true = zeros(Nt,Nx,Ny,'single');

I_true(1,round(Nx/2),round(Ny/2)) = 50; % infection bump at center
dt_fd = 0.1;  % ONLY for generating synthetic data (not used in PINN residual)

Lap = @(U) circshift(U,[1,0]) + circshift(U,[-1,0]) + ...
           circshift(U,[0,1]) + circshift(U,[0,-1]) - 4*U;

for k = 2:Nt
    S_prev = squeeze(S_true(k-1,:,:));
    I_prev = squeeze(I_true(k-1,:,:));
    R_prev = squeeze(R_true(k-1,:,:));

    S_next = S_prev + dt_fd*(D_S*Lap(S_prev) - beta*S_prev.*I_prev);
    I_next = I_prev + dt_fd*(D_I*Lap(I_prev) + beta*S_prev.*I_prev - gamma*I_prev);
    R_next = R_prev + dt_fd*(D_R*Lap(R_prev) + gamma*I_prev);

    % Dirichlet boundary on I (matches PINN BC)
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;

    S_true(k,:,:) = S_next;
    I_true(k,:,:) = I_next;
    R_true(k,:,:) = R_next;
end

%% ---------- Build sparse data anchors from FD baseline ----------
tIdx = round([1, Nt/2, Nt]);   % choose a few anchor times
[Xg,Yg,Tg] = ndgrid(x,y,t(tIdx));

xData = single(Xg(:)); 
yData = single(Yg(:)); 
tData = single(Tg(:));

Svals = single(S_true(tIdx,:,:));  Svals = Svals(:);
Ivals = single(I_true(tIdx,:,:));  Ivals = Ivals(:);
Rvals = single(R_true(tIdx,:,:));  Rvals = Rvals(:);

Xdata = dlarray([tData'; xData'; yData'],'CB');  % (3 × Nd) dlarray single
Sdata = dlarray(Svals','CB'); 
Idata = dlarray(Ivals','CB'); 
Rdata = dlarray(Rvals','CB');

%% ---------- Define network ----------
layers = [
    featureInputLayer(3,"Normalization","none","Name","in")
    fullyConnectedLayer(64,"Name","fc1")
    tanhLayer("Name","t1")
    fullyConnectedLayer(64,"Name","fc2")
    tanhLayer("Name","t2")
    fullyConnectedLayer(64,"Name","fc3")
    tanhLayer("Name","t3")
    fullyConnectedLayer(3,"Name","out")   % outputs [S,I,R]
];
net = dlnetwork(layers);

%% ---------- Training setup ----------
numEpochsIC  = 500;
numEpochsDyn = 500;
learnRate = 3e-3;
avgGrad = []; avgSqGrad = [];

Sscale = single(50); Iscale = single(50); Rscale = single(50);

wIC_I_strong = 5.0;  % strong weight on I hotspot during IC pretrain
wIC_in_full  = 0.5;  % keep some IC in full loop but much lighter than PDE+data
wBC          = 1.0;  % boundary weight
wData_SIR    = [1, 5, 2]; % S/I/R anchor weights

% Convert collocation (and IC/BC/interior) to single dlarrays (CB)
toCB = @(A) dlarray(single(A)','CB');

tInt = toCB(collocDyn(:,1));   xInt = toCB(collocDyn(:,2));   yInt = toCB(collocDyn(:,3));
tBd  = toCB(collocBd(:,1));    xBd  = toCB(collocBd(:,2));    yBd  = toCB(collocBd(:,3));
tIc  = toCB(collocIC(:,1));    xIc  = toCB(collocIC(:,2));    yIc  = toCB(collocIC(:,3));

%% ---------- Loop 1: IC pretraining ----------
for epoch = 1:numEpochsIC
    [lossIC,grads, s_ini, i_ini, r_ini] = dlfeval(@ICLoss,net,xIc,yIc,Nx,Ny,Sscale,Iscale,Rscale,wIC_I_strong);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
    if mod(epoch,50)==0
        fprintf('IC loop epoch %d | IC loss %.3e\n',epoch,extractdata(lossIC));
    end
end

%% ---------- Loop 2: Full PINN training ----------
for epoch = 1:numEpochsDyn
    [loss,grads] = dlfeval(@FullLoss,net, ...
        tInt,xInt,yInt, ...         % interior
        tBd,xBd,yBd,  ...           % boundary
        tIc,xIc,yIc,  ...           % IC
        beta,gamma,D_S,D_I,D_R, ...
        Sscale,Iscale,Rscale, ...
        wIC_in_full,wBC,wData_SIR, ...
        Xdata,Sdata,Idata,Rdata);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);

    if mod(epoch,100)==0
        fprintf('Full loop epoch %d | Total loss %.3e\n',epoch,extractdata(loss));
    end
end

% ---------- PINN Prediction over full grid ----------
[Tg,Xg,Yg] = ndgrid(t,x,y);
inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
out = predict(net,inp);
out = extractdata(out); % [3 × (Nt*Nx*Ny)]

S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
R_pred = reshape(out(3,:),[Nt,Nx,Ny]);

%% ---------- True and Predicted totals ----------
S_true_sum = squeeze(sum(S_true,[2 3]));
I_true_sum = squeeze(sum(I_true,[2 3]));
R_true_sum = squeeze(sum(R_true,[2 3]));

S_pred_sum = squeeze(sum(S_pred,[2 3]));
I_pred_sum = squeeze(sum(I_pred,[2 3]));
R_pred_sum = squeeze(sum(R_pred,[2 3]));

%% ---------- Plot totals ----------
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

%% ---------- Heatmaps ----------
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

%% ======================= Loss Functions =======================
function [lossIC,grads, S_ic, I_ic, R_ic] = ICLoss(net,xIc,yIc,Nx,Ny,Sscale,Iscale,Rscale,wIC_I)
    % Build (t=0, x, y) inputs for IC
    t0 = zeros(1,numel(xIc),'like',extractdata(xIc));
    Xic = dlarray([t0; xIc; yIc],'CB');
    Yic = forward(net,Xic);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    % Target IC fields
    S0 = 50*ones(Nx,Ny,'single'); 
    I0 = zeros(Nx,Ny,'single'); I0(round(Nx/2),round(Ny/2)) = 50;
    R0 = zeros(Nx,Ny,'single');

    S0 = dlarray(reshape(S0,1,[]),'CB');
    I0 = dlarray(reshape(I0,1,[]),'CB');
    R0 = dlarray(reshape(R0,1,[]),'CB');

    % Scaled MSEs
    lossIC = mse(S_ic./Sscale,S0./Sscale) ...
           + wIC_I*mse(I_ic./Iscale,I0./Iscale) ...
           + mse(R_ic./Rscale,R0./Rscale);

    grads = dlgradient(lossIC,net.Learnables);
end

function [loss,grads] = FullLoss(net, ...
    tInt,xInt,yInt, ...        % interior
    tBd,xBd,yBd,  ...          % boundary
    tIc,xIc,yIc,  ...          % IC
    beta,gamma,D_S,D_I,D_R, ...
    Sscale,Iscale,Rscale, ...
    wIC_in_full,wBC,wData_SIR, ...
    Xdata,Sdata,Idata,Rdata)

    %% -------- Interior PDE residuals (continuous form; NO dt here) --------
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

    S_xx = dlgradient(sum(S_x),xInt);
    S_yy = dlgradient(sum(S_y),yInt);
    I_xx = dlgradient(sum(I_x),xInt);
    I_yy = dlgradient(sum(I_y),yInt);
    R_xx = dlgradient(sum(R_x),xInt);
    R_yy = dlgradient(sum(R_y),yInt);

    fS = S_t - (D_S*(S_xx + S_yy) - beta*S.*I);
    fI = I_t - (D_I*(I_xx + I_yy) + beta*S.*I - gamma*I);
    fR = R_t - (D_R*(R_xx + R_yy) + gamma*I);

    ZS = zerosLike(fS); ZI = zerosLike(fI); ZR = zerosLike(fR);
    lossPDE = mse(fS,ZS) + mse(fI,ZI) + mse(fR,ZR);

    %% -------- Boundary condition: I(t, boundary) = 0 --------
    Ybd = forward(net,[tBd;xBd;yBd]);
    lossBC = wBC * mse(Ybd(2,:), zerosLike(Ybd(2,:)));

    %% -------- Light IC penalty in full loop --------
    % t0 = zeros(1,numel(xIc),'single','like',extractdata(xIc));
    t0 = zeros(1,numel(xIc),'like',extractdata(xIc));
    Yic = forward(net,[dlarray(t0,'CB'); xIc; yIc]);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    S0 = 50*ones(1,numel(xIc),'single'); 
    I0 = zeros(1,numel(xIc),'single'); 
    % hotspot index in flattened order:
    % (i0,j0) -> idx = sub2ind([Nx,Ny],i0,j0)
    Nx = 5;
    Ny = 5;
    i0 = round(Nx/2); j0 = round(Ny/2);
    I0(sub2ind([Nx,Ny],i0,j0)) = 50; 
    R0 = zeros(1,numel(xIc),'single');

    S0 = dlarray(S0,'CB'); I0 = dlarray(I0,'CB'); R0 = dlarray(R0,'CB');

    lossIC = wIC_in_full * ( ...
        mse(S_ic./Sscale, S0./Sscale) + ...
        mse(I_ic./Iscale, I0./Iscale) + ...
        mse(R_ic./Rscale, R0./Rscale) );

    %% -------- Sparse data anchors (S/I/R) --------
    Yd = forward(net,Xdata);
    Sd = Yd(1,:); Id = Yd(2,:); Rd = Yd(3,:);

    lossData = ...
        wData_SIR(1) * mse(Sd./Sscale, Sdata./Sscale) + ...
        wData_SIR(2) * mse(Id./Iscale, Idata./Iscale) + ...
        wData_SIR(3) * mse(Rd./Rscale, Rdata./Rscale);

    %% -------- Total --------
    % loss = lossPDE + lossBC + lossIC + lossData;
    loss = lossPDE + lossIC;
    grads = dlgradient(loss,net.Learnables);
end

%% -------- helpers --------
function Z = zerosLike(A)
    Z = zeros(size(A),'like',A);
end
