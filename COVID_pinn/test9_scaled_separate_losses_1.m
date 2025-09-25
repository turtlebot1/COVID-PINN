%% ================= PINN for 2D SIR PDE with Data Anchors =================
clear; clc; rng(0);

%% Parameters
beta  = 0.10; 
gamma = 0.10;
D_S   = 0.05; 
D_I   = 0.05; 
D_R   = 0.05;

Nx = 5; Ny = 5; Nt = 10;        % grid size + time steps
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Collocation points (t,x,y)
[X,Y,T] = ndgrid(x,y,t);
points = [T(:), X(:), Y(:)];

isBoundary = (X==0 | X==1 | Y==0 | Y==1);
isInitial  = (T==0);
isInterior = ~(isBoundary | isInitial);

collocIC  = points(isInitial(:),:);    % IC
collocDyn = points(isInterior(:),:);   % interior dynamics
collocBd  = points(isBoundary(:),:);   % boundary (not used in ODE-type loss)

%% ---------- Synthetic baseline (FD) ----------
S_true = 50*ones(Nt,Nx,Ny,'single'); 
I_true = zeros(Nt,Nx,Ny,'single'); 
R_true = zeros(Nt,Nx,Ny,'single');

I_true(1,round(Nx/2),round(Ny/2)) = 50; % hotspot
dt_fd = 0.1;

Lap = @(U) circshift(U,[1,0]) + circshift(U,[-1,0]) + ...
           circshift(U,[0,1]) + circshift(U,[0,-1]) - 4*U;

for k = 2:Nt
    S_prev = squeeze(S_true(k-1,:,:));
    I_prev = squeeze(I_true(k-1,:,:));
    R_prev = squeeze(R_true(k-1,:,:));

    S_next = S_prev + dt_fd*(D_S*Lap(S_prev) - beta*S_prev.*I_prev);
    I_next = I_prev + dt_fd*(D_I*Lap(I_prev) + beta*S_prev.*I_prev - gamma*I_prev);
    R_next = R_prev + dt_fd*(D_R*Lap(R_prev) + gamma*I_prev);

    % Dirichlet BC: I=0 on boundary
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;

    S_true(k,:,:) = S_next;
    I_true(k,:,:) = I_next;
    R_true(k,:,:) = R_next;
end

%% ---------- Data anchors ----------
tIdx = round([1, Nt/2, Nt]);
[Xg,Yg,Tg] = ndgrid(x,y,t(tIdx));

xData = single(Xg(:)); 
yData = single(Yg(:)); 
tData = single(Tg(:));

Svals = single(S_true(tIdx,:,:));  Svals = Svals(:);
Ivals = single(I_true(tIdx,:,:));  Ivals = Ivals(:);
Rvals = single(R_true(tIdx,:,:));  Rvals = Rvals(:);

Xdata = dlarray([tData'; xData'; yData'],'CB');  
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
    fullyConnectedLayer(3,"Name","out_raw")
    softplusLayer("out_pos")   % ensures outputs >= 0  % [S,I,R]
];
net = dlnetwork(layers);

%% ---------- Training setup ----------
numEpochsIC  = 500;
numEpochsDyn = 500;
learnRate = 3e-3;
avgGrad = []; avgSqGrad = [];

% Weights (from paper notation ω)
wData = 1.0;        % ω_D
wS    = 1.0;        % ω_S
wI    = 1.0;        % ω_I
wR    = 1.0;        % ω_R
wIC_S = 1.0;        % ω_{S0}
wIC_I = 5.0;        % ω_{I0}
wIC_R = 1.0;        % ω_{R0}

% Constants in ODE form (match paper notation)
C1 = 1.0; 
C2 = gamma; 

% IC targets
Ntot = 50* Nx*Ny;   % total population scale
I0   = 50; 
C    = 1.0;         % normalization factor
S0_target = (Ntot - I0)/C; 
I0_target = I0/C;  

% Convert collocation
toCB = @(A) dlarray(single(A)','CB');
tInt = toCB(collocDyn(:,1)); xInt = toCB(collocDyn(:,2)); yInt = toCB(collocDyn(:,3));
tIc  = toCB(collocIC(:,1));  xIc  = toCB(collocIC(:,2));  yIc  = toCB(collocIC(:,3));

%% ---------- Loop 1: IC pretraining ----------
for epoch = 1:numEpochsIC
    [lossIC,grads] = dlfeval(@ICLoss,net,xIc,yIc,Nx,Ny,S0_target,I0_target,wIC_S,wIC_I,wIC_R);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);
    if mod(epoch,50)==0
        fprintf('IC loop epoch %d | IC loss %.3e\n',epoch,extractdata(lossIC));
    end
end

%% ---------- Loop 2: Full PINN training ----------
for epoch = 1:numEpochsDyn
    [loss,grads] = dlfeval(@FullLoss,net, ...
        tInt,xInt,yInt, ...
        tIc,xIc,yIc, ...
        Xdata,Idata, ...
        beta,C1,C2, ...
        S0_target,I0_target, ...
        wS,wI,wR,wIC_S,wIC_I,wIC_R,wData);
    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad,epoch,learnRate);

    if mod(epoch,100)==0
        fprintf('Full loop epoch %d | Total loss %.3e\n',epoch,extractdata(loss));
    end
end

%% ---------- PINN prediction ----------
[Tg,Xg,Yg] = ndgrid(t,x,y);
inp = dlarray(single([Tg(:),Xg(:),Yg(:)])','CB');
out = predict(net,inp);
out = extractdata(out);

S_pred = reshape(out(1,:),[Nt,Nx,Ny]);
I_pred = reshape(out(2,:),[Nt,Nx,Ny]);
R_pred = reshape(out(3,:),[Nt,Nx,Ny]);

%% ---------- Totals ----------
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
function [lossIC,grads] = ICLoss(net,xIc,yIc,Nx,Ny,S0_target,I0_target,wS,wI,wR)
    t0 = zeros(1,numel(xIc),'like',xIc);
    Xic = dlarray([t0; xIc; yIc],'CB');
    Yic = forward(net,Xic);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    lossIC = wS*mean((S_ic - S0_target).^2) + ...
             wI*mean((I_ic - I0_target).^2) + ...
             wR*mean((R_ic - 0).^2);

    grads = dlgradient(lossIC,net.Learnables);
end

function [loss,grads] = FullLoss(net, ...
    tInt,xInt,yInt, ...
    tIc,xIc,yIc, ...
    Xdata,Idata, ...
    beta,C1,C2, ...
    S0_target,I0_target, ...
    wS,wI,wR,wIC_S,wIC_I,wIC_R,wData)

    %% Interior PDE residuals
    Yall = forward(net,[tInt;xInt;yInt]);
    S = Yall(1,:).^2; I = Yall(2,:).^2; R = Yall(3,:).^2;

    S_t = dlgradient(sum(S),tInt,EnableHigherDerivatives=true);
    I_t = dlgradient(sum(I),tInt,EnableHigherDerivatives=true);
    R_t = dlgradient(sum(R),tInt,EnableHigherDerivatives=true);

    fS = S_t + C1*beta*S.*I;
    fI = I_t - C1*beta*S.*I + C2*I;
    fR = R_t - C1*I;

    lossODE = mean(wS*(fS.^2) + wI*(fI.^2) + wR*(fR.^2));

    %% Initial condition loss
    t0 = zeros(1,numel(xIc),'like',xIc);
    Yic = forward(net,[t0;xIc;yIc]);
    S_ic = Yic(1,:); I_ic = Yic(2,:); R_ic = Yic(3,:);

    lossIC = wIC_S * mean((S_ic - S0_target).^2) + ...
             wIC_I * mean((I_ic - I0_target).^2) + ...
             wIC_R * mean((R_ic - 0).^2);

    %% Data loss
    Yd = forward(net,Xdata);
    Id = Yd(2,:);
    lossData = wData * mean((Id - Idata).^2);

    %% Total
    loss = lossODE + lossIC + lossData;
    grads = dlgradient(loss,net.Learnables);
end


