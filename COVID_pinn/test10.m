% %% ================== SLIR PINN over 50×50 grid (Non-Scaled, Hard IC) ==================
% clear; clc; close all
% rng(1)
% 
% %% ------------------- Provided temporal data (S, I, R) ------------------------
% S_data = [1133506.1,1133238.1,1132784.1,1132363.1,1131902.1,1131453.1,...
%           1130944.1,1130585.1,1130219.1,1129600.1,1129022.1,1128384.1,...
%           1127811.1,1127220.1,1126773.1,1126366.1,1125672.1,1125055.1,...
%           1124469.1,1123874.1,1123356.1,1122912.1,1122457.1,1121758.1,...
%           1121097.1,1120418.1,1120173.1,1119553.1,1119063.1,1118561.1,...
%           1117712.1,1116949.1,1116297.1,1115622.1,1114949.1,1114404.1,...
%           1113954.1,1113123.1,1112412.1,1111752.1,1111131.1,1110575.1,...
%           1110107.1,1109719.1,1109041.1];
% 
% I_data = [2725.462016,2738.692403,2968.938481,3211.787696,3597.357539,...
%           3954.831508,4331.158302,4517.030060,4811.965692,5364.145074,...
%           5925.459402,6549.137958,6935.876807,7294.945205,7387.303010,...
%           7437.395396,7758.752438,8152.126439,8458.252734,8554.534487,...
%           8523.939376,8359.077509,8383.914056,8708.426648,9220.442922,...
%           9830.172268,9804.632503,10055.025349,10227.002882,10588.772987,...
%           11432.603097,12328.458612,12985.960478,13402.650777,13614.175818,...
%           13578.297349,13933.353467,14545.050414,15073.407777,15408.663166,...
%           15289.450156,14980.323141,14749.496246,14664.378627,15077.335071];
% 
% R_data = [1625.307597,1845.061519,2016.212304,2089.642461,2176.168492,...
%           2303.841698,2474.969940,2545.034308,2602.854926,2614.540598,...
%           2621.862042,2805.123193,3032.054795,3382.696990,3735.604604,...
%           4103.247562,4319.873561,4596.747266,5087.465513,5629.060624,...
%           6229.922491,6657.085944,7022.573352,7161.557078,7223.827732,...
%           7489.367497,7849.974651,8162.997118,8300.227013,8294.396903,...
%           8160.541388,8148.039522,8400.349223,8854.824182,9424.702651,...
%           9512.646533,9724.949586,9894.592223,10212.336834,10944.549844,...
%           11799.676859,12489.503754,12953.621373,13203.664929,13213.770865];
% 
% days = 1:numel(S_data); 
% T    = numel(days);
% 
% %% ------------------- Grid (interior 48×48 = 2304 cells) ----------------
% Nx = 50; Ny = 50;
% xg = linspace(0,50,Nx); yg = linspace(0,50,Ny);
% [ix_in,iy_in] = deal(2:Nx-1, 2:Ny-1);
% [Xint, Yint]  = meshgrid(xg(ix_in), yg(iy_in));
% nCells = numel(Xint);  % 2304
% 
% %% ------------------- Expand scalar SIR into spatial fields -------------
% % Replace with real spatial fields if you have them
% S_fields = zeros(nCells,T); I_fields = zeros(nCells,T); R_fields = zeros(nCells,T);
% for k=1:T
%     noiseS = 0.95+0.10*rand(nCells,1); noiseS = noiseS/mean(noiseS);
%     noiseI = 0.95+0.10*rand(nCells,1); noiseI = noiseI/mean(noiseI);
%     noiseR = 0.95+0.10*rand(nCells,1); noiseR = noiseR/mean(noiseR);
%     S_fields(:,k) = S_data(k)*noiseS;
%     I_fields(:,k) = I_data(k)*noiseI;
%     R_fields(:,k) = R_data(k)*noiseR;
% end
% 
% %% ------------------- Normalization stats -------------------------------
% muX = mean([S_fields(:) I_fields(:) R_fields(:)],1)';
% sdX = std([S_fields(:) I_fields(:) R_fields(:)],0,1)' + 1e-8;
% 
% %% ------------------- Physics parameters --------------------------------
% params.DS=6.7e-4; params.DL=0.0816; params.DI=0.0062; params.DR=5.3e-8;
% params.lambda=0.68; params.theta=2.82e-5; params.phi=4.69e-4;
% params.delta=1.0e-5; params.omega=0.0517; params.sigma=0.5;
% params.dt=1;  % one day step
% params.lambda_data=1.0; params.lambda_phys=0.1;
% 
% %% ------------------- Define LSTM (dlNetwork) ---------------------------
% layers = [ ...
%     sequenceInputLayer(3,"Name","in")
%     lstmLayer(128,"OutputMode","sequence","Name","lstm1")
%     dropoutLayer(0.1,"Name","drop1")
%     lstmLayer(128,"OutputMode","sequence","Name","lstm2")
%     fullyConnectedLayer(3,"Name","fc") ];
% 
% net = dlnetwork(layerGraph(layers));
% 
% %% ------------------- Helper: physics residual function -----------------
% function f = physicsResidual(S,I,R,params)
%     % No diffusion here; add neighbor coupling if needed
%     fS = params.lambda - params.theta.*S - params.phi.*S.*I;
%     fI = params.phi.*S.*I - params.delta.*I - params.sigma.*I;  % sigma term approx
%     fR = -params.theta.*R + params.omega.*I;
%     f = [fS; fI; fR];  % 3×nCells
% end
% 
% %% ------------------- Loss function (data + physics) --------------------
% function [loss,gradients] = modelLoss(net,Xseq,Ytrue,muX,sdX,params)
%     % Xseq: input seq [3×1×batch], Ytrue: true next step [3×batch]
%     Ypred_norm = forward(net,Xseq);  % 3×1×batch
%     Ypred_norm = squeeze(Ypred_norm); % 3×batch
%     Ypred = Ypred_norm .* sdX + muX;  % un-normalize
% 
%     % Data loss
%     mseData = mean((Ypred - Ytrue).^2,"all");
% 
%     % Physics loss (residuals)
%     f = physicsResidual(Ypred(1,:),Ypred(2,:),Ypred(3,:),params);
%     msePhys = mean(f.^2,"all");
% 
%     loss = params.lambda_data*mseData + params.lambda_phys*msePhys;
%     gradients = dlgradient(loss,net.Learnables);
% end
% 
% %% ------------------- Training loop -------------------------------------
% numEpochs=2000; learnRate=1e-3; batchSize=256;
% avgGrad=[]; avgGradSq=[];
% 
% for epoch=1:numEpochs
%     % Create random mini-batch of cells and timesteps
%     idxCells = randperm(nCells,batchSize);
%     t = randi([1,T-1],1);  % pick random day
% 
%     % Inputs (previous day, normalized)
%     X = [S_fields(idxCells,t)'; I_fields(idxCells,t)'; R_fields(idxCells,t)'];
%     Xnorm = (X - muX) ./ sdX;
%     Xseq = dlarray(reshape(Xnorm,[3 1 batchSize]),"CBT");
% 
%     % Targets (next day, true un-normalized)
%     Ytrue = [S_fields(idxCells,t+1)'; I_fields(idxCells,t+1)'; R_fields(idxCells,t+1)'];
% 
%     [loss,gradients] = dlfeval(@modelLoss,net,Xseq,Ytrue,muX,sdX,params);
%     [net,avgGrad,avgGradSq] = adamupdate(net,gradients,avgGrad,avgGradSq,epoch,learnRate);
% 
%     if mod(epoch,10)==0
%         fprintf("Epoch %d, Loss %.4e\n",epoch,extractdata(loss));
%     end
% end
% 
% %% ------------------- Rollout (autoregressive from Day 1) ----------------
% S_pred=zeros(nCells,T); I_pred=zeros(nCells,T); R_pred=zeros(nCells,T);
% S_pred(:,1)=S_fields(:,1); I_pred(:,1)=I_fields(:,1); R_pred(:,1)=R_fields(:,1);
% 
% for t=2:T
%     X = [S_pred(:,t-1)'; I_pred(:,t-1)'; R_pred(:,t-1)'];
%     Xnorm = (X - muX) ./ sdX;
%     Xseq = dlarray(reshape(Xnorm,[3 1 nCells]),"CBT");
%     Ypred_norm = predict(net,Xseq);
%     Ypred_norm = squeeze(Ypred_norm);
%     Ypred = Ypred_norm .* sdX + muX;
% 
%     Ypred = extractdata(Ypred);   % convert dlarray → numeric
% 
%     S_pred(:,t) = Ypred(1,:)';
%     I_pred(:,t) = Ypred(2,:)';
%     R_pred(:,t) = Ypred(3,:)';
% end
% 
% %% ------------------- Plot mean curves ----------------------------------
% S_mean=mean(S_pred,1); I_mean=mean(I_pred,1); R_mean=mean(R_pred,1);
% figure;
% subplot(3,1,1); plot(days,S_data,'k--','LineWidth',2); hold on; plot(days,S_mean,'r-','LineWidth',2); ylabel('S'); legend('True','PI-LSTM');
% subplot(3,1,2); plot(days,I_data,'k--','LineWidth',2); hold on; plot(days,I_mean,'r-','LineWidth',2); ylabel('I'); legend('True','PI-LSTM');
% subplot(3,1,3); plot(days,R_data,'k--','LineWidth',2); hold on; plot(days,R_mean,'r-','LineWidth',2); ylabel('R'); xlabel('Day'); legend('True','PI-LSTM');

%% ================== SLIR PINN (env-conditioned, hard IC, balanced loss) ==================
clear; clc; close all
rng(1)

%% ------------------- Temporal SIR scalars (45 days) -------------------
S_data = [1133506.1,1133238.1,1132784.1,1132363.1,1131902.1,1131453.1,...
          1130944.1,1130585.1,1130219.1,1129600.1,1129022.1,1128384.1,...
          1127811.1,1127220.1,1126773.1,1126366.1,1125672.1,1125055.1,...
          1124469.1,1123874.1,1123356.1,1122912.1,1122457.1,1121758.1,...
          1121097.1,1120418.1,1120173.1,1119553.1,1119063.1,1118561.1,...
          1117712.1,1116949.1,1116297.1,1115622.1,1114949.1,1114404.1,...
          1113954.1,1113123.1,1112412.1,1111752.1,1111131.1,1110575.1,...
          1110107.1,1109719.1,1109041.1];

I_data = [2725.462016,2738.692403,2968.938481,3211.787696,3597.357539,...
          3954.831508,4331.158302,4517.030060,4811.965692,5364.145074,...
          5925.459402,6549.137958,6935.876807,7294.945205,7387.303010,...
          7437.395396,7758.752438,8152.126439,8458.252734,8554.534487,...
          8523.939376,8359.077509,8383.914056,8708.426648,9220.442922,...
          9830.172268,9804.632503,10055.025349,10227.002882,10588.772987,...
          11432.603097,12328.458612,12985.960478,13402.650777,13614.175818,...
          13578.297349,13933.353467,14545.050414,15073.407777,15408.663166,...
          15289.450156,14980.323141,14749.496246,14664.378627,15077.335071];

R_data = [1625.307597,1845.061519,2016.212304,2089.642461,2176.168492,...
          2303.841698,2474.969940,2545.034308,2602.854926,2614.540598,...
          2621.862042,2805.123193,3032.054795,3382.696990,3735.604604,...
          4103.247562,4319.873561,4596.747266,5087.465513,5629.060624,...
          6229.922491,6657.085944,7022.573352,7161.557078,7223.827732,...
          7489.367497,7849.974651,8162.997118,8300.227013,8294.396903,...
          8160.541388,8148.039522,8400.349223,8854.824182,9424.702651,...
          9512.646533,9724.949586,9894.592223,10212.336834,10944.549844,...
          11799.676859,12489.503754,12953.621373,13203.664929,13213.770865];

days   = 1:numel(S_data);
t_norm = (days - 1) / (max(days)-1);  % [0,1], day1=0
T      = numel(days);

%% ------------------- Physics params (your system with sigma*L) --------
params.DS=6.7e-4;  params.DL=0.0816;   params.DI=0.0062;   params.DR=5.3e-8;
params.lambda=0.68; params.theta=2.82e-5; params.phi=4.69e-4;
params.delta=1.0e-5; params.omega=0.0517; params.sigma=0.5;

params.tmin=1; params.tmax=45;                      % time scaling for dt/dτ
params.lambda_data=1.0;                              % data loss weight
params.lambda_pde=0.0;                               % will ramp up during training
params.lambda_ic = 100.0;                            % enforce t=0 strongly
params.rel_w = [1/S_data(1), 1/I_data(1), 1/R_data(1)]; % relative scaling per comp

%% ------------------- Grid and interior points -------------------------
Nx=50; Ny=50;
xg=linspace(0,50,Nx); yg=linspace(0,50,Ny);
ix_in=2:Nx-1; iy_in=2:Ny-1;
[Xint,Yint]=meshgrid(xg(ix_in),yg(iy_in));
Xint_vec=Xint(:)'; Yint_vec=Yint(:)';              % 1×2304
nSpace=numel(Xint_vec); nTimes=T; nTotal=nSpace*nTimes;

%% ------------------- Non-uniform initial conditions (day1) ------------
noiseS0 = 0.9 + 0.2*rand(1,nSpace); noiseS0=noiseS0/mean(noiseS0);
noiseI0 = 0.9 + 0.2*rand(1,nSpace); noiseI0=noiseI0/mean(noiseI0);
noiseR0 = 0.9 + 0.2*rand(1,nSpace); noiseR0=noiseR0/mean(noiseR0);

S0_vec = single(S_data(1)*noiseS0);
I0_vec = single(I_data(1)*noiseI0);
R0_vec = single(R_data(1)*noiseR0);

% Replicate IC fields across all times to match batch shapes
S0_rep=repmat(S0_vec,1,nTimes);
I0_rep=repmat(I0_vec,1,nTimes);
R0_rep=repmat(R0_vec,1,nTimes);

%% ------------------- Build supervised space–time data ------------------
Xd_all=[]; Yd_all=[]; Td_all=[];
S0_all=[]; I0_all=[]; R0_all=[];
Sd_all=[]; Id_all=[]; Rd_all=[];

for k=1:nTimes
    tk=t_norm(k);
    Xd_all=[Xd_all, Xint_vec];
    Yd_all=[Yd_all, Yint_vec];
    Td_all=[Td_all, tk*ones(size(Xint_vec))];

    S0_all=[S0_all, S0_vec]; I0_all=[I0_all, I0_vec]; R0_all=[R0_all, R0_vec];

    if k==1
        % Use EXACT same IC field for day1 supervision to avoid contradictions
        Sd_all=[Sd_all, double(S0_vec)];
        Id_all=[Id_all, double(I0_vec)];
        Rd_all=[Rd_all, double(R0_vec)];
    else
        % non-uniform slabs per day (replace with real per-cell data if you have it)
        nS=0.95+0.10*rand(size(Xint_vec)); nS=nS/mean(nS);
        nI=0.95+0.10*rand(size(Xint_vec)); nI=nI/mean(nI);
        nR=0.95+0.10*rand(size(Xint_vec)); nR=nR/mean(nR);
        Sd_all=[Sd_all, S_data(k)*nS];
        Id_all=[Id_all, I_data(k)*nI];
        Rd_all=[Rd_all, R_data(k)*nR];
    end
end

% dlarrays
Xd_dl = dlarray(single(Xd_all),"CB");
Yd_dl = dlarray(single(Yd_all),"CB");
Td_dl = dlarray(single(Td_all),"CB");
S0_dl = dlarray(single(S0_all),"CB");
I0_dl = dlarray(single(I0_all),"CB");
R0_dl = dlarray(single(R0_all),"CB");
Sd_dl = dlarray(single(Sd_all),"CB");
Id_dl = dlarray(single(Id_all),"CB");
Rd_dl = dlarray(single(Rd_all),"CB");

%% ------------------- Network: inputs [x,y,t,S0,I0,R0] -----------------
layers = featureInputLayer(6);
for i=1:6, layers=[layers; fullyConnectedLayer(64); tanhLayer]; end
layers=[layers; fullyConnectedLayer(4)];  % outputs raw [S~,L~,I~,R~]
net = dlnetwork(layers);

%% ------------------- Mini-batch sampler --------------------------------
batchSize = 8192;   % 8k points per step (fast + stable)
idxAll = 1:nTotal;  % total columns

%% ------------------- Training with PDE ramp ----------------------------
numIters = 12000;           % ~epochs when sampling
learnRate = 1e-3;
avgGrad=[]; avgGradSq=[];
monitor = trainingProgressMonitor(Metrics=["Loss","mseData","msePhys","mseIC"],XLabel="Iter");

for it=1:numIters
    % --- sample a mini-batch of spatio-temporal points
    idx = randsample(idxAll, batchSize);
    Xb = Xd_dl(:,idx); Yb=Yd_dl(:,idx); Tb=Td_dl(:,idx);
    S0b=S0_dl(:,idx);  I0b=I0_dl(:,idx); R0b=R0_dl(:,idx);
    Sdb=Sd_dl(:,idx);  Idb=Id_dl(:,idx); Rdb=Rd_dl(:,idx);

    % --- ramp physics weight: 0 -> 1 over first 4k iters
    ramp = min(1.0, it/4000);
    params.lambda_pde = 1.0 * ramp;

    % --- step
    [loss,grads,metrics] = dlfeval(@pinnLoss_envIC, net, ...
        Xb,Yb,Tb,S0b,I0b,R0b,Sdb,Idb,Rdb, params);
    % [net,avgGrad,avgGradSq] = adamupdate(net,grads,avgGrad,avgGradSq,it,learnRate,GradientThreshold=5.0);
    [net,avgGrad,avgGradSq] = adamupdate(net,grads,avgGrad,avgGradSq,it,learnRate);


    recordMetrics(monitor,it, ...
        Loss=double(gather(extractdata(loss))), ...
        mseData=double(gather(extractdata(metrics.mseData))), ...
        msePhys=double(gather(extractdata(metrics.msePhys))), ...
        mseIC=double(gather(extractdata(metrics.mseIC))));
end

%% ------------------- Predict temporal curves (means) -------------------
days_all=1:T;
S_pred=zeros(1,T); I_pred=zeros(1,T); R_pred=zeros(1,T);

for k=1:T
    t_norm_k=(days_all(k)-1)/(T-1);
    [Xk,Yk]=meshgrid(xg(ix_in),yg(iy_in));
    npts=numel(Xk);
    XYTIC=[Xk(:)'; Yk(:)'; t_norm_k*ones(1,npts); S0_vec; I0_vec; R0_vec];
    U = extractdata(forward(net,dlarray(single(XYTIC),"CB")));
    S_pred(k)=mean(U(1,:));
    I_pred(k)=mean(U(3,:));
    R_pred(k)=mean(U(4,:));
end

figure;
subplot(3,1,1); plot(days,S_data,'k--','LineWidth',2); hold on
plot(days,S_pred,'r-','LineWidth',2); ylabel('S'); legend('True','PINN'); title('Env-conditioned PINN (hard IC, balanced loss, PDE ramp)');
subplot(3,1,2); plot(days,I_data,'k--','LineWidth',2); hold on
plot(days,I_pred,'r-','LineWidth',2); ylabel('I'); legend('True','PINN');
subplot(3,1,3); plot(days,R_data,'k--','LineWidth',2); hold on
plot(days,R_pred,'r-','LineWidth',2); ylabel('R'); xlabel('Day'); legend('True','PINN');

%% ------------------- Heatmaps (pred vs "true" slabs) -------------------
Ny=50; Nx=50; daysToPlot=[1,15,30,45];
for d=daysToPlot
    t_norm_k=(d-1)/(T-1);
    [Xk,Yk]=meshgrid(xg(ix_in),yg(iy_in)); npts=numel(Xk);
    XYTIC=[Xk(:)'; Yk(:)'; t_norm_k*ones(1,npts); S0_vec; I0_vec; R0_vec];
    U=extractdata(forward(net,dlarray(single(XYTIC),"CB")));
    Smap_pred=nan(Ny,Nx); Imap_pred=nan(Ny,Nx); Rmap_pred=nan(Ny,Nx);
    Smap_pred(iy_in,ix_in)=reshape(U(1,:),numel(iy_in),numel(ix_in));
    Imap_pred(iy_in,ix_in)=reshape(U(3,:),numel(iy_in),numel(ix_in));
    Rmap_pred(iy_in,ix_in)=reshape(U(4,:),numel(iy_in),numel(ix_in));
    Smap_true=nan(Ny,Nx); Imap_true=nan(Ny,Nx); Rmap_true=nan(Ny,Nx);
    Smap_true(iy_in,ix_in)=S_data(d); Imap_true(iy_in,ix_in)=I_data(d); Rmap_true(iy_in,ix_in)=R_data(d);

    figure('Name',sprintf('Day %d',d)); tiledlayout(3,2,'TileSpacing','compact'); colormap('turbo');
    nexttile; imagesc(xg,yg,Smap_true,'AlphaData',~isnan(Smap_true)); axis image; colorbar; title('S True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Smap_pred,'AlphaData',~isnan(Smap_pred)); axis image; colorbar; title('S PINN'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Imap_true,'AlphaData',~isnan(Imap_true)); axis image; colorbar; title('I True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Imap_pred,'AlphaData',~isnan(Imap_pred)); axis image; colorbar; title('I PINN'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Rmap_true,'AlphaData',~isnan(Rmap_true)); axis image; colorbar; title('R True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Rmap_pred,'AlphaData',~isnan(Rmap_pred)); axis image; colorbar; title('R PINN'); set(gca,'YDir','normal');
end

%% ============================ Local functions ============================
function [loss,gradients,metrics] = pinnLoss_envIC(net, Xdl,Ydl,Tdl, ...
    S0dl,I0dl,R0dl,Sddl,Iddl,Rddl, params)

    % Env-conditioned inputs + hard-IC embedding in outputs
    XYTIC = cat(1,Xdl,Ydl,Tdl,S0dl,Iddl*0+I0dl,R0dl); % keep shapes
    Uraw  = forward(net, XYTIC);     % raw deviations
    % Enforce S,I,R exactly equal to IC at t=0 via t-multiplier
    S = S0dl + Tdl .* Uraw(1,:);     % hard IC for S
    L =          Tdl .* Uraw(2,:);   % latent free (no data term)
    I = I0dl + Tdl .* Uraw(3,:);     % hard IC for I
    R = R0dl + Tdl .* Uraw(4,:);     % hard IC for R

    % ---------- derivatives ----------
    gS = dlgradient(sum(S,"all"),{Xdl,Ydl,Tdl},EnableHigherDerivatives=true);
    Sx=gS{1}; Sy=gS{2}; St=gS{3};
    gL = dlgradient(sum(L,"all"),{Xdl,Ydl,Tdl},EnableHigherDerivatives=true);
    Lx=gL{1}; Ly=gL{2}; Lt=gL{3};
    gI = dlgradient(sum(I,"all"),{Xdl,Ydl,Tdl},EnableHigherDerivatives=true);
    Ix=gI{1}; Iy=gI{2}; It=gI{3};
    gR = dlgradient(sum(R,"all"),{Xdl,Ydl,Tdl},EnableHigherDerivatives=true);
    Rx=gR{1}; Ry=gR{2}; Rt=gR{3};

    Sxx=dlgradient(sum(Sx,"all"),Xdl,EnableHigherDerivatives=true);
    Syy=dlgradient(sum(Sy,"all"),Ydl,EnableHigherDerivatives=true);
    Lxx=dlgradient(sum(Lx,"all"),Xdl,EnableHigherDerivatives=true);
    Lyy=dlgradient(sum(Ly,"all"),Ydl,EnableHigherDerivatives=true);
    Ixx=dlgradient(sum(Ix,"all"),Xdl,EnableHigherDerivatives=true);
    Iyy=dlgradient(sum(Iy,"all"),Ydl,EnableHigherDerivatives=true);
    Rxx=dlgradient(sum(Rx,"all"),Xdl,EnableHigherDerivatives=true);
    Ryy=dlgradient(sum(Ry,"all"),Ydl,EnableHigherDerivatives=true);

    ts = (params.tmax - params.tmin);  % scale dt/dτ from t_norm to days

    % ---------- PDE residuals (sigma*L instead of convolution) ----------
    fS = ts*St - params.DS*(Sxx+Syy) + params.lambda - params.theta.*S - params.phi.*S.*I;
    fL = ts*Lt - params.DL*(Lxx+Lyy) + params.phi.*S.*I - params.sigma.*L;
    fI = ts*It - params.DI*(Ixx+Iyy) + params.sigma.*L - params.delta.*I;
    fR = ts*Rt - params.DR*(Rxx+Ryy) - params.theta.*R + params.omega.*I;

    % ---------- mean MSEs (no implicit huge sums) ----------
    msePhys = mean(fS.^2,'all') + mean(fL.^2,'all') + mean(fI.^2,'all') + mean(fR.^2,'all');

    % Relative-error data loss (balance compartments)
    rw = params.rel_w;  % [wS,wI,wR]
    mseS = mean(((S - Sddl).*rw(1)).^2,'all');
    mseI = mean(((I - Iddl).*rw(2)).^2,'all');
    mseR = mean(((R - Rddl).*rw(3)).^2,'all');
    mseData = mseS + mseI + mseR;

    % Strong IC enforcement ONLY on t=0 samples
    is_t0 = (Tdl==0);
    if any(gather(extractdata(is_t0)))
        mseIC = mean(((S(is_t0)-S0dl(is_t0)).*rw(1)).^2,'all') + ...
                mean(((I(is_t0)-I0dl(is_t0)).*rw(2)).^2,'all') + ...
                mean(((R(is_t0)-R0dl(is_t0)).*rw(3)).^2,'all');
    else
        mseIC = dlarray(0,'CB');
    end

    loss = params.lambda_pde*msePhys + params.lambda_data*mseData + params.lambda_ic*mseIC;

    gradients = dlgradient(loss, net.Learnables);

    metrics.mseData = mseData;
    metrics.msePhys = msePhys;
    metrics.mseIC   = mseIC;
end
