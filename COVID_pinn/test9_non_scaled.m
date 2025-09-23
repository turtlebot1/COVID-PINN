%% ================== SLIR PINN over 50×50 grid (Non-Scaled, Hard IC) ==================
clear; clc; close all
rng(1)

%% ------------------- Provided temporal data (S, I, R) ------------------------
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

days   = 1:numel(S_data);       % 1..45
t_norm = (days - 1) / (max(days)-1);  % [0,1] with day1 = 0

%% ------------------------ Physics parameters (your new PDE) --------------
params.DS     = 6.7e-4;    
params.DL     = 0.0816;
params.DI     = 0.0062;
params.DR     = 5.3e-8;
params.lambda = 0.68;        % recruitment/birth
params.theta  = 2.82e-5;     % natural removal
params.phi    = 4.69e-4;     % infection
params.delta  = 1.0e-5;      % removal of I
params.omega  = 0.0517;      % I -> R contribution
params.sigma  = 0.5;         % L -> I

params.tmin   = 1; params.tmax = 45;
params.lambda_data = 10.0;   % data weight
params.lambda_pde  = 1.0;    % physics weight

%% ------------------------ Build spatial grid ----------------------------
Nx = 50; Ny = 50;
xg = linspace(0,50,Nx);
yg = linspace(0,50,Ny);
[Xgrid, Ygrid] = meshgrid(xg, yg);

ix_in = 2:Nx-1; iy_in = 2:Ny-1;
[Xint,Yint] = meshgrid(xg(ix_in),yg(iy_in));
Xint_vec = Xint(:)'; Yint_vec = Yint(:)';      % 1x2304

%% ------------------------ Non-uniform Initial Conditions ----------------
% Make IC non-uniform & use SAME IC in the supervised data at day1
noiseS0 = 0.9 + 0.2*rand(size(Xint_vec));
noiseI0 = 0.9 + 0.2*rand(size(Xint_vec));
noiseR0 = 0.9 + 0.2*rand(size(Xint_vec));

S0_vec  = single(S_data(1) * noiseS0);
I0_vec  = single(I_data(1) * noiseI0);
R0_vec  = single(R_data(1) * noiseR0);

% S0_dl = dlarray(S0_vec, "CB");
% I0_dl = dlarray(I0_vec, "CB");
% R0_dl = dlarray(R0_vec, "CB");
nTimes = numel(t_norm);          % 45
nSpace = numel(Xint_vec);        % 2304
nTotal = nTimes * nSpace;        % 103,680

% Non-uniform IC at t=0
noiseS0 = 0.9 + 0.2*rand(1,nSpace);
noiseI0 = 0.9 + 0.2*rand(1,nSpace);
noiseR0 = 0.9 + 0.2*rand(1,nSpace);

S0_vec = single(S_data(1) * noiseS0);
I0_vec = single(I_data(1) * noiseI0);
R0_vec = single(R_data(1) * noiseR0);

% Replicate across all time points
S0_rep = repmat(S0_vec, 1, nTimes);
I0_rep = repmat(I0_vec, 1, nTimes);
R0_rep = repmat(R0_vec, 1, nTimes);

% dlarrays with same size as Sd_dl, Id_dl, Rd_dl
S0_dl = dlarray(S0_rep,"CB");
I0_dl = dlarray(I0_rep,"CB");
R0_dl = dlarray(R0_rep,"CB");

%% ------------------------ Build distributed data ------------------------
Xd_all = []; Yd_all = []; Td_all = [];
Sd_all = []; Id_all = []; Rd_all = [];

for k = 1:numel(t_norm)
    tk = t_norm(k);

    Xd_all = [Xd_all, Xint_vec];
    Yd_all = [Yd_all, Yint_vec];
    Td_all = [Td_all, tk*ones(size(Xint_vec))];

    if k == 1
        % Use EXACT same non-uniform IC for supervised data at day1
        Sd_all = [Sd_all, double(S0_vec)];
        Id_all = [Id_all, double(I0_vec)];
        Rd_all = [Rd_all, double(R0_vec)];
    else
        % For later days, keep non-uniform supervision (optional)
        Sk = S_data(k); Ik = I_data(k); Rk = R_data(k);
        noiseS = 0.9 + 0.2*rand(size(Xint_vec));
        noiseI = 0.9 + 0.2*rand(size(Xint_vec));
        noiseR = 0.9 + 0.2*rand(size(Xint_vec));
        Sd_all = [Sd_all, Sk*noiseS];
        Id_all = [Id_all, Ik*noiseI];
        Rd_all = [Rd_all, Rk*noiseR];
    end
end

Xd_dl = dlarray(single(Xd_all), "CB");
Yd_dl = dlarray(single(Yd_all), "CB");
Td_dl = dlarray(single(Td_all), "CB");
Sd_dl = dlarray(single(Sd_all), "CB");
Id_dl = dlarray(single(Id_all), "CB");
Rd_dl = dlarray(single(Rd_all), "CB");

%% ------------------------- Network definition ----------------------------
layers = featureInputLayer(3);  % [x,y,t_norm]
for i=1:7
    layers = [layers; fullyConnectedLayer(32); tanhLayer];
end
layers = [layers; fullyConnectedLayer(4)]; % outputs raw [S~,L~,I~,R~]
net = dlnetwork(layers);

%% ------------------------- Loss function (Hard IC) -----------------------
function [loss, gradients] = slirLoss2D_noLdata_IC(net, Xdl, Ydl, Tdl, ...
    Sd_dl, Id_dl, Rd_dl, S0_dl, I0_dl, R0_dl, params)

    XYT = cat(1,Xdl,Ydl,Tdl);
    Uraw = forward(net,XYT);           % raw NN output

    % Hard enforce IC at t=0 (day1)
    S = S0_dl + Tdl .* Uraw(1,:);
    L =          Tdl .* Uraw(2,:);
    I = I0_dl + Tdl .* Uraw(3,:);
    R = R0_dl + Tdl .* Uraw(4,:);

    % First derivatives
    gS = dlgradient(sum(S,"all"), {Xdl,Ydl,Tdl}, EnableHigherDerivatives=true);
    Sx=gS{1}; Sy=gS{2}; St=gS{3};
    gL = dlgradient(sum(L,"all"), {Xdl,Ydl,Tdl}, EnableHigherDerivatives=true);
    Lx=gL{1}; Ly=gL{2}; Lt=gL{3};
    gI = dlgradient(sum(I,"all"), {Xdl,Ydl,Tdl}, EnableHigherDerivatives=true);
    Ix=gI{1}; Iy=gI{2}; It=gI{3};
    gR = dlgradient(sum(R,"all"), {Xdl,Ydl,Tdl}, EnableHigherDerivatives=true);
    Rx=gR{1}; Ry=gR{2}; Rt=gR{3};

    % Second derivatives
    Sxx=dlgradient(sum(Sx,"all"), Xdl, EnableHigherDerivatives=true);
    Syy=dlgradient(sum(Sy,"all"), Ydl, EnableHigherDerivatives=true);
    Lxx=dlgradient(sum(Lx,"all"), Xdl, EnableHigherDerivatives=true);
    Lyy=dlgradient(sum(Ly,"all"), Ydl, EnableHigherDerivatives=true);
    Ixx=dlgradient(sum(Ix,"all"), Xdl, EnableHigherDerivatives=true);
    Iyy=dlgradient(sum(Iy,"all"), Ydl, EnableHigherDerivatives=true);
    Rxx=dlgradient(sum(Rx,"all"), Xdl, EnableHigherDerivatives=true);
    Ryy=dlgradient(sum(Ry,"all"), Ydl, EnableHigherDerivatives=true);

    % Time scaling factor to map t_norm back to days
    ts = (params.tmax - params.tmin);

    % PDE residuals: (sigma*L instead of convolution), your coefficients
    fS = ts*St - params.DS*(Sxx+Syy) + params.lambda - params.theta.*S - params.phi.*S.*I;
    fL = ts*Lt - params.DL*(Lxx+Lyy) + params.phi.*S.*I - params.sigma.*L;
    fI = ts*It - params.DI*(Ixx+Iyy) + params.sigma.*L - params.delta.*I;
    fR = ts*Rt - params.DR*(Rxx+Ryy) - params.theta.*R + params.omega.*I;

    zero = zeros(size(fS), "like", fS);
    mseF = l2loss(fS,zero)+l2loss(fL,zero)+l2loss(fI,zero)+l2loss(fR,zero);

    % Data mismatch (S,I,R only — ignore L)
    mseData = l2loss(S,Sd_dl) + l2loss(I,Id_dl) + l2loss(R,Rd_dl);

    loss = params.lambda_pde*mseF + params.lambda_data*mseData;
    gradients = dlgradient(loss, net.Learnables);
end

%% ------------------------- Training loop --------------------------------
numEpochs = 5000;
learnRate = 1e-3;
avgGrad = []; avgGradSq = [];
monitor = trainingProgressMonitor(Metrics="Loss",XLabel="Epoch");

for ep = 1:numEpochs
    [loss,gradients] = dlfeval(@slirLoss2D_noLdata_IC,net,Xd_dl,Yd_dl,Td_dl, ...
        Sd_dl,Id_dl,Rd_dl,S0_dl,I0_dl,R0_dl,params);
    [net,avgGrad,avgGradSq] = adamupdate(net,gradients,avgGrad,avgGradSq,ep,learnRate);
    recordMetrics(monitor,ep,Loss=double(gather(extractdata(loss))));
end

%% ------------------ Temporal true vs predicted curves ------------------
days_all = 1:45;
S_pred = zeros(size(days_all));
I_pred = zeros(size(days_all));
R_pred = zeros(size(days_all));

for k = 1:numel(days_all)
    t_norm_k = (days_all(k)-1)/44;
    [Xint_k,Yint_k] = meshgrid(xg(ix_in),yg(iy_in));
    XYT = [Xint_k(:)'; Yint_k(:)'; t_norm_k*ones(1,numel(Xint_k))];
    U = extractdata(forward(net,dlarray(single(XYT),"CB")));
    S_pred(k) = mean(U(1,:));
    I_pred(k) = mean(U(3,:));
    R_pred(k) = mean(U(4,:));
end

figure;
subplot(3,1,1); plot(days_all,S_data,'k--','LineWidth',2); hold on
plot(days_all,S_pred,'r-','LineWidth',2); ylabel('S'); legend('True','PINN');
subplot(3,1,2); plot(days_all,I_data,'k--','LineWidth',2); hold on
plot(days_all,I_pred,'r-','LineWidth',2); ylabel('I'); legend('True','PINN');
subplot(3,1,3); plot(days_all,R_data,'k--','LineWidth',2); hold on
plot(days_all,R_pred,'r-','LineWidth',2); ylabel('R'); xlabel('Day'); legend('True','PINN');

%% ------------------ Heatmaps: true vs predicted ------------------------
daysToPlot = [1,15,30,45];
for d = daysToPlot
    t_norm_k = (d-1)/44;
    [Xint_k,Yint_k] = meshgrid(xg(ix_in),yg(iy_in));
    XYT = [Xint_k(:)'; Yint_k(:)'; t_norm_k*ones(1,numel(Xint_k))];
    U = extractdata(forward(net,dlarray(single(XYT),"CB")));

    Smap_pred = nan(Ny,Nx); Imap_pred = nan(Ny,Nx); Rmap_pred = nan(Ny,Nx);
    Smap_pred(iy_in,ix_in) = reshape(U(1,:),numel(iy_in),numel(ix_in));
    Imap_pred(iy_in,ix_in) = reshape(U(3,:),numel(iy_in),numel(ix_in));
    Rmap_pred(iy_in,ix_in) = reshape(U(4,:),numel(iy_in),numel(ix_in));

    % "True" maps: uniform slabs per your current practice
    Smap_true = nan(Ny,Nx); Imap_true = nan(Ny,Nx); Rmap_true = nan(Ny,Nx);
    Smap_true(iy_in,ix_in) = S_data(d);
    Imap_true(iy_in,ix_in) = I_data(d);
    Rmap_true(iy_in,ix_in) = R_data(d);

    figure('Name',sprintf('Day %d',d)); tiledlayout(3,2,'TileSpacing','compact'); colormap('turbo');
    nexttile; imagesc(xg,yg,Smap_true,'AlphaData',~isnan(Smap_true)); axis image; colorbar; title('S True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Smap_pred,'AlphaData',~isnan(Smap_pred)); axis image; colorbar; title('S PINN'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Imap_true,'AlphaData',~isnan(Imap_true)); axis image; colorbar; title('I True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Imap_pred,'AlphaData',~isnan(Imap_pred)); axis image; colorbar; title('I PINN'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Rmap_true,'AlphaData',~isnan(Rmap_true)); axis image; colorbar; title('R True'); set(gca,'YDir','normal');
    nexttile; imagesc(xg,yg,Rmap_pred,'AlphaData',~isnan(Rmap_pred)); axis image; colorbar; title('R PINN'); set(gca,'YDir','normal');
end
