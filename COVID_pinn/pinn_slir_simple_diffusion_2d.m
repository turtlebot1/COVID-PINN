%% ===== Simple PINN for SLIR (2-D), classical diffusion, Neumann BC via stencil =====
% PDE:
%   S_t = ηS ΔS - θ S - φ I S + λ
%   L_t = ηL ΔL + φ I S - ε L
%   I_t = ηI ΔI - δ I + ε L
%   R_t = ηR ΔR - θ R + ω I
%
% Grid: 50×50 on [-1,1]^2; boundaries (first/last rows/cols) are masked (no values).
% Time: days [1,45] mapped to [0,1] in-network; ∂/∂t multiplied by 44 in residuals.
% Laplacian: 5-point stencil with Neumann BC via edge replication (zero normal derivative).
% ------------------------------------------------------------------------------------
clear; clc; close all
rng(1)

%% ----------------------- Parameters --------------------------------------------
p.etaS  = 1e-3; p.etaL = 1e-3; p.etaI = 1e-3; p.etaR = 1e-3;   % diffusion
p.theta = 0.05;   % outflow from S and R
p.phi   = 1.00;   % S*I -> L
p.eps   = 0.60;   % L -> I
p.delta = 0.40;   % removal from I
p.omega = 0.40;   % I -> R
p.lambdaS = 0.00; % inflow into S

p.tmin = 1; p.tmax = 45;     % days
timeScale = p.tmax - p.tmin; % = 44

% Loss weights
w.res  = 1.0;    % PDE residuals (interior)
w.ic   = 1.0;    % initial condition at day 1 on interior
w.cons = 1e-2;   % soft S+L+I+R ≈ 1 (optional; set 0 to disable)

%% ----------------------- Grid ---------------------------------------------------
Nx = 50; Ny = 50;
x = linspace(-1,1,Nx);
y = linspace(-1,1,Ny);
[XX,YY] = meshgrid(x,y);

ix = 2:Nx-1; iy = 2:Ny-1;        % interior indices (48×48)
hx = x(2)-x(1); hy = y(2)-y(1);  % uniform spacing

%% ----------------------- Network ------------------------------------------------
numLayers = 7; numNeurons = 32;  % small & simple
layers = featureInputLayer(3);   % [x; y; t_norm]
for i = 1:numLayers-1
    layers = [layers
              fullyConnectedLayer(numNeurons)
              tanhLayer];
end
layers = [layers
          fullyConnectedLayer(4)];   % [S; L; I; R]
net = dlnetwork(layers);

%% ----------------------- Training setup ----------------------------------------
numEpochs  = 1500;
learnRate  = 1e-3;
beta1=0.9; beta2=0.999; eps=1e-8;
avgGrad=[]; avgGradSq=[];
batchT    = 12;                    % time slices per epoch step

monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Epoch");

%% ----------------------- Initial condition (day 1) -----------------------------
% Localized latent bump; I=0, R=0, S=1-L
L0 = 0.15 * exp(-40*(XX.^2 + YY.^2));
I0 = zeros(Ny,Nx); R0 = zeros(Ny,Nx);
S0 = max(0, 1 - L0 - I0 - R0);

% keep only interior for IC loss (zeros elsewhere so boundary has no role)
S0 = keep_interior(S0, ix, iy);
L0 = keep_interior(L0, ix, iy);
I0 = keep_interior(I0, ix, iy);
R0 = keep_interior(R0, ix, iy);

%% ----------------------- Training loop -----------------------------------------
for ep = 1:numEpochs
    t_norm = rand(1,batchT);   % random normalized times in (0,1]

    [loss, grads] = dlfeval(@loss_step_simple, net, ...
        x, y, ix, iy, hx, hy, t_norm, timeScale, p, w, S0, L0, I0, R0);

    [net, avgGrad, avgGradSq] = adamupdate(net, grads, avgGrad, avgGradSq, ep, ...
                                           learnRate, beta1, beta2, eps);

    recordMetrics(monitor, ep, Loss=double(gather(extractdata(loss))));
    updateInfo(monitor, Epoch=ep);

    if mod(ep,600)==0, learnRate = learnRate*0.5; end
end

%% ----------------------- Visualization -----------------------------------------
days_to_plot = [1 15 30 45];
for d = 1:numel(days_to_plot)
    tday  = days_to_plot(d);
    tnorm = (tday - p.tmin)/timeScale;

    XYT = grid_query(x,y,tnorm);
    U = forward(net, XYT); U = extractdata(U);
    S = reshape(U(1,:,:), Ny,Nx);
    L = reshape(U(2,:,:), Ny,Nx);
    I = reshape(U(3,:,:), Ny,Nx);
    R = reshape(U(4,:,:), Ny,Nx);

    S = mask_nan(S, ix, iy); L = mask_nan(L, ix, iy);
    I = mask_nan(I, ix, iy); R = mask_nan(R, ix, iy);

    figure('Name',sprintf('Day %d',tday)); tiledlayout(2,2,'TileSpacing','compact'); colormap('turbo');
    nexttile; imagesc(x,y,S,'AlphaData',~isnan(S)); axis image; set(gca,'YDir','normal'); colorbar; title('S');
    nexttile; imagesc(x,y,L,'AlphaData',~isnan(L)); axis image; set(gca,'YDir','normal'); colorbar; title('L');
    nexttile; imagesc(x,y,I,'AlphaData',~isnan(I)); axis image; set(gca,'YDir','normal'); colorbar; title('I');
    nexttile; imagesc(x,y,R,'AlphaData',~isnan(R)); axis image; set(gca,'YDir','normal'); colorbar; title('R');
end

%% ============================== Local functions =================================
function [loss, grads] = loss_step_simple(net, x, y, ix, iy, hx, hy, t_norm, timeScale, p, w, S0, L0, I0, R0)
Nx = numel(x); Ny = numel(y);
batchT = numel(t_norm);

res_acc = dlarray(0); cons_acc = dlarray(0);

for b = 1:batchT
    tn = t_norm(b);

    % network fields on full grid at this time
    XYT = grid_query(x,y,tn);
    U = forward(net, XYT);
    S = squeeze(U(1,:,:)); L = squeeze(U(2,:,:));
    I = squeeze(U(3,:,:)); R = squeeze(U(4,:,:));

    % classical Laplacians with Neumann BC via edge replication
    Sxx_yy = laplacian_neumann(S, hx, hy);
    Lxx_yy = laplacian_neumann(L, hx, hy);
    Ixx_yy = laplacian_neumann(I, hx, hy);
    Rxx_yy = laplacian_neumann(R, hx, hy);

    % time derivatives via AD (sum-trick)
    tnVar = dlarray(tn,"CB");
    Ut = forward(net, grid_query(x,y,tnVar));
    St = dlgradient(sum(Ut(1,:,:),'all'), tnVar) * ones(size(S),'like',S);
    Lt = dlgradient(sum(Ut(2,:,:),'all'), tnVar) * ones(size(L),'like',L);
    It = dlgradient(sum(Ut(3,:,:),'all'), tnVar) * ones(size(I),'like',I);
    Rt = dlgradient(sum(Ut(4,:,:),'all'), tnVar) * ones(size(R),'like',R);

    ts = timeScale;

    % PDE residuals
    fS = ts*St - p.etaS*Sxx_yy - p.theta*S - p.phi*(I.*S) + p.lambdaS;
    fL = ts*Lt - p.etaL*Lxx_yy + p.phi*(I.*S) - p.eps*L;
    fI = ts*It - p.etaI*Ixx_yy - p.delta*I + p.eps*L;
    fR = ts*Rt - p.etaR*Rxx_yy - p.theta*R + p.omega*I;

    % interior only
    res_acc = res_acc + mean(fS(iy,ix).^2,'all') + mean(fL(iy,ix).^2,'all') + ...
                           mean(fI(iy,ix).^2,'all') + mean(fR(iy,ix).^2,'all');

    % optional conservation (interior)
    cons = (S+L+I+R) - 1;
    cons_acc = cons_acc + mean(cons(iy,ix).^2,'all');
end

% initial condition at t_norm = 0 on interior
U0 = forward(net, grid_query(x,y,dlarray(0,"CB")));
S0p = squeeze(U0(1,:,:)); L0p = squeeze(U0(2,:,:));
I0p = squeeze(U0(3,:,:)); R0p = squeeze(U0(4,:,:));

ic_acc = mean((S0p(iy,ix)-S0(iy,ix)).^2,'all') + ...
         mean((L0p(iy,ix)-L0(iy,ix)).^2,'all') + ...
         mean((I0p(iy,ix)-I0(iy,ix)).^2,'all') + ...
         mean((R0p(iy,ix)-R0(iy,ix)).^2,'all');

loss = w.res*(res_acc/batchT) + w.ic*ic_acc + w.cons*(cons_acc/batchT);
grads = dlgradient(loss, net.Learnables);
end

function XYT = grid_query(x,y,t_norm)
% 3×(Ny*Nx) dlarray for the full grid at a single time
[X,Y] = meshgrid(x,y);
T = ones(size(X),'like',X).*t_norm;
XYT = dlarray([X(:)'; Y(:)'; T(:)'], "CB");
end

function L = laplacian_neumann(U, hx, hy)
% 5-point Laplacian with Neumann (zero-flux) via edge replication.
% We build a padded array with ghost cells equal to the nearest interior value.
[Ny,Nx] = size(U);
G = zeros(Ny+2, Nx+2, 'like', U);
% interior
G(2:end-1, 2:end-1) = U;
% replicate edges (Neumann)
G(1,2:end-1)   = U(1,:);      % top ghost = first row
G(end,2:end-1) = U(end,:);    % bottom ghost = last row
G(2:end-1,1)   = U(:,1);      % left ghost = first col
G(2:end-1,end) = U(:,end);    % right ghost = last col
% corners
G(1,1)     = U(1,1);
G(1,end)   = U(1,end);
G(end,1)   = U(end,1);
G(end,end) = U(end,end);
% 5-point stencil
Ux_plus  = G(2:end-1,3:end);
Ux_minus = G(2:end-1,1:end-2);
Uy_plus  = G(3:end,  2:end-1);
Uy_minus = G(1:end-2,2:end-1);
Ucen     = G(2:end-1,2:end-1);
L = (Ux_plus - 2*Ucen + Ux_minus)/(hx^2) + (Uy_plus - 2*Ucen + Uy_minus)/(hy^2);
L = dlarray(L);
end

function A = keep_interior(A, ix, iy)
B = zeros(size(A),'like',A); B(iy,ix) = A(iy,ix); A = B;
end
function A = mask_nan(A, ix, iy)
B = nan(size(A),'like',A); B(iy,ix) = A(iy,ix); A = B;
end
