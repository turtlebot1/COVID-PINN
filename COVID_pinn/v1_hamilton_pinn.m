%% PINN for 2D Reaction–Diffusion SIR PDE with comparisons
clear; clc;
parallel.gpu.enableCUDAForwardCompatibility(true)

%% Parameters
load('params_mid.mat')
params.e = 0.002;
gpuDevice(1);

Nx = 60; Ny = 60; Nt = 20;
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Environment
env = load('env_updated_boundary.mat')
boundaryMap = arrayfun(@(c) c.boundry, env.env.env);
[xGrid, yGrid] = ndgrid(1:60, 1:60);
boundaryX = xGrid(boundaryMap);
boundaryY = yGrid(boundaryMap);

boundaryPoints = [boundaryX(:), boundaryY(:)];
% boundary check figure
% figure;
% imagesc(boundaryMap);
% axis equal tight;
% title('Boundary Map');
% hold on;
% plot(boundaryY, boundaryX, 'r.', 'MarkerSize', 10);
% Expand boundary OUTWARD by one cell
se = strel('square', 3);
expandedBoundary = imdilate(boundaryMap, se);
% Fill inside region to get the full closed interior
filledRegion = imfill(expandedBoundary, 'holes');
% Everything INSIDE the expanded boundary = collocation domain
collocationMask = filledRegion;
% Get collocation point coordinates
collocX = xGrid(collocationMask);
collocY = yGrid(collocationMask);
collocationPoints = [collocX(:), collocY(:)];

% Visualization
% figure;
% imshow(collocationMask, []);
% hold on;
% plot(yGrid(boundaryMap), xGrid(boundaryMap), 'r.', 'DisplayName','Original Boundary');
% plot(yGrid(expandedBoundary), xGrid(expandedBoundary), 'b.', 'DisplayName','Expanded Boundary');
% plot(collocY(:), collocX(:), 'g.', 'DisplayName','Collocation Points');
% legend;
% title('Collocation Points inside the Outward-Expanded Boundary');
% 
% figure; imshow(domainMask, []); hold on;
% plot(origY, origX, 'r.', 'DisplayName', 'Original Boundary');
% plot(expY, expX, 'b.', 'DisplayName', 'Expanded Boundary (+1 cell)');
% plot(intY, intX, 'g.', 'DisplayName', 'Interior Collocation');
% legend;
% title('Boundary Expansion (Red=Original, Blue=Expanded, Green=Interior)');

%% True data baseline (PDE simulation)
Nx = 60;
Ny = 60;
Nt = 45;
Time=45;
[S0, I0, R0, S_True, I_True, R_True] = fillInsideBoundary(env, Nx, Ny, Time, boundaryMap);
[boundaryPts, initPts, collocPts] = make_PINN_colloc_points(boundaryMap, Nx, Ny, Nt);
ICmask = dlarray(single(initPts), 'SS');
BCmask = dlarray(single(boundaryPts), 'SS');
IPmask = dlarray(single(collocPts), 'SS');
S0_dl = dlarray(single(S0));
I0_dl = dlarray(single(I0));
R0_dl = dlarray(single(R0));

trAvg = []; trAvgSq = [];

% % Initial conditions (t = 0)
% S0 = dlarray(single(S_array(:,:,1)), 'SS');
% I0 = dlarray(single(I_array(:,:,1)), 'SS');
% R0 = dlarray(single(R_array(:,:,1)), 'SS');
% ICmask = dlarray(single(collocationMask), 'SS'); % keep consistent with interior

%% ---- 2) Network: 3->4 (S,L,I,R) ----
layers = [
    featureInputLayer(3, "Normalization","none");
    fullyConnectedLayer(128);     
    tanhLayer();
    fullyConnectedLayer(128);     
    tanhLayer();
    fullyConnectedLayer(128);     
    tanhLayer();
    fullyConnectedLayer(4)        % [S, L, I, R]
];
net = dlnetwork(layers);
net = initialize(net);
net.Learnables.Value = arrayfun(@(x) gpuArray(x{:}), net.Learnables.Value, ...
    'UniformOutput', false);

%% separate
% S_mean = mean(S_True(:)); S_std = std(S_True(:));
% I_mean = mean(I_True(:)); I_std = std(I_True(:));
% R_mean = mean(R_True(:)); R_std = std(R_True(:));
% 
% S_True = (S_True - S_mean) / S_std;
% I_True = (I_True - I_mean) / I_std;
% R_True = (R_True - R_mean) / R_std;
% S0 = sum(S_True(:,:,1))
% I0 = sum(I_True(:,:,1))
% R0 = sum(R_True(:,:,1))

S0_dl = dlarray(single(S0));
I0_dl = dlarray(single(I0));
R0_dl = dlarray(single(R0));

%% ---- 3) Loss weights & training setup ----
w_pde  = 1.0; 
w_ic   = 1.0;
w_data = 1.0;
w_bc = 1.0;

numEpochs   = 22000;
learnRate    = 8e-4;
miniColloc   = 4096;  % collocation samples per step
miniDataT    = 3;     % number of time frames for data-fitting each step
dataFrac     = 0.02;  % fraction of spatial points per chosen frame

% Create training progress monitor
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch","LearnRate", "Loss"], ...
    XLabel="Epoch");

% Move all data/masks to the GPU
IPmask = gpuArray(IPmask);
ICmask = gpuArray(ICmask);
BCmask = gpuArray(BCmask);

S_True = gpuArray(S_True);
I_True = gpuArray(I_True);
R_True = gpuArray(R_True);

% Convert initial conditions to dlarray on the GPU
S0_dl = dlarray(gpuArray(S0));
I0_dl = dlarray(gpuArray(I0));
R0_dl = dlarray(gpuArray(R0));

%% ---- 4) Training loop ----
for epoch = 1:numEpochs
    % Collocation batch (uniform over interior & time)
    % idx = randi(numColloc, miniColloc, 1);
    % xt = single(xc(idx));
    % yt = single(yc(idx));
    % tt = single(rand(miniColloc,1));  % t in [0,1]
    % 
    % % Sample a few time frames and sparse spatial points for supervised data (S,I,R)
    % if w_data > 0
    %     tIdx = unique(randi(Time, miniDataT, 1));
    %     numDataPerT = max(1, round(dataFrac * nnz(collocationMask)));
    %     % pick random valid interior points
    %     [ixAll, iyAll] = find(collocationMask);
    %     pick = randi(numel(ixAll), numDataPerT, 1);
    %     ix = ixAll(pick); iy = iyAll(pick);
    %     xd = single((ix-1)/(Nx-1));  yd = single((iy-1)/(Ny-1));
    % else
    %     tIdx = []; xd = []; yd = [];
    % end

    % Compute loss + gradients
    [loss, parts, gradients] = dlfeval(@modelGradients, net, params, IPmask, ICmask, BCmask, S0_dl, I0_dl, R0_dl, ...
        S_True, I_True, R_True, w_pde, w_ic, w_data, w_bc);


    % Adam update
    [net, trAvg, trAvgSq] = adamupdate(net, gradients, trAvg, trAvgSq, epoch, learnRate);

    % if mod(epoch, 50) == 0
    %     fprintf('Epoch %5d | Loss %.3e | PDE %.2e | IC %.2e | Data %.2e\n', ...
    %         epoch, gather(extractdata(loss)), parts.pde, parts.ic, parts.data);
    % end
     disp("Epoch " + epoch + " | Loss " + gather(extractdata(loss)))
      % Update monitor
    recordMetrics(monitor,epoch,Loss=double(loss));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate,Loss=double(loss));
    monitor.Progress = 100*epoch/numEpochs;

    % Allow early stopping from GUI
    if monitor.Stop
        break
    end

end
