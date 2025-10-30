%% PINN for 2D Reaction–Diffusion SIR PDE with comparisons
clear; clc;

%% Parameters
beta = 0.1; gamma = 0.1;
D_S = 0.05; D_I = 0.05; D_R = 0.05;

Nx = 5; Ny = 5; Nt = 10;
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
numEpochs = 3000;
learnRate = 8e-3;
avgGrad = []; avgSqGrad = [];
% tInt = dlarray(collocInt(:,1)','CB');
% xInt = dlarray(collocInt(:,2)','CB');
% yInt = dlarray(collocInt(:,3)','CB');

% Create training progress monitor
monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch","LearnRate", "Loss"], ...
    XLabel="Epoch");

%% Finite-difference baseline (true simulation)
S_true = 20*ones(Nt,Nx,Ny); 
% enforce zero Dirichlet BCs for all SIR compartments
S_true(:,1,:)=0; S_true(:,end,:)=0; S_true(:,:,1)=0; S_true(:,:,end)=0;
I_true = zeros(Nt,Nx,Ny); 
R_true = zeros(Nt,Nx,Ny);

I_true(1,round(Nx/2),round(Ny/2)) = 20; % infection bump at center
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
    % enforce zero Dirichlet BCs for all SIR compartments
    S_next(1,:)=0; S_next(end,:)=0; S_next(:,1)=0; S_next(:,end)=0;
    I_next(1,:)=0; I_next(end,:)=0; I_next(:,1)=0; I_next(:,end)=0;
    R_next(1,:)=0; R_next(end,:)=0; R_next(:,1)=0; R_next(:,end)=0;

    S_true(k,:,:) = S_next;
    I_true(k,:,:) = I_next;
    R_true(k,:,:) = R_next;
end


%% Data prep for data loss
% Pick 3 anchor slices (early, mid, late)
tIdx = round([1, Nt/2, Nt]);   % e.g., [1, 5, 10] if Nt=10
[Xg, Yg, Tg] = ndgrid(1:Nx, 1:Ny, tIdx);   % full spatial grid at those times
Nd = numel(Xg);                            % number of anchor points
% Flatten to [Nx*Ny*Nt, 1]
S_flat = reshape(permute(S_true, [2 3 1]), [], Nt);  % (Nx*Ny) × Nt
I_flat = reshape(permute(I_true, [2 3 1]), [], Nt);
R_flat = reshape(permute(R_true, [2 3 1]), [], Nt);
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

%% Training loop
for epoch = 1:numEpochs
    disp(epoch);
    % Wrap collocation points as dlarray before dlfeval
    tInt = dlarray(collocInt(:,1)','CB');
    xInt = dlarray(collocInt(:,2)','CB');
    yInt = dlarray(collocInt(:,3)','CB');

    % If normal loss
    % [loss,grads, Ybd, Yic, S0] = dlfeval(@modelLoss,net,tInt,xInt,yInt, ...
    %                        collocBd,collocIC, ...
    %                        beta,gamma,D_S,D_I,D_R);
    % If data loss
    beta = 0.1; gamma = 0.1;
    D_S = 0.05; D_I = 0.05; D_R = 0.05;
    [loss, grads, Ybd, Yic, S0, I0, R0, fS, fI, fR, test1, test2, test3, test4] = dlfeval(@modelLoss, net, tInt, xInt, yInt, ...
                             collocBd, collocIC, collocInt, ...
                             beta, gamma, D_S, D_I, D_R, ...
                             Xdata, Sdata, Idata, Rdata, epoch);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grads,avgGrad,avgSqGrad, ...
    epoch,learnRate);

    % Update monitor
    recordMetrics(monitor,epoch,Loss=double(loss));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    monitor.Progress = 100*epoch/numEpochs;

    % Allow early stopping from GUI
    if monitor.Stop
        break
    end
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
% 
% %% Plot curves
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
% 
% %% Heatmap comparisons
% selDays = [1, round(Nt/2), Nt];
% for d = selDays
%     figure;
%     subplot(1,2,1)
%     imagesc(x,y,squeeze(I_true(d,:,:))); axis equal tight; colorbar;
%     title(['True I at day ',num2str(d)])
% 
%     subplot(1,2,2)
%     imagesc(x,y,squeeze(I_pred(d,:,:))); axis equal tight; colorbar;
%     title(['PINN Predicted I at day ',num2str(d)])
% end

%% Final printing and additional data
Ybd = extractdata(Ybd);
Yic = extractdata(Yic);
test1 = extractdata(test1);
test2 = extractdata(test2);
test3 = extractdata(test3);
test4 = extractdata(test4);
%% ---------------- Loss function ----------------

%% calculate gradients by summing over all points
% function [loss,grads, Ybd, Yic, S0] = modelLoss(net,t,x,y,collocBd,collocIC, ...
%                                   beta,gamma,D_S,D_I,D_R)
% 
%     % Pinterior collocation points into one dlarray [t,x,y]
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
%     fS = S_t - (D_S*(S_xx+S_yy) - beta*S.*I);
%     fI = I_t - (D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
%     fR = R_t - (D_R*(R_xx+R_yy) + gamma*I);
%     % fI
% 
%     lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
%         + mse(fI, zeros(size(fI),'like',fI)) ...
%         + mse(fR, zeros(size(fR),'like',fR));
% 
%     % --- Boundary condition: I=0 ---
%     Xbd = dlarray(single(collocBd)','CB');
%     Ybd = forward(net,Xbd);
%     % lossBC = mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd));
%     lossBC = mse(Ybd(1,:), zeros(size(Ybd(1,:)),'like',Ybd)) + ...   % S = 0
%          mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd)) + ...   % I = 0
%          mse(Ybd(3,:), zeros(size(Ybd(3,:)),'like',Ybd));        % R = 0
% 
%     % --- Initial condition ---
%     Xic = dlarray(single(collocIC)','CB');
%     Yic = forward(net,Xic);
%     S0 = 50 * ones(size(Yic(1,:)),'like',Yic);   % Susceptibles in thousands
%     Nx = 5; Ny = 5;
% 
%     % Zero out the boundaries (as before)
%     x_coords = Xic(2,:);   % x ∈ [0,1]
%     y_coords = Xic(3,:);   % y ∈ [0,1]
% 
%     % Grid step
%     dx = 1 / (Nx - 1);
%     dy = 1 / (Ny - 1);
% 
%     % Boundary mask
%     isBoundary = (x_coords == 0) | (x_coords == 1) | ...
%                  (y_coords == 0) | (y_coords == 1);
% 
%     % Find center point and its immediate neighbors
%     center_x = 0.5;
%     center_y = 0.5;
% 
%     % Any point within ±dx of center in both x and y
%     % isInfectedRegion = (abs(x_coords - center_x) <= dx) & ...
%     %                    (abs(y_coords - center_y) <= dy);
% 
%     % Combine boundary and infected region
%     isNotSusceptible = isBoundary;
% 
%     % Set S0 = 0 where people are not susceptible
%     S0(isNotSusceptible) = 0;
%     % Infected: 50 only at center cell
%     x_coords = Xic(2,:);   % x ∈ [0,1]
%     y_coords = Xic(3,:);   % y ∈ [0,1]
% 
%     % Find closest grid point to (0.5,0.5)
%     [~,centerIdx] = min((x_coords-0.5).^2 + (y_coords-0.5).^2);
% 
%     I0 = zeros(size(Yic(2,:)),'like',Yic);
%     I0(centerIdx) = 50;
%     R0 = zeros(size(Yic(3,:)),'like',Yic);       % Initially recovered
%     lossIC = mse(Yic(1,:), S0) ...
%        + mse(Yic(2,:), I0) ...
%        + mse(Yic(3,:), R0);
% 
%     % --- Total loss ---
%     % disp("lossPDE", lossPDE, "lossBC", lossBC, "lossIC", lossIC)
%     % fprintf('lossPDE: %f, lossBC: %f, lossIC: %f\n', lossPDE, lossBC, lossIC);
%     loss = lossPDE + lossBC + lossIC;
%     disp(loss);
% 
%     % Gradients w.r.t. learnable params
%     grads = dlgradient(loss,net.Learnables);
% end

%% Loss with including data loss
function [loss,grads, Ybd, Yic, S0, I0, R0, fS, fI, fR, test1, test2, test3, test4] = modelLoss(net,t,x,y,collocBd,collocIC,collocInt, ...
                                  beta,gamma,D_S,D_I,D_R, ...
                                  Xdata,Sdata,Idata,Rdata,epoch)

    % ---------------- Interior collocation points ----------------
    % Xall = dlarray([t; x; y],'CB');   % [3 × N]
    Xall = dlarray(single(collocInt)', 'CB');
    Yall = forward(net,Xall);         % [3 × N]

    S = Yall(1,:); I = Yall(2,:); R = Yall(3,:);

     
    % --- Time derivatives ---
    S_t = dlgradient(sum(S),t,EnableHigherDerivatives=true);
    I_t = dlgradient(sum(I),t,EnableHigherDerivatives=true);
    R_t = dlgradient(sum(R),t,EnableHigherDerivatives=true);

    % --- Spatial derivatives ---
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

    % --- PDE residuals ---
    fS = S_t - (D_S*(S_xx+S_yy) - beta*S.*I);
    fI = I_t - (D_I*(I_xx+I_yy) + beta*S.*I - gamma*I);
    fR = R_t - (D_R*(R_xx+R_yy) + gamma*I);
    % disp("fS:"); disp(size(fS)); disp(dims(fS));
    % disp("fI:"); disp(size(fI)); disp(dims(fI));
    % disp("fR:"); disp(size(fR)); disp(dims(fR));

    lossPDE = mse(fS, zeros(size(fS),'like',fS)) ...
        + mse(fI, zeros(size(fI),'like',fI)) ...
        + mse(fR, zeros(size(fR),'like',fR));

    % ---------------- Boundary condition (S=IR==0) ----------------
    Xbd = dlarray(single(collocBd)','CB');
    Ybd = forward(net,Xbd);
    lossBC = mse(Ybd(1,:), zeros(size(Ybd(1,:)),'like',Ybd)) + ...   % S = 0
         mse(Ybd(2,:), zeros(size(Ybd(2,:)),'like',Ybd)) + ...   % I = 0
         mse(Ybd(3,:), zeros(size(Ybd(3,:)),'like',Ybd));        % R = 0

    % ---------------- Initial condition ----------------
    Xic = dlarray(single(collocIC)','CB');
    Yic = forward(net,Xic);
    S0 = 20*ones(size(Yic(1,:)),'like',Yic);   % Susceptibles in thousands
    Nx = 5; Ny = 5;

    % Zero out the boundaries (as before)
    x_coords = Xic(2,:);   % x ∈ [0,1]
    y_coords = Xic(3,:);   % y ∈ [0,1]
    
    % Grid step
    dx = 1 / (Nx - 1);
    dy = 1 / (Ny - 1);
    
    % Boundary mask
    isBoundary = (x_coords == 0) | (x_coords == 1) | ...
                 (y_coords == 0) | (y_coords == 1);
    
    % Find center point and its immediate neighbors
    center_x = 0.5;
    center_y = 0.5;
    
    % Any point within ±dx of center in both x and y
    % isInfectedRegion = (abs(x_coords - center_x) <= dx) & ...
    %                    (abs(y_coords - center_y) <= dy);
    
    % Combine boundary and infected region
    isNotSusceptible = isBoundary;
    
    % Set S0 = 0 where people are not susceptible
    S0(isNotSusceptible) = 0;
    % Infected: 50 only at center cell
    x_coords = Xic(2,:);   % x ∈ [0,1]
    y_coords = Xic(3,:);   % y ∈ [0,1]

    % Find closest grid point to (0.5,0.5)
    [~,centerIdx] = min((x_coords-0.5).^2 + (y_coords-0.5).^2);

    I0 = zeros(size(Yic(2,:)),'like',Yic);
    I0(centerIdx) = 20;
    R0 = zeros(size(Yic(3,:)),'like',Yic);       % Initially recovered
    lossIC = mse(Yic(1,:), S0) ...
       + mse(Yic(2,:), I0) ...
       + mse(Yic(3,:), R0);

    % ---------------- Data anchors (option 1) ----------------
    Yd = forward(net, Xdata);   % predictions at anchor points
    Sd = Yd(1,:);  Id = Yd(2,:);  Rd = Yd(3,:);
    lambdaS = 1.0; lambdaI = 3.0; lambdaR = 2.0;
    lossData = lambdaS*mse(Sd,Sdata) + lambdaI*mse(Id,Idata) + lambdaR*mse(Rd,Rdata);
    test1=Sd;
    test2=Sdata;
    test3=Id;
    test4=Idata;

    % ---------------- Curriculum PDE weight (option 2) ----------------
    wPDE = min(1.0, epoch/300);  % slowly ramp PDE weight

    % ---------------- Total loss ----------------
    loss = lossPDE + 5*lossIC + 10*lossData + lossBC;

    fprintf('Epoch %d | lossPDE: %.3f | lossBC: %.3f | lossIC: %.3f | lossData: %.3f\n',...
            epoch, extractdata(lossPDE), extractdata(lossBC), ...
            extractdata(lossIC), extractdata(lossData));

    % Gradients
    grads = dlgradient(loss,net.Learnables);
end
