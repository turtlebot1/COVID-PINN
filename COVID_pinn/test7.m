%% SIMPLE PDE + RNN (LSTM) HYBRID EXAMPLE
% Predicts S,I,R for day t+1 given last 15 days
% Enforces PDE residual with Laplacian

clear; clc;

% --------------------------------------------------
% Example synthetic data (replace with your own)
H = 10; W = 10; T = 60;      % grid size and days
S = rand(H,W,T); I = rand(H,W,T)*0.1; R = zeros(H,W,T);

dt = 1; dx = 1; dy = 1;
beta = 0.3; gamma = 0.1;
D_S = 0.05; D_I = 0.05; D_R = 0.05;

% --------------------------------------------------
% Build a simple dataset: one cell -> one sequence
win = 15;   % history length
Xseq = {}; 
Ytar = [];   % make responses a cell array

for t = win:(T-1)
    % input sequence [3 × win]
    xSeq = [ squeeze(S(5,5,t-win+1:t));
             squeeze(I(5,5,t-win+1:t));
             squeeze(R(5,5,t-win+1:t)) ];
    xSeq = reshape(xSeq,[3,win]);  
    Xseq{end+1} = xSeq;

    % target output [3 × 1]
    Ytar(:,end+1) = [S(5,5,t+1); I(5,5,t+1); R(5,5,t+1)];
end

% --------------------------------------------------
% Define simple LSTM network
layers = [ ...
    sequenceInputLayer(3)
    lstmLayer(32,"OutputMode","last")
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(3)
    regressionLayer];

options = trainingOptions("adam", ...
    MaxEpochs=5, ...
    MiniBatchSize=16, ...
    Verbose=1);

disp(numel(Xseq))
disp(numel(Ytar))
size(Xseq{1})   % should be [3 15]
Ytar = Ytar';
size(Ytar)   % should be [3 1]


% --------------------------------------------------
% Train (data only, no PDE yet)
net = trainNetwork(Xseq,Ytar,layers,options);

% --------------------------------------------------
% PDE residual check (for one prediction)
xTest = Xseq{end};
yPred = predict(net,{xTest});   % { } required

% Compute residual (very simplified)
S_next = yPred(1); I_next = yPred(2); R_next = yPred(3);
S_prev = xTest(1,end); I_prev = xTest(2,end); R_prev = xTest(3,end);

% Forward Euler residual for I
resI = (I_next - I_prev)/dt - (beta*S_next*I_next - gamma*I_next);
disp("Residual I:"); disp(resI);