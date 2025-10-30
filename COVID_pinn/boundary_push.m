%% Boundary push and filtering
clear; clc;

%% Parameters
params = load('param_mid_tau5.mat')
Nx = 60; Ny = 60; Nt = 20;
x = linspace(0,1,Nx);
y = linspace(0,1,Ny);
t = linspace(0,1,Nt);

%% Environment
env = load('env_PINN.mat')
boundaryMap = arrayfun(@(c) c.boundry, env.env);
[x, y] = ndgrid(1:60, 1:60);
boundaryX = x(boundaryMap);
boundaryY = y(boundaryMap);

boundaryPoints = [boundaryX(:), boundaryY(:)];
figure;
imagesc(boundaryMap);
axis equal tight;
title('Boundary Map');
hold on;
plot(boundaryY, boundaryX, 'r.', 'MarkerSize', 10);
boundaryMap2 = load('boundaryMap2.mat');

for i = 1:60
    for j = 1:60
        env.env(i,j).boundry = logical(boundaryMap2.boundaryMap2(i,j));
    end
end
% 
boundaryMap = arrayfun(@(c) c.boundry, env.env);
[x, y] = ndgrid(1:60, 1:60);
boundaryX = x(boundaryMap);
boundaryY = y(boundaryMap);

boundaryPoints = [boundaryX(:), boundaryY(:)];
figure;
imagesc(boundaryMap);
axis equal tight;
title('Boundary Map');
hold on;
plot(boundaryY, boundaryX, 'r.', 'MarkerSize', 10);
% boundaryMap2 = boundaryMap;
