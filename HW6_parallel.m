%*************************************************************************
% Econ 899 HW6 
% Shiyan Wei
% 12/17/2018
% ************************************************************************

% ************************************************************************
% This script is used to find the transition in the OLG model
% the homework 6 for Econ 899
% ***********************************************************************
clear
clc
close all

% Import labor effciency
global ef
ef = importdata('ef.txt');

%% Calculate the stationary distirbution
theta0 = 0.11;
[out0] = StationDist(theta0);

thetaN = 0;
[outN] = StationDist(thetaN);

%% Calculate the transition path
K_0 = out0.K;
K_N = outN.K;


% set the adjustment time period to N = 30
N = 30; % this is the transition path time
Na = 200;
J = 66; % The is the total age people can have  

K_t =linspace(K_0, K_N, N)';
theta_t = [theta0 * ones(N-1,1); thetaN];

%% calculate the transition path
v1_zh = zeros(Na,J,N);
v1_zl = zeros(Na,J,N); %This is the value function for each parth
v1_zh(:,:,N) = outN.v_zh;
v1_zl(:,:,N) = outN.v_zl;

K_new = zeros(N,1);
K_new(N,1) = K_N;
for t = N-1:-1:1
    tic
% bundle the parameter input for calculating the
input_par.theta = theta_t(t);
input_par.K = K_t(t);
input_par.v0_zh = v1_zh(:,:,t+1);
input_par.v0_zl = v1_zl(:,:,t+1);

out_t = TransDist(input_par);
% updating the ouput
v1_zh(:,:,t) = out_t.v_zh;
v1_zl(:,:,t) = out_t.v_zl;
K_new(t) = out_t.K;
fprintf('This is for %d transition period. \n',t);
toc
end

