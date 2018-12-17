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
N = 30;
K_t =linspace(K_0, K_N, N)';
theta_t = [theta0 * ones(N-1,1); thetaN];

%%
v1_zh_temp = outN.v_zh;
v1_zl_temp = outN.v_zl;

for t = N-1:-1:1
% bundle the parameter input for calculating the
input_par.theta = theta_t(t);
input_par.K = K_t(t);
input_par.v0_zh = v1_zh_temp;
input_par.v0_zl = v1_zl_temp;

out_t = TransDist(input_par);
% updating the 
v1_zh_temp = out_t.v_zh;
v1_zl_temp = out_t.v_zl;
end

