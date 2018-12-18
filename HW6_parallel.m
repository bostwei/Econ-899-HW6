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

n = 0.011;
% Grid Number
Na = 200;

% worker retire
JR = 46;

% Benefits

ggama = 0.42;
ssigma = 2;

% idiosyncratic productivity
z = [3;0.5];
% The probability of productivity
Pz = [0.2037;0.7963];

% Trhansition probability
Pi = [0.9261 1-0.9261;
     1-0.9811 0.9811];

% capital share
aalpha = 0.36;
% depreciation rate
delta = 0.06;


%% Calculate the stationary distirbution
theta0 = 0.11;
[out0] = StationDist(theta0);

thetaN = 0;
[outN] = StationDist(thetaN);

%% Calculate the transition path
K_0 = out0.K;
K_N = outN.K;

L_0 = out0.L;
L_N = outN.L;
% set the adjustment time period to N = 30
N = 30; % this is the transition path time
Na = 200;
J = 66; % The is the total age people can have  

% Generate the linear of K_t
K_t =linspace(K_0, K_N, N)';

% Generate the linear policy indicator 
theta_t = [theta0 * ones(N-1,1); thetaN];

% First update the L to be constaint
L_t =  linspace(0.3008,0.3132,N)';


%% calculate the transition path
v1_zh = zeros(Na,J,N);
v1_zl = zeros(Na,J,N); %This is the value function for each parth
v1_zh(:,:,N) = outN.v_zh;
v1_zl(:,:,N) = outN.v_zl;

K_new = zeros(N,1);
K_new(N,1) = K_N;

L_new = zeros(N,1);
L_new(N,1) = L_N;

w_t = zeros(N,1);
w_t(N,1) = outN.w;

r_t = zeros(N,1);
r_t(N,1) = outN.r;

% 
diff = 10;
Iter = 0;
tol_price = 0.0001;

while diff > tol_price
tic
for t = N-1:-1:1

% bundle the parameter input for calculating the
input_par.theta = theta_t(t);
input_par.K = K_t(t);
input_par.L = L_t(t);
input_par.v0_zh = v1_zh(:,:,t+1);
input_par.v0_zl = v1_zl(:,:,t+1);

out_t = TransDist(input_par);
% updating the ouput
v1_zh(:,:,t) = out_t.v_zh;
v1_zl(:,:,t) = out_t.v_zl;
K_new(t) = out_t.K;
L_new(t) = out_t.L;
% store the wage and interest rate 
w_t(t) = out_t.w;
r_t(t) = out_t.r;
% fprintf('This is for %d transition period. \n',t);
end
% calclulate the difference 
c_K = K_t - K_new; % the delta of capital
c_L = L_t - L_new; % the delta of labor
c_Max = max(max(c_K,c_L));
diff = abs(c_Max);

% update K and L
K1 = 0.9 * K_t + 0.1 * K_new;
L1 = 0.9 * L_t + 0.1 * L_new;

K_t = K1;
L_t = L1;

Iter = Iter + 1;
fprintf('Iteration %d: the difference is %.4f\n', Iter, diff);
toc
end
%% Plot the transition path
figure (1)
t=1:N;
subplot(2,2,1);
plot(t,K_t);
title('The transition path of the Capital')
subplot(2,2,2);
plot(t,L_t);
title('The transition path of the Labor')
subplot(2,2,3);
plot(t,r_t);
title('The transition path of Interest Rate');
subplot(2,2,4);
plot(t,w_t);
title('The transition path of Wage');

%% calculate the 
v_0_zh = out0.v_zh;
v_0_zl = out0.v_zl;

v_N_zh = outN.v_zh;
v_N_zl = outN.v_zl;


EV0_zh = (v_N_zh./v_0_zh).^(1/(ggama*(1-ssigma)))-1;
EV0_zl = (v_N_zl./v_0_zl).^(1/(ggama*(1-ssigma)))-1;

% store the density
mu0_zh = out0.mu_zh;
mu0_zl = out0.mu_zl;

EV_zh = sum(EV0_zh .* mu0_zh); 
EV_zl = sum(EV0_zl .* mu0_zl);
EV = EV_zh + EV_zl;


figure (2)
j = 1:66;
plot(j,EV');
title('EV of transition.')