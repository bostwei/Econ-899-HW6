function [out] = TransDist(input_par)
%STATIONDIST compute the stationary distribution given the social security
%level theta
% 
% - Input theta contain thress variable 
% - input.theta = ttheta % labor income tax
% - input.K = K % aggregate capital choice 



% out.mu_phi_zh  % stationary distribution for zh people
% out.mu_phi_zl % stationary distribution for zl people
% out.v_zh % stationary value function for zh people
% out.v_zl  % stationary value function for zl people
% out.L  % accumulative distribution for labor surply
% out.K  % accumulative distribution for capital surply
%% Data Initiated 
global ef

N = 66;
n = 0.011;
% Grid Number
Na = 200;

% worker retire
JR = 46;

% labor income tax
ttheta  = input_par.theta;

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

% Asset Space
alb = 0;
aub = 5;

A = linspace(alb, aub, Na)';

a = A* ones(1,Na);
aa = A* ones(1,Na); 




% initiate the density mu of each age corhort, mu(i) is the density of 
% agent in age i.   
mu = ones(N,1);
for i = 1: N-1
    mu(i+1) = mu(i)/(1+n);
end
% normalized mu, the sized, of population to be 1
mu = mu./sum(mu);

%% Dynamic Programming Problem
L0 = input_par.L; %sum(sum(mu(1:JR-1)));
% Inital guess of capital 
K0 = input_par.K; %

% Note that v0 is the future variable, v1 is the current variable
v0_zh = input_par.v0_zh; 
v0_zl = input_par.v0_zl; 

v0_r = v0_zh(:,JR:N); % cutting the aging group down 
%-------------- Begin trail and error to find the euqilibirumn wage -----
% diff = 10;
% Iter = 0;
% tol_price = 0.001;
% 
% while diff > tol_price
% tic

% compute the wage and interest rate given guess L0 and K0
w = (1-aalpha) * (K0 / L0) ^aalpha; 
r = aalpha * (L0 / K0) ^ (1-aalpha) - delta;
% pention benefit
 b = (ttheta * w *L0)/(sum(mu(JR:N)));

bbeta = 0.97;

%------------- The utility for the retireed agent ----------------
% consumption of returement J = N is to consume all of its storage
%----------- The value function for retired agent ----------------
c_r = (1+r)*a + b - aa';
c_r(c_r <=0) = NaN;

% utility of the retirement
u_r = (c_r .^ ((1-ssigma) .* ggama))./(1-ssigma);
u_r(isnan(u_r)) = -inf;

v1_r = zeros(Na,N-JR+1); % v1 is the current period of value function.

v1_r(:,N-JR+1) = u_r(:,1); % J = N person will not choose saving for next period.

dec_r = zeros(Na,N-JR+1);
% find the value function for the j<T age agent
for j = N-JR:-1:1
    w_r = u_r + bbeta .*v0_r(:,j+1)';
    [v1_r(:,j), dec_r(:,j)] = max(w_r,[],2);
end

%%
%----------- The utility function for worked agent ----------------
% labor efficiency 
% - first row is ef if it get z = 3
% - second row is the ef if it get z = 0.5
e = z * ef';

% the last period of value function of working agent is the same as the
% retire people

v0_w_zh = v0_zh(:,1:JR);
v0_w_zl = v0_zl(:,1:JR);


% Initiate the decision rule for people d_l(a| j) d_aa(a|j)
% - row of each is the a
% - collumn is the age

% bundling data
data.a = a; % the current holding asset
data.aa = aa; % the next period asst 
data.e = e;

% % bundling the parameter for labor calculation
theta_l.gamma = ggama;
theta_l.theta = ttheta;
theta_l.w = w;
theta_l.r = r;

% compute the asset choice for each age group of people
dec_aa_zh = zeros(Na,JR-1);
dec_aa_zl = zeros(Na,JR-1);

% compute the labor choice for each age group of people
dec_l_zh = zeros(Na,JR-1);
dec_l_zl = zeros(Na,JR-1);

v1_w_zh = zeros(Na,JR-1);
v1_w_zl = zeros(Na,JR-1);

for j = JR-1
     % extract the value function for the previous period
    v0_w_zh_temp = v0_w_zh(:,j+1);
    v0_w_zl_temp = v0_w_zl(:,j+1);

    % calculate the labor choice given agent j
    l_zh = labor(data,theta_l,j,1); % labor choic when status is high
    l_zl = labor(data,theta_l,j,2); % labor choice when status is low

    % searh for the best aa' for each a


    % extract the labor efficiency for the current age
    e1j = e(1,j);
    e2j = e(2,j);

    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e1j * l_zh + (1+r) * a - aa';
    c_w_zl = w * (1-ttheta)* e2j * l_zl + (1+r) * a - aa';
    c_w_zh(c_w_zh <0) = NaN; 
    c_w_zl(c_w_zl <0) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l_zh).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l_zl).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(isnan(u_w_zh)) = -inf;
    u_w_zl(isnan(u_w_zl)) = -inf;
    
    w_zh = u_w_zh + bbeta * v0_w_zh_temp';
    w_zl = u_w_zl + bbeta * v0_w_zl_temp';
   
   % --------------make choice of (aa) given a ------------------
   % we frist choose aa
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l
   [v1_zh_temp, dec_zh_aa] = max(w_zh,[],2);
   [v1_zl_temp, dec_zl_aa] = max(w_zl,[],2); 
   
   % storage the asset choice
    dec_aa_zh(:,j) = dec_zh_aa;
    dec_aa_zl(:,j) = dec_zl_aa;  
   
   % storage the labor choice 
   for i=1:Na
   dec_l_zh(i,j) =  l_zh(i,dec_zh_aa(i));
   dec_l_zl(i,j) =  l_zl(i,dec_zl_aa(i));
   end
%    
    % storage the value function
    v1_w_zh(:,j) = v1_zh_temp;
    v1_w_zl(:,j) = v1_zl_temp;
end

%%

%------- For the rest of the working agent  -------------------------------
 for j = JR-2:-1:1
    % extract the value function for the previous period
    v0_w_zh_temp = v0_zh(:,j+1);
    v0_w_zl_temp = v0_zl(:,j+1);

    % the transition probability
    P11= Pi(1,1);
    P21= Pi(2,1);
    P12= Pi(1,2);
    P22= Pi(2,2);

    % calculate the labor choice given agent j
    l_zh = labor(data,theta_l,j,1); % labor choic when status is high
    l_zl = labor(data,theta_l,j,2); % labor choice when status is low
    
    % searh for the best aa' for each a
    % extract the labor efficiency for the current age
    e1j = e(1,j);
    e2j = e(2,j);
          
    % ------------ the labor l and future asset aaa choice of worker --------------------
    % consumption of worker
    c_w_zh = w * (1-ttheta)* e1j * l_zh + (1+r) * a - aa';
    c_w_zl = w * (1-ttheta)* e2j * l_zl + (1+r) * a - aa';
    c_w_zh(c_w_zh <0) = NaN; 
    c_w_zl(c_w_zl <0) = NaN;
  % utility of the worker is 
    u_w_zh = (c_w_zh.^ggama.*(1-l_zh).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zl = (c_w_zl.^ggama.*(1-l_zl).^(1-ggama)).^(1-ssigma)/(1-ssigma);
    u_w_zh(isnan(u_w_zh)) = -inf;
    u_w_zl(isnan(u_w_zl)) = -inf;
    
    w_zh = u_w_zh + bbeta * (P11 * v0_w_zh_temp' + P12 * v0_w_zl_temp');
    w_zl = u_w_zl + bbeta * (P21 * v0_w_zh_temp' + P22 * v0_w_zl_temp') ;
   
   % --------------make choice of aa given a ------------------
   % we frist choose aa
   % - v1_zh_aa the optimal value w_zh after choosing aa, given a. The row of
   % w_zh is varies of l.
   % - dec_zh_aa is the optimal choice of aa given varies of l
   [v1_zh_temp, dec_zh_aa] = max(w_zh,[],2);
   [v1_zl_temp, dec_zl_aa] = max(w_zl,[],2); 

   
    % storage the asset choice
    dec_aa_zh(:,j) = dec_zh_aa;
    dec_aa_zl(:,j) = dec_zl_aa;  
   
   % storage the labor choice 
   for i=1:Na
   dec_l_zh(i,j) =  l_zh(i,dec_zh_aa(i));
   dec_l_zl(i,j) =  l_zl(i,dec_zl_aa(i));
   end
%    
    % storage the value function
    v1_w_zh(:,j) = v1_zh_temp;
    v1_w_zl(:,j) = v1_zl_temp;
    
%fprintf('The current age group is %d .\n',j);

 end

% % plot the policy function for h0d0
% figure(1)
% plot(A,A(dec_aa_zh(:,1)),A,A(dec_aa_zl(:,1)));% the policy function for employment state
% legend({'high efficiency policy function','low efficiency policy function'},'Location','southeast')
% xlabel('a') 
% ylabel('aa')
% refline(1,0) 

%% Stationary distribution


% -------- compute the asset choice --------------------
% compute the decision matrix for working agent
% - first argument is a
% - second argument is its asset choice in the next period aa
% - the third argument is age j
g_aa_zh = zeros(Na,Na,JR-1);
g_aa_zl = zeros(Na,Na,JR-1);
for j = 1:JR-1
    
    for i = 1:Na
        % create the transition criterion
        g_aa_zh(i,dec_aa_zh(i,j),j) = 1;
                % create the transition criterion
        g_aa_zl(i,dec_aa_zl(i,j),j) = 1;
    end
    
end

% compute the decision matrix for retiring agent
% - first argument is a
% - second argument is its asset choice in the next period aa
% - the third argument is age j

g_aa_r = zeros(Na,Na,N-JR);
% replace the last collumn to always choose asset 1
dec_r(:,N-JR) = ones(Na,1);
for j = 1:N-JR % the reason that I putt J-JR-1 here is that the last coloum of dec_r = 0
    for i = 1:Na
        % create the transition criterion
        g_aa_r(i,dec_r(i,j),j) = 1;
                % create the transition criterion
        g_aa_r(i,dec_r(i,j),j) = 1;
    end
end

% compute the transition matrix for working agent
 trans_aa=zeros(2*Na,2*Na,JR-1);
for j = 1:JR-1
    trans_aa(:,:,j) = [g_aa_zh(:,:,j)*P11, g_aa_zl(:,:,j)*P12 
                       g_aa_zh(:,:,j)*P21, g_aa_zl(:,:,j)*P22 ];
    trans_aa(:,:,j) = trans_aa(:,:,j)'; 
end


%%
% -------- the wealth distribution of each cohort -------------
% - phi_zh(a,j) is the density of agent asset holding aa in age j with high
% working efficiency
% - phi_zl(a,j) is the density of agent asset holding aa in age j with low
% working efficiency
phi_zh = zeros(Na,JR-1); 
phi_zl = zeros(Na,JR-1);

% initate the new born generate people will be hold 0 asset
phi_zh(1,1) = Pz(1); 
phi_zl(1,1) = Pz(2); 

phi = [phi_zh;phi_zl];

% calculate the transition of asset holding choice for working agent
for j = 1:JR-1

    phi(:,j+1) = trans_aa(:,:,j) *  phi(:,j);

end

phi_zh = phi(1:Na,:);
phi_zl = phi(Na+1:2*Na,:);

% calcualte the transition of asset holding for retired agent
phi_r = zeros(Na,N-JR);
% extract the asset holding before retire
retire_a = reshape(phi(:,JR),[Na,2]);
% sum over the asset holding choice over
phi_r(:,1) = sum(retire_a,2);

% calculate the transition of asset holding choice for retired agent
% Note: The first period is the same as the last period of working agent 
for j = 1:N-JR

    phi_r(:,j+1) = g_aa_r(:,:,j)' *  phi_r(:,j);

end

% divid population amount into working group and retire group
mu_work = diag(mu(1:JR-1));
mu_r = diag(mu(JR:N));

% rescale the working population density of each age cohort by age amount
mu_phi_zh = phi_zh(:,1:JR-1) * mu_work;
mu_phi_zl = phi_zl(:,1:JR-1) * mu_work;

% rescale the retiring population density of each age cohort by age amount
mu_phi_r = phi_r * mu_r;

% merge the two density distribution
mu_zh = [mu_phi_zh, mu_phi_r];
mu_zl = [mu_phi_zl, mu_phi_r];

% merge the two value function
v_zh = [v1_w_zh, v1_r];
v_zl = [v1_w_zl, v1_r];



%% Updating Criterian

K_new = sum(A.* sum((mu_phi_zh + mu_phi_zl),2) + A.* sum(mu_phi_r,2));

% ---- calculate the labor choices distribution -----------------
% mu_l_zh is the labor choice density after adjust the cohor distribution
mu_l_zh =  dec_l_zh .*  mu_zh(:,1:JR-1);
mu_l_zl =  dec_l_zl .*  mu_zl(:,1:JR-1);

% calcualte the labor offer for different z
l_zh = mu_l_zh * diag(e(1,:));
l_zl = mu_l_zl * diag(e(2,:));

L_new = sum(sum(l_zh+l_zl));


% % update K and L
% K1 = 0.5 * K0 + 0.5 * K_new;
% L1 = 0.5 * L0 + 0.5 * L_new;
% 
% diff = [K_new-K0, L_new-L0];
% diff = max(abs(diff));
% 
% K0 = K1;
% L0 = L1;

% Iter = Iter + 1;
% 
% %fprintf('Iteration %d: the difference is %.4f, wage is %.4f, interest rate is %.4f \n ', Iter, diff,w, r);
% 
% toc
% end

% bundling the output
out.mu_phi_zh = mu_phi_zh; % stationary distribution for zh people
out.mu_phi_zl = mu_phi_zl; % stationary distribution for zl people
out.v_zh = v_zh; % stationary value function for zh people
out.v_zl = v_zl; % stationary value function for zl people
out.L = L_new; % accumulative distribution for labor surply
out.K = K_new; % accumulative distribution for capital surply
out.w = w; % output the wage
out.r = r; % output interest rate



end

