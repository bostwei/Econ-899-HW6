function l = labor(data,theta,j,zi)
%LABOR This function generate the labor choice given a a' e j;
%   imput data is the data given
%   theta is the parameter given
%   j is the current age
%   z is current working status

% Unbundling data 
a = data.a ; % the current holding asset
aa = data.aa ; % the next period asst 
e = data.e ;

% bundling the parameter
ggama = theta.gamma;
ttheta = theta.theta;
w = theta.w;
r = theta.r;

l = (ggama .* (1- ttheta) .* e(zi, j) .* w - (1- ggama).* ((1+ r) .* a - aa'))./...
    ((1- ttheta) .* e(zi, j) .* w);

% control labor always positive
l(l<0) = 0;
% control labor always <1
l(l>1) = 1;
end

