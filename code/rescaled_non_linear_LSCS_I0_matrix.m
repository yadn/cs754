function reconRescaledNLLSCS = rescaled_non_linear_LSCS_I0_matrix(y,dim,idx1,I_0_matrix,fistaMaxIter,lambda0)

addpath('FISTA-master/');
addpath('FISTA-master/proj/');
addpath('FISTA-master/utils/');
theta_est = zeros(dim(1)*dim(2),1);
%lambda = 1;
gradFn = @(z) computeGradientRescaled_I0_matrix(z,y,dim,idx1,I_0_matrix);
calc_f = @(z) compute_f_valueRescaled_I0_matrix(z,y,dim,idx1,I_0_matrix);
calc_F = @(z) compute_cost_valueRescaled_I0_matrix(z,y,dim,idx1,lambda0,I_0_matrix);
opts = struct('lambda',lambda0,'L0',1000,'eta',1.1,'max_iter',fistaMaxIter,'tol',1e-4);
theta_est = fista_backtracking_updated(calc_f, gradFn, theta_est, opts,calc_F);
theta_est = reshape(theta_est,[dim(1) dim(2)]);
reconRescaledNLLSCS= idct2(theta_est);
% fprintf('No. of negative values');
% length(find(reconNLLSCS<0))