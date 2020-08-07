function func_value = compute_f_valueRescaled_wted_prior(theta,y,dim,idx1,lambda1,I_0,W,eigenVecs,alphas,meanTemplate)

%f(theta) = sum_i((y - I_0 exp(-phi psi theta)^2./(I_0 exp(-phi psi theta)) 


theta1 = reshape(theta,[dim(1) dim(2)]);
image = idct2(theta1);
proj = I_0.*exp(-radon(image,idx1));
vec = (y - proj(:)).^2;
vec = vec./proj(:);
func_value1 = sum(vec);


image = image(:);

% prior = meanTemplate + eigenVecs*alphas;
% prior = prior - min(prior(:));
% prior = prior./max(prior(:));
% term3 = W(:).*(image - (prior));

term3 = W(:).*(image - (meanTemplate + eigenVecs*alphas));
func_value3 = lambda1*norm(term3,2);

func_value = func_value1 + func_value3;
