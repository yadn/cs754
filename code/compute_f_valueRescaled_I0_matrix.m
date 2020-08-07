function func_value = compute_f_valueRescaled_I0_matrix(theta,y,dim,idx1,I_0_matrix)

%f(theta) = sum_i((y - I_0 exp(-phi psi theta)^2./(I_0 exp(-phi psi theta)) 


theta1 = reshape(theta,[dim(1) dim(2)]);
image = idct2(theta1);
proj = I_0_matrix.*exp(-radon(image,idx1));
vec = (y - proj(:)).^2;
vec = vec./proj(:);
func_value1 = sum(vec);

func_value = func_value1 ;
