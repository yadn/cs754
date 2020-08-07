function func_value = compute_calc_f2(theta,I_0,v,dim,idx1)

%f(theta) = I_0 exp(-phi psi theta) + v(phi psi theta)
theta1 = reshape(theta,[dim(1) dim(2)]);
image = idct2(theta1);
func_value1 = I_0.*exp(-radon(image,idx1));
func_value1 = sum(func_value1(:));

projections = radon(image,idx1);
func_value2 = v.*projections(:);
func_value2 = sum(func_value2);


func_value = func_value1 + func_value2 ;
