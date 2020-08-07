function gradient = computeGradientRescaled_I0_matrix(theta,y,dim,idx1,I_0_matrix)


theta1 = reshape(theta,[dim(1) dim(2)]);
image = idct2(theta1);
proj = I_0_matrix.*exp(-radon(image,idx1));
var1 = y - proj(:);

var2 = 1+ 0.5*(var1./proj(:));

var = var1.*var2;

numAngles = length(idx1);
var = reshape(var,[(size(y,1))/numAngles numAngles]);

var = iradon(var,idx1,'linear','Cosine');
var = var(2:2+dim(1)-1,2:2+dim(2)-1);
var = dct2(var);
var = var(:);

gradient = 2.*var;

