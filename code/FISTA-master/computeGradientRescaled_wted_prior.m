function gradient = computeGradientRescaled_wted_prior(theta,y,dim,idx1,lambda1,I_0,W,eigenVecs,alphas,meanTemplate)


theta1 = reshape(theta,[dim(1) dim(2)]);
image = idct2(theta1);
proj = I_0.*exp(-radon(image,idx1));
var1 = y - proj(:);

var2 = 1+ 0.5*(var1./proj(:));

var = var1.*var2;

numAngles = length(idx1);
var = reshape(var,[(size(y,1))/numAngles numAngles]);

var = iradon(var,idx1,'linear','Cosine');
var = var(2:2+dim(1)-1,2:2+dim(2)-1);
var = dct2(var);
var = var(:);

image = image(:);

% prior = meanTemplate + eigenVecs*alphas;
% prior = prior - min(prior(:));
% prior = prior./max(prior(:));
% term2 = W(:).*(image - (prior));

term2 = W(:).*(image - (meanTemplate + eigenVecs*alphas));
term2 = W(:).*term2;
term2 = reshape(term2,dim);
term2 = dct2(term2);
term2 = lambda1.*term2(:);

gradient = 2.*var + 2.*term2;

