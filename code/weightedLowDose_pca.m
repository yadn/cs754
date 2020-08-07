function resultWeightedPrior = weightedLowDose_pca(y,dim,idx1,I_0,W,eigenVecs,meanTemplate,alphas,lambda0,lambda1,fistaMaxIter)

epsilon = (min(y(:)));
if epsilon < 0
    epsilon = -epsilon + 0.001;
elseif epsilon==0
    epsilon = 0.001;
else
    epsilon = 0;
end

% if(strcmp(methodName,'Rescaled_Non_Linear_Least_Squares_CS'))
%     display(' using Rescaled_Non_Linear_Least_Squares_CS for weighted prior reconstruction');
    v_init = y + epsilon;
    resultWeightedPrior = rescaled_non_linear_LSCS_wted_prior(v_init,dim,idx1,I_0,W,eigenVecs,alphas,meanTemplate,lambda0,lambda1,fistaMaxIter);
end