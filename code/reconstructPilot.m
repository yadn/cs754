function [pilot,ssimPilot,rmsePilot] = reconstructPilot(y,angleArr,dim,I_0,testIm,methodName,fistaMaxIter,lambda)

if (strcmp(methodName,'Rescaled_Non_Linear_Least_Squares_CS'))
    
    fprintf(' using Rescaled_Non_Linear_Least_Squares_CS for pilot reconstruction');
    reconRescaledNLLSCS = rescaled_non_linear_LSCS(y,dim,angleArr,I_0,fistaMaxIter,lambda);
    
%     name = sprintf('%s/result_RescaledNLLSCS_%d_angles_%d_I_0.mat',outDirectory,numAngles,I_0);
%     save(name,'reconRescaledNLLSCS');
    output = reconRescaledNLLSCS;
    output = output - min(output(:));
    output = output./max(output(:));
%     name = sprintf('%s/result_RescaledNLLSCS_%d_angles_%d_I_0.png',outDirectory,numAngles,I_0);
%     imwrite(output,name);
    ssimPilot = ssim(output,testIm);
    rmsePilot  = sqrt(immse(output,testIm));
    pilot = reconRescaledNLLSCS;
    
end
        
% fname = sprintf('%s/pilot_%d.mat',outDirectory,I_0);
% save(fname,'pilot');
% name = sprintf('%s/pilot_%d.png',outDirectory,I_0);
% output = pilot;
% output = output - min(output(:));
% output = output./max(output(:));  
% imwrite(output,name);   
% name = sprintf('%s/pilot_%s.mat',outDirectory,methodName);
% save(name,'pilot','ssimPilot','meanSqErrorPilot');