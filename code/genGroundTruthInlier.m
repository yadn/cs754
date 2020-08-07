function [inlier, inlierProj, binaryInlierProj] = genGroundTruthInlier(angleArr,eigenVecs1,meanTemplateIm1,y_noNoise,dim,outDirectory)


numAngles = length(angleArr);
projNoNoise = reshape(y_noNoise,[size(y_noNoise,1)/numAngles numAngles]);
result_fbpNoNoise = iradon(projNoNoise,angleArr,'linear','cosine');
result_fbpNoNoise = result_fbpNoNoise(2:2+dim(1)-1,2:2+dim(2)-1);

% project y_noNoise onto the Projection-Eigenspace
%[eigenVecs1,meanTemplateIm1] = genEigenSpaceSpatialDomain(idx1,ratio);
coeffNoNoise = eigenVecs1'*(result_fbpNoNoise(:) - meanTemplateIm1);  
reconImNoNoise = zeros(size(result_fbpNoNoise(:)));
numEigenVectors = length(coeffNoNoise);
for j = 1:numEigenVectors
    reconImNoNoise = reconImNoNoise + (coeffNoNoise(j)*eigenVecs1(:,j));
end
reconImNoNoise = reconImNoNoise + meanTemplateIm1;
reconImNoNoise  = reshape(reconImNoNoise,size(result_fbpNoNoise));
errorImNoNoise = abs(result_fbpNoNoise - reconImNoNoise);

fname = sprintf('%s/inlier.mat',outDirectory);
save(fname,'errorImNoNoise');
name = sprintf('%s/inlier.png',outDirectory);
temp = errorImNoNoise;
temp = temp - min(temp(:));
temp = temp./max(temp(:));
imwrite(temp,name);


%% -----------------------Computing the projections of inlier alone---------------------------------------------

file = load(sprintf('%s/inlier.mat',outDirectory));
inlier = file.errorImNoNoise;
%figure;imshow(inlier,[]);
inlierProj = radon(inlier,angleArr);
binaryInlierProj = imbinarize(inlierProj);
end

% name = sprintf('%s/groundTruth_inlier.mat',outDirectory);
% save(name,'inlier','inlierProj','binaryInlierProj');
