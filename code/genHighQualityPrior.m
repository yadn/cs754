function [eigenVecs,meanTemplate,alpha] = genHighQualityPrior(pilot,ratio,templateNos,outDirectory)

numTemplates = length(templateNos);
numSlices = 123; % number of slices in a volume;
sizeOfSlice = size(pilot);

volumes  = zeros(sizeOfSlice(1),sizeOfSlice(2),numTemplates);

% Read the templates in 3D format-------------------

for i = 1:numTemplates  
   name = sprintf('../data/template_%d.mat',templateNos(i)) ;
   fileData = load(name);
   fdkVol = fileData.validIm;
   volumes(:,:,i) = fdkVol;
end


% Get the templates and compute their mean----------------------------------------------------------
templateIm = cell(1,numTemplates);
for i = 1:numTemplates
    
    in = double(volumes(:,:,i));
    %figure;imshow(in,[]);colorbar;
%     if i==1
%         mini = min(in(:));
%         dummy = in - mini;
%         maxi = max(dummy(:));
%     end
    in = mat2gray(in);
    temp = in./ratio;

%     imshow(temp,[]); title(i); pause(0.5);    
    if i==1
        templates = zeros((size(temp,1)*size(temp,2)), numTemplates);
        templates(:,1) = temp(:);
        sumI = zeros(size(templates(:,i)));       

    end    
    templates(:,i) = temp(:);
    sumI = sumI + templates(:,i);
    templateIm{i} = temp;
end

meanTemplate = sumI./numTemplates;

% Compute the Covariance Matrix--------------------------------------------------------------------
% Refer to Prof. Ajit's notes on face recognition using PCA

templates = templates -repmat(meanTemplate,1,numTemplates);
L = templates'*templates;
[W,D] = eig(L);
V = templates*W;
V = normc(V);
[m, n] = size(V);

% picking top k eigen values and their corresponding vectors-----------------------------------------------------
% This forms the eigen space of the covariance matrix of the templates-----------------                  

numDim = numTemplates-1;
eigenVals = zeros(1,numDim);
eigenVecs = zeros(m,numDim);

for j = 1:numDim    
    eigenVals(j) = D(n-j+1,n-j+1);
    eigenVecs(:,j) = V(:,n-j+1);
    %imshow(reshape(eigenVecs(:,j),size(temp)),[]);title(j);pause(0.1);
end


%-------------------------------------------------------------------------
%------------Project the FBP of test image onto this eigen space---------------------------
%------------------------------------------------------------------------
% Compute the weights ('alpha') for the FBP of the test image
% Note: Here- cols of eigenVecs are eigenvectors.
alpha = eigenVecs'*(pilot(:) - meanTemplate);  
name = sprintf('%s/highQualityEigenSpace.mat',outDirectory);
save(name,'eigenVecs','meanTemplate','alpha');
end

% name = sprintf('%s/highQualityEigenSpace_%s.mat',outDirectory,methodName);
% save(name,'eigenVecs','meanTemplate','alpha');