close all;
clear all;
clc;
dataset = 'okra';
numAngles = 360;
reconMethod = 4;
hypoMethod = 2;
ratio = 100;


gaussianNoisePercent = 2 
switch hypoMethod
    case 1
        hypothesisTestName = 'T-test'
    case 2
        hypothesisTestName = 'Z-test'
    case 3
        hypothesisTestName = 'K-S-test'
end

switch reconMethod
    case 1
        methodName = 'Post_log_FBP'
    case 2
        methodName = 'Non_Linear_Least_Squares_CS'
    case 3
        methodName = 'NLL_Poisson_Gaussian'
    case 4
        methodName = 'Rescaled_Non_Linear_Least_Squares_CS'
end          


outDirectory = sprintf('results/svm/training/');
mkdir(outDirectory);
name = sprintf('../../../mPICCS/2D/data/templates/okra/okra6_okra%d_reg_450views_fdk.mat',4) ; % one of the templates
fileData = load(name);
volume = fileData.FDK;

%-------start: get one of the templates for referenfce intensity----
name = sprintf('../../../mPICCS/2D/data/templates/okra/okra6_okra%d_reg_450views_fdk.mat',5) ; % one of the templates
templateFileData = load(name);
template_fdkVol = templateFileData.FDK;
templateVolume = double(template_fdkVol(15:end-14,15:end-14,:));
mini = min(templateVolume(:));
dummy = templateVolume - mini;
maxi = max(dummy(:));
%-------end: get one of the templates for referenfce intensity--------------------------------------------

testIm= double(volume(15:end-14,15:end-14,30));
testIm = testIm - min(testIm(:));
testIm = testIm./max(testIm(:));
testIm = testIm./ratio;

fname = sprintf('%s/testIm.mat',outDirectory);
save(fname,'testIm');
name = sprintf('%s/testIm.png',outDirectory);
imwrite(testIm,name);
testIm1 = testIm - min(testIm(:));
testIm1 = testIm1./max(testIm1(:));
dim = size(testIm);

%% Generate mesurements
I_0 = 2000;
[y, y_noNoise, idx1, noiseSD, NSR] = generateMeasurements(testIm,I_0,numAngles,gaussianNoisePercent,outDirectory);

%         name = sprintf('%s/measurements.mat',outDirectory);
%         f = load(name);
%         y = f.y;
%         y_noNoise = f.y_noNoise;
%         idx1 = f.idx1;
%         noiseSD = f.sigma;
%         NSR = f.NSR;

lambda1_value = 1.1;
lambda0 = 1;    


%% Pilot reconstruction
fistaMaxIter = 100;
lambda = 1;
[pilot, ssimPilot, meanSqErrorPilot] = reconstructPilot(y,idx1,dim,I_0,noiseSD,testIm1,methodName,outDirectory,fistaMaxIter,lambda);

%         name = sprintf('%s/pilot_%s.mat',outDirectory,methodName);
%         f = load(name);
%         pilot = f.pilot;
%         ssimPilot = f.ssimPilot;
%         meanSqErrorPilot = f.meanSqErrorPilot;

figure;imshow(pilot,[]);colorbar;

         %% Generate high quality prior
         
          [eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior_training(idx1,pilot,ratio,outDirectory,methodName);
         
%          name = sprintf('%s/highQualityEigenSpace_%s.mat',outDirectory,methodName);
%          f = load(name);
%          eigenVecsSpatial = f.eigenVecs;
%          meanTemplateSpatial = f.meanTemplate;
%          alpha = f.alpha;
% 
         meanImage = reshape(meanTemplateSpatial,dim);
         prior = meanTemplateSpatial + (eigenVecsSpatial*alpha);
         prior = reshape(prior,dim);         

        fname = sprintf('%s/prior.mat',outDirectory);
        save(fname,'prior');
        name = sprintf('%s/prior.png',outDirectory);
        output = prior;
        output = output - min(output(:));
        output = output./max(output(:));  
        imwrite(output,name); 


%%  Project test measurements onto Eigenspace built from template measurements
[allReconProj] = project_on_MeasurementEigenSpace_2D_okra_training(idx1,y,I_0,dim,ratio,outDirectory);

%         name = sprintf('%s/projectionEigenSpace.mat',outDirectory);
%         f = load(name);
%         allReconProj = f.allReconProj;

%%  Generate ground truth inlier info

[inlier, inlierProj, binaryInlierProj] = genGroundTruthInlier(idx1,eigenVecsSpatial,meanTemplateSpatial,y_noNoise,dim,outDirectory);     

%         name = sprintf('%s/groundTruth_inlier.mat',outDirectory);
%         f = load(name);
%         inlier = f.inlier;
%         inlierProj = f.inlierProj;
%         binaryInlierProj = f.binaryInlierProj;

figure;imshow(inlier,[]);colorbar();
binaryInlier1 = imbinarize(inlier, .0029);
SE = strel('sphere',4);
binaryInlier = imdilate(binaryInlier1,SE);
figure;imshow(binaryInlier,[]);


%% hypothesis testing ------------------------------------------------

[hypothesisTestResultFBP, hypothesisTestGrayResultFBP]= performHypothesisTesting(allReconProj,y,numAngles,noiseSD,hypothesisTestName,idx1,dim,outDirectory);

figure;imshow(hypothesisTestGrayResultFBP,[]);
figure;imshow(hypothesisTestResultFBP,[]);

%% Train svm classifier

name = sprintf('%s/svm_traning_data',outDirectory);
save(name,'binaryInlier','hypothesisTestResultFBP');









