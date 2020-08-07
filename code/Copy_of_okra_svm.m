close all; clc;
numAngles = 360;
reconMethod = 4;
methodName = 'Rescaled_Non_Linear_Least_Squares_CS';
gaussianNoisePercent = 1; 
testImNo=3;
validImNo = 4;
templateNos = [5,6,7];

%% Traning Part %%
% prior templates: 3,4,5 validation Image: 6

outDirectory = sprintf('./result/matnew');
mkdir(outDirectory);

ImAddr = sprintf('/home/yadnyesh/Desktop/seminar/fewviews/2D/okra/data_okra/okra6_okra%d_reg_450views_fdk.mat',validImNo); % validation image
valid_data = load(ImAddr);
valid_data = valid_data.FDK;
validIm= double(valid_data(20:end-19,20:end-19,30));%(15:end-14,15:end-14,30));
name = sprintf('%s/validIm_template_%d.mat',outDirectory, validImNo);
% save(name,'validIm');
% validIm = load(name);
% validIm = validIm.validIm;
figure('Name','Valid_Image','NumberTitle','off');imshow(validIm,[]);
validIm = mat2gray(validIm);
ratio = 100;
validIm = validIm./ratio;
dim = size(validIm);
angleArr = linspace(0,180,numAngles+1);
angleArr = angleArr(1:numAngles);
%% Generate mesurements
I_0 = 2000;
[y, y_noNoise] = generateMeasurements(validIm,I_0, angleArr, gaussianNoisePercent);

%% Pilot reconstruction
fistaMaxIter = 100;
lambda0 = 1;
[pilot, ssimPilot,rmsePilot ] = reconstructPilot(y,angleArr,dim,I_0,validIm,methodName,fistaMaxIter,lambda0);
name = sprintf('%s/pilotn_%d_angles_%d_I_0.mat',outDirectory,numAngles,I_0);
save(name,'pilot');
pilot = load(name);
pilot = pilot.pilot;
figure('Name','Pilot_validIm','NumberTitle','off');imshow(pilot,[]);


%% Generate HQ eigen space in spatial domain from templateNos %%
[eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior(pilot,ratio,templateNos, outDirectory);

%% project pilot reconstruction of validIm on this eig space %%
% meanImage = reshape(meanTemplateSpatial,dim);
pilotOnEigSpace = meanTemplateSpatial + (eigenVecsSpatial*alpha);
pilotOnEigSpace = reshape(pilotOnEigSpace,dim);
pilotOnEigSpace_norm = mat2gray(pilotOnEigSpace);
figure('Name','PilotProjOnEigSpace','NumberTitle','off');
imshow(pilotOnEigSpace_norm);

%% calculate residual error for validation image
residual = abs(pilot-pilotOnEigSpace);
figure('Name','residual_validImage','NumberTitle','off');imshow(residual,[]);

residual_weiner = wiener2(residual,[12 12]);
figure('Name','residual_weiner','NumberTitle','off');imshow(residual_weiner,[]);
residual_cut = residual_weiner(:,101:200);

% residual_norm =  mat2gray(residual_weiner);
T = adaptthresh(residual_weiner,0,'ForegroundPolarity','bright','NeighborhoodSize',[37 37] ,'Statistic','mean');
residual_bin = imbinarize(residual_weiner,T);
figure('Name','imbinarize_adaptive residual','NumberTitle','off');imshow(residual_bin);

% residual_clust = bwareaopen(residual_bin, 3); % remove clusters of size <5
% figure('Name','residual_clust','NumberTitle','off');imshow(residual_clust);

% residual_cut = residual_bin(:,101:200);
SE = strel('sphere',3);
residual_cut_close = imclose(residual_bin,SE);%residual_cut, SE);
figure('Name','imclose','NumberTitle','off');imshow(residual_cut_close);

residual_clust = bwareaopen(residual_cut_close, 5); % remove clusters of size <5
figure('Name','residual_clust','NumberTitle','off');imshow(residual_clust);

mask = residual_clust(:,101:200);
%{
%% Select Region of change 
roiwindow = CROIEditor(residual_cut);
% wait for roi to be assigned
waitfor(roiwindow,'roi');
if ~isvalid(roiwindow)
    disp('you closed the window without applying a ROI, exiting...');
    exit();
end

% get ROI information, like binary mask, labelled ROI, and number of
% ROIs defined
[mask, labels, n] = roiwindow.getROIData;
delete(roiwindow);
% addlistener(roiwindow,'MaskDefined',@myfun);
% mask = roiwindow.roi;
%}
%% show mask
% figure;imshow(mask);
name = sprintf('./result/mat/adaptive_mask1st_%d_template.mat',validImNo);
% save(name,'mask');
logicalMask = load(name);
logicalMask = logicalMask.mask;
figure('Name','logicalMask','NumberTitle','off');
imshow(logicalMask);
mask_cut = logicalMask;
figure;imshow(mask_cut);
%% Train SVM 
addpath('libsvm-3.24/matlab');
patchWidth = 10;
patchHeight = 10;
center = 5;
dim = size(residual_cut);
[positiveLabels,positivePatchVector,negativeLabels,negativePatchVector] = create_training_dataset(dim,...
                                                                           residual_cut, mask_cut,...
                                                                           patchWidth,patchHeight,center);
Xtrain = cat(2,positivePatchVector,negativePatchVector);
Xtrain = Xtrain';
name = sprintf('./result/mat/Xtrain_%d.mat',validImNo);
save(name,'Xtrain');
Ytrain = cat(2,positiveLabels,negativeLabels);
Ytrain = Ytrain';
name = sprintf('./result/mat/Ytrain_%d.mat',validImNo);
save(name,'Ytrain');
% SVMModel = fitcsvm(Xtrain',Ytrain,'KernelFunction','linear',...
%           'Standardize',true,'OptimizeHyperparameters','all');
diary crossvalidation_log
bestcv = 0;
for c = [1000]
  for g = [0.1 1 10 100]
    cmd = ['-v 10 -h 0 -c ', num2str(c), ' -g ', num2str(g)];
    cv = svmtrain(Ytrain, Xtrain, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc =c; bestg = g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c, g, cv, bestc, bestg, bestcv);
  end
end
diary off

model = svmtrain(Ytrain, Xtrain, '-c 1 -g 1 -v 10');%-c 10000 -g 10 -v 10

%%  Test on validation Image
Xtest = create_test_input(dim,residual_cut, patchWidth,patchHeight);
Xtest = Xtest';
Ytest =rand(size(Ytrain));
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(Ytest, Xtest, model);
% numPatches = (dim(1)/patchWidth) *(dim(2)/patchHeight);
detectedInlier = zeros(size(mask_cut));
patchNumber = 0;
for i = 1:dim(1)-patchWidth+1  
    for j = 1:dim(2)-patchHeight+1 
        startWidth = j;
        endWidth = startWidth + patchWidth-1;
        startHeight = i;
        endHeight = startHeight + patchHeight-1;
        patchNumber = patchNumber + 1;
        detectedInlier(startHeight+center,startWidth+center) = predict_label_L(patchNumber);
%         detectedInlier(startHeight:endHeight,startWidth:endWidth) = predict_label_L(patchNumber);
    end
end
figure('Name','Detected Inlier-Validation','NumberTitle','off');
imshow(detectedInlier,[]);

resultStitched_svm = detectedInlier.*pilot(:,101:200) + (1-detectedInlier).*pilotOnEigSpace(:,101:200);
figure('Name','Validation resultStitched_svm','NumberTitle','off');
imshow(resultStitched_svm,[]);

%% Test on unseen Image 
ImAddr = sprintf('/home/yadnyesh/Desktop/seminar/fewviews/okra/data_okra/okra6_okra%d_reg_450views_fdk.mat',testImNo); % test image
test_data = load(ImAddr);
test_data = test_data.FDK;
testIm= double(test_data(20:end-19,20:end-19,30));
name = sprintf('./result/mat/testIm_template_%d.mat',testImNo);
% save(name,'testIm');
testIm = load(name);
testIm = testIm.testIm;
testIm = testIm - min(testIm(:));
testIm = testIm./max(testIm(:));
testIm2 = testIm;
figure('Name','Test_Image','NumberTitle','off');imshow(testIm);
%% test measurements
testIm = testIm./ratio;
I_0 = 2000;
[y_test] = generateMeasurements(testIm,I_0, angleArr, gaussianNoisePercent);
%% pilot reconstruction of test image
fistaMaxIter = 100;
lambda = 1;
dim=size(testIm);
% [pilot_test] = reconstructPilot(y_test,angleArr,dim,I_0,testIm,methodName,fistaMaxIter,lambda);
name = sprintf('./result/mat/pilot_test_%d_angles_%d_I_0.mat',numAngles,I_0);
% save(name,'pilot_test');
pilot_test = load(name);
pilot_test = pilot_test.pilot_test;
figure('Name','Pilot_testIm','NumberTitle','off');imshow(pilot_test,[]);

%% project pilot reconstruction of test Image on HQ eig space of 3,4,5 %%
[eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior(pilot_test,ratio,templateNos);
% meanImage = reshape(meanTemplateSpatial,dim);
pilot_testOnEigSpace = meanTemplateSpatial + (eigenVecsSpatial*alpha);
pilot_testOnEigSpace = reshape(pilot_testOnEigSpace,dim);
pilot_testOnEigSpace_norm = mat2gray(pilot_testOnEigSpace);
figure('Name','Pilot_Test_ProjOnEigSpace','NumberTitle','off');
imshow(pilot_testOnEigSpace_norm);

%% calculate residual error for test image
residual_test = abs(pilot_test - pilot_testOnEigSpace);
figure('Name','residual_TestImage','NumberTitle','off');imshow(residual_test,[]);
residual_test_weiner = wiener2(residual_test,[12 12]);
residual_cut_test = residual_test_weiner(:,101:200);
% figure;imshow(residual_test_weiner,[]);
% residual_test_norm = mat2gray(residual_test_weiner);
% residual_test_norm = residual_test_norm(:,101:200);
figure('Name','residual_test_weiner','NumberTitle','off');imshow(residual_cut_test,[]);

%%  Test on test Image
dim = size(residual_cut_test);
Xtest = create_test_input(dim,residual_cut_test, patchWidth,patchHeight);
Xtest = Xtest';
name = sprintf('./result/mat/Xtest_%d.mat',testImNo);
save(name,'Xtest');
Ytest = rand(size(Xtest,1) , 1);
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(Ytest, Xtest, model);
% numPatches = (dim(1)/patchWidth) *(dim(2)/patchHeight);
detectedInlier = zeros(size(residual_cut_test));
patchNumber = 0;
for i = 1:dim(1)-patchWidth+1  
    for j = 1:dim(2)-patchHeight+1 
        startWidth = j;
        endWidth = startWidth + patchWidth-1;
        startHeight = i;
        endHeight = startHeight + patchHeight-1;
        patchNumber = patchNumber + 1;
%         detectedInlier(startHeight:endHeight,startWidth:endWidth) = Ytest(patchNumber);
        detectedInlier(startHeight+center,startWidth+center) = predict_label_L(patchNumber);
    end
end
figure('Name','Detected Inlier-Test','NumberTitle','off');
imshow(detectedInlier,[]);

SE = strel('sphere',2);
InlierClosed = imclose(detectedInlier,SE);
figure('Name','InlierClosed','NumberTitle','off');
imshow(InlierClosed,[]);

residual_clust_test = bwareaopen(InlierClosed, 5); % remove clusters of size <5
figure('Name','residual_clust_test','NumberTitle','off');imshow(residual_clust_test);

weights = zeros(300,300);
weights(:,101:200)=residual_clust_test;

name = sprintf('./result/mat/mask_predicted_test_%d_template.mat',testImNo);
% save(name,'residual_clust_test');
residual_clust_test= load(name); 
residual_clust_test = residual_clust_test.residual_clust_test;
resultStitched_svm = weights.*pilot_test + (1-weights).*pilot_testOnEigSpace;
figure('Name','Test resultStitched_svm','NumberTitle','off');
imshow(resultStitched_svm,[]);

%% Using weights in objective function optimization

figure('Name','weights','NumberTitle','off');imshow(weights);
fistaMaxIter = 100;
numCycles = 5;
dim=size(weights);
rel_tol = 0.001; % relative target duality gap
HighQualityEigenVecs = eigenVecsSpatial;
HighQualityMeanTemplate = meanTemplateSpatial;

weights_bar = imcomplement(weights);
figure;imshow(weights_bar);

lambda1 = 1200;% 900 1200];
%lambda1List = [900];% 1000 1200];
%         for lambda1Iter = 1:length(lambda1List)

[resultWeightedPrior1] = weightedLowDose_pca(y_test,dim,angleArr,I_0,weights_bar,HighQualityEigenVecs, ...
                                                                        HighQualityMeanTemplate,alpha,lambda1,fistaMaxIter);
%figure;imshow([resultWeightedPrior1],[]);impixelinfo;

figure('Name','weighted_recon_svmWeights','NumberTitle','off');
imshow(resultWeightedPrior1,[]);


%% Irradiation
projectionsW = radon(weights,angleArr);
figure('Name','projectionsW','NumberTitle','off');imshow(projectionsW,[]);
projectionsW_binary = imbinarize(projectionsW);
figure('Name','projectionsW_binary','NumberTitle','off');imshow(projectionsW_binary,[]);
projectionsW_binary_fbp = iradon(projectionsW_binary,angleArr,'linear','Cosine');
dim = size(weights);
projectionsW_binary_fbp = projectionsW_binary_fbp(2:2+dim(1)-1,2:2+dim(2)-1);
figure('Name','projectionsW_binary_fbp','NumberTitle','off');imshow(projectionsW_binary_fbp,[]);

I_0_high = 5000;
projectionsHighDose = I_0_high*exp(-radon(testIm,angleArr));
combinedProjections = zeros(size(projectionsW_binary));
combinedProjections(projectionsW_binary==1) = projectionsHighDose(projectionsW_binary==1);
y_matrix = reshape(y_test,[(size(y_test,1))/numAngles numAngles]);
combinedProjections(projectionsW_binary==0) = y_matrix(projectionsW_binary==0);
figure('Name','combinedProjections ','NumberTitle','off');imshow(combinedProjections,[]);
combinedProjections = combinedProjections(:);
I_0_matrix = zeros(size(projectionsW_binary));
I_0_matrix(projectionsW_binary==1) = I_0_high;
I_0_matrix(projectionsW_binary==0) = I_0;
figure('Name','I0 matrix','NumberTitle','off');imshow(I_0_matrix,[]);

fistaMaxIter = 100; lambda = 1;
reconIrradiation = rescaled_non_linear_LSCS_I0_matrix(combinedProjections,dim,angleArr,I_0_matrix,...
                                                                                    fistaMaxIter,lambda);
figure('Name','reconIrradiation','NumberTitle','off');imshow(reconIrradiation,[]);

 %% Irradiation for a bigger region

SE1 = strel('disk', 5);
W = imdilate(weights,SE1);
figure('Name','W dialate','NumberTitle','off');imshow(W,[]);
% SE2 = strel('disk', 1);
% W = imerode(W,SE2);
% figure;imshow(W,[]);

projectionsW = radon(W,angleArr);
figure('Name','projectionsW','NumberTitle','off');imshow(projectionsW,[]);
projectionsW_binary = imbinarize(projectionsW);
figure('Name','projectionsW_binary','NumberTitle','off');imshow(projectionsW_binary,[]);
projectionsW_binary_fbp = iradon(projectionsW_binary,angleArr,'linear','Cosine');
dim = size(W);
projectionsW_binary_fbp = projectionsW_binary_fbp(2:2+dim(1)-1,2:2+dim(2)-1);
figure('Name','projectionsW_binary_fbp','NumberTitle','off');imshow(projectionsW_binary_fbp,[]);

I_0_high = 4000;
projectionsHighDose = I_0_high*exp(-radon(testIm,angleArr));
combinedProjections = zeros(size(projectionsW_binary));
combinedProjections(projectionsW_binary==1) = projectionsHighDose(projectionsW_binary==1);
y_matrix = reshape(y_test,[(size(y_test,1))/numAngles numAngles]);
combinedProjections(projectionsW_binary==0) = y_matrix(projectionsW_binary==0);
figure('Name','combinedProjections ','NumberTitle','off');imshow(combinedProjections,[]);
combinedProjections = combinedProjections(:);
I_0_matrix = zeros(size(projectionsW_binary));
I_0_matrix(projectionsW_binary==1) = I_0_high;
I_0_matrix(projectionsW_binary==0) = I_0;
figure('Name','I0 matrix','NumberTitle','off');imshow(I_0_matrix,[]);

%reconInlier = non_linear_LSCS_I0_matrix(combinedProjections,dim,idx_high,I_0_matrix);
fistaMaxIter = 100; lambda=1;
reconIrradiation = rescaled_non_linear_LSCS_I0_matrix(combinedProjections,dim,angleArr,I_0_matrix,fistaMaxIter,lambda);
figure('Name','reconIrradiation','NumberTitle','off');imshow(reconIrradiation,[]); 

% result_stitched = weights.*reconIrradiation + (1-weights).*pilot_testOnEigSpace;
% figure('Name','Final_stitched','NumberTitle','off');
% imshow(result_stitched,[]);

