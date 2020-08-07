close all; clc;clear all;
numAngles = 360;
reconMethod = 4;
methodName = 'Rescaled_Non_Linear_Least_Squares_CS';
gaussianNoisePercent = 1; 
testImNo = 3;
validImNo = 4;
templateNos = [5,6,7];

%% Traning Part %%
% prior templates: 3,4,5 validation Image: 6

outDirectory = sprintf('./result/mathod_manual/svm/');
mkdir(outDirectory);

name = sprintf('../data/validIm_template_4.mat');
validIm = load(name);
validIm = validIm.validIm;
figure('Name','Valid_Image','NumberTitle','off');imshow(validIm,[]);
name = sprintf('%s/validIm.png',outDirectory);
validIm = mat2gray(validIm);
imwrite(validIm,name);
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
name = sprintf('%s/validIm_pilot.png',outDirectory);
imwrite(mat2gray(pilot),name);

%% Generate HQ eigen space in spatial domain from templateNos %%
[eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior(pilot,ratio,templateNos, outDirectory);

%% project pilot reconstruction of validIm on this eig space %%
% meanImage = reshape(meanTemplateSpatial,dim);
pilotOnEigSpace = meanTemplateSpatial + (eigenVecsSpatial*alpha);
pilotOnEigSpace = reshape(pilotOnEigSpace,dim);
pilotOnEigSpace_norm = mat2gray(pilotOnEigSpace);
figure('Name','PilotProjOnEigSpace','NumberTitle','off');
imshow(pilotOnEigSpace_norm);
name = sprintf('%s/validIm_pilotProj.png',outDirectory);
imwrite(pilotOnEigSpace_norm,name);
%% calculate residual error for validation image
residual = abs(pilot-pilotOnEigSpace);
figure('Name','residual_validImage','NumberTitle','off');imshow(residual,[]);

residual_weiner = wiener2(residual,[12 12]);
figure('Name','residual_weiner','NumberTitle','off');imshow(residual_weiner,[]);
name = sprintf('%s/residual_valid_weiner.png',outDirectory);
imwrite(mat2gray(residual_weiner),name);

residual_cut = residual_weiner(:,101:200);
residual_cut = mat2gray(residual_cut);
figure('Name','residual_cut','NumberTitle','off');imshow(residual_cut);
% %% create y for model
% [inlier, inlierProj, binaryInlierProj] = genGroundTruthInlier(angleArr, eigenVecsSpatial,meanTemplateSpatial,y_noNoise,dim,outDirectory);
% figure('Name','inlier','NumberTitle','off');imshow(inlier,[]);
% name = sprintf('%s/Actual_inlier_valid.png',outDirectory);
% imwrite(mat2gray(inlier),name);
% binaryInlier = imbinarize(mat2gray(inlier), 0.3);
% figure('Name','binaryinlier','NumberTitle','off');imshow(binaryInlier);
% SE = strel('sphere',3);
% inlierclosed = imclose(binaryInlier,SE);
% figure('Name','binaryinlierClosed','NumberTitle','off');imshow(inlierclosed);
% name = sprintf('%s/bin_inlier_pilot_train.png',outDirectory);
% imwrite(inlierclosed,name);
% mask_cut = inlierclosed(:,101:200);
% figure('Name','mask_cut','NumberTitle','off');imshow(mask_cut);
% %% end create y for model

% residual_norm =  mat2gray(residual_weiner);
% T = adaptthresh(residual_weiner,0,'ForegroundPolarity','bright','NeighborhoodSize',[37 37] ,'Statistic','mean');
% residual_bin = imbinarize(residual_weiner,T);
% figure('Name','imbinarize_adaptive residual','NumberTitle','off');imshow(residual_bin);
% 
% % residual_clust = bwareaopen(residual_bin, 3); % remove clusters of size <5
% % figure('Name','residual_clust','NumberTitle','off');imshow(residual_clust);
% 
% % residual_cut = residual_bin(:,101:200);
% SE = strel('sphere',3);
% residual_cut_close = imclose(residual_bin,SE);%residual_cut, SE);
% figure('Name','imclose','NumberTitle','off');imshow(residual_cut_close);
% 
% residual_clust = bwareaopen(residual_cut_close, 5); % remove clusters of size <5
% figure('Name','residual_clust','NumberTitle','off');imshow(residual_clust);
% 
% mask = residual_clust(:,101:200);

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

%% show mask
figure;imshow(mask);
full_mask = zeros(300,300);
full_mask(:,101:200) = mask;
figure;imshow(full_mask);
name = sprintf('%s/manual_mask_%d_template.mat',outDirectory,validImNo);
save(name,'full_mask');
logicalMask = load(name);
logicalMask = logicalMask.full_mask;
figure('Name','logicalMask','NumberTitle','off');imshow(logicalMask);
mask_cut = logicalMask;
figure;imshow(mask_cut);

name = sprintf('%s/bin_inlier_pilot_train.png',outDirectory);
imwrite(mask_cut,name);
mask_cut = mask_cut(:,101:200);
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
name = sprintf('%s/Xtrain_%d.mat',outDirectory, validImNo);
save(name,'Xtrain');
Ytrain = cat(2,positiveLabels,negativeLabels);
Ytrain = Ytrain';
name = sprintf('%s/Ytrain_%d.mat',outDirectory,validImNo);
save(name,'Ytrain');
% SVMModel = fitcsvm(Xtrain',Ytrain,'KernelFunction','linear',...
%           'Standardize',true,'OptimizeHyperparameters','all');
% diary crossvalidation_log
% bestcv = 0;
% for c = [0.1 1 10]
%   for g = [0.1 1 10 ]
%     cmd = ['-v 10 -h 0 -c ', num2str(c), ' -g ', num2str(g)];
%     cv = svmtrain(Ytrain, Xtrain, cmd);
%     if (cv >= bestcv),
%       bestcv = cv; bestc =c; bestg = g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c, g, cv, bestc, bestg, bestcv);
%   end
% end
% diary off

model = svmtrain(Ytrain, Xtrain, '-c 2 -g 0.5');%-c 10000 -g 10 -v 10
name = sprintf('%s/trainedmodel.mat',outDirectory);
save(name,'model');

%%  Test on validation Image
dim = size(residual_cut);
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
testIm = load('../data/testIm_template_3.mat');
testIm = testIm.testIm;
testIm = mat2gray(testIm);
figure('Name','Test_Image','NumberTitle','off');imshow(testIm,[]);
name = sprintf('%s/testIm.png',outDirectory);
imwrite(mat2gray(testIm),name);
%% test measurements
testIm = testIm./ratio;
I_0 = 2000;
[y_test, y_test_noNoise] = generateMeasurements(testIm,I_0, angleArr, gaussianNoisePercent);

%% pilot reconstruction of test image
fistaMaxIter = 100;
lambda0 = 1;
dim=size(testIm);
[pilot_test] = reconstructPilot(y_test,angleArr,dim,I_0,testIm,methodName,fistaMaxIter,lambda0);
name = sprintf('%s/pilot_test_%d_angles_%d_I_0.mat',outDirectory,numAngles,I_0);
% save(name,'pilot_test');
pilot_test = load(name);
pilot_test = pilot_test.pilot_test;
figure('Name','Pilot_testIm','NumberTitle','off');imshow(pilot_test,[]);
name = sprintf('%s/testIm_pilot.png',outDirectory);
imwrite(mat2gray(pilot_test),name);

%% project pilot reconstruction of test Image on HQ eig space of 3,4,5 %%
[eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior(pilot_test,ratio,templateNos,outDirectory);
% meanImage = reshape(meanTemplateSpatial,dim);
pilot_testOnEigSpace = meanTemplateSpatial + (eigenVecsSpatial*alpha);
pilot_testOnEigSpace = reshape(pilot_testOnEigSpace,dim);
pilot_testOnEigSpace_norm = mat2gray(pilot_testOnEigSpace);
figure('Name','Pilot_Test_ProjOnEigSpace','NumberTitle','off');imshow(pilot_testOnEigSpace_norm);
name = sprintf('%s/testIm_pilotProj.png',outDirectory);
imwrite(pilot_testOnEigSpace_norm,name);

%% calculate residual error for test image
residual_test = abs(pilot_test - pilot_testOnEigSpace);
figure('Name','residual_TestImage','NumberTitle','off');imshow(residual_test,[]);
residual_test_weiner = wiener2(residual_test,[12 12]);
residual_cut_test = mat2gray(residual_test_weiner);
name = sprintf('%s/residual_test_weiner.png',outDirectory);
imwrite(mat2gray(residual_test_weiner),name);
residual_cut_test = residual_cut_test(:,101:200);
% figure;imshow(residual_test_weiner,[]);
% residual_test_norm = mat2gray(residual_test_weiner);
% residual_test_norm = residual_test_norm(:,101:200);
figure('Name','residual_test_weiner','NumberTitle','off');imshow(residual_cut_test,[]);

%%  Testing trained SVM model
dim = size(residual_cut_test);
Xtest = create_test_input(dim,residual_cut_test, patchWidth,patchHeight);
Xtest = Xtest';
name = sprintf('%s/Xtest_%d.mat',outDirectory,testImNo);
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
full_detectedInlier = zeros(300,300);
full_detectedInlier(:,101:200)=detectedInlier;
name = sprintf('%s/bin_inlier_test_detected.png',outDirectory);
imwrite(full_detectedInlier,name);

SE = strel('sphere',5);
InlierClosed = imclose(detectedInlier,SE);
figure('Name','InlierClosed','NumberTitle','off');imshow(InlierClosed,[]);
SE = strel('sphere',1);
% InlierCloseddilate = imdilate(InlierClosed,SE);
% figure('Name','InlierCloseddilate','NumberTitle','off');imshow(InlierCloseddilate,[]);
residual_clust_test = bwareaopen(InlierClosed, 5); % remove clusters of size <5
figure('Name','residual_clust_test','NumberTitle','off');imshow(residual_clust_test);

weights = zeros(300,300);
weights(:,101:200)=residual_clust_test;
name = sprintf('%s/final_weights_detected.png',outDirectory);
imwrite(weights,name);

name = sprintf('%s/mask_predicted_test_%d_template.mat',outDirectory,testImNo);
save(name,'weights');
weights= load(name); 
weights = weights.weights;
resultStitched_svm = weights.*pilot_test + (1-weights).*pilot_testOnEigSpace;
figure('Name','Test resultStitched_svm','NumberTitle','off');
imshow(resultStitched_svm,[]);
name = sprintf('%s/resultStitched_svm.png',outDirectory);
imwrite(mat2gray(resultStitched_svm),name);
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

lambda0 = 1; 
lambda1 = 1200;% 900 1200];
%lambda1List = [900];% 1000 1200];
%         for lambda1Iter = 1:length(lambda1List)

[resultWeightedPrior1] = weightedLowDose_pca(y_test,dim,angleArr,I_0,weights_bar,HighQualityEigenVecs, ...
                                                                        HighQualityMeanTemplate,alpha,lambda0,lambda1,fistaMaxIter);
%figure;imshow([resultWeightedPrior1],[]);impixelinfo;

figure('Name','weighted_recon_svmWeights','NumberTitle','off');
imshow(resultWeightedPrior1,[]);
name = sprintf('%s/resultWeightedPrior_test.png',outDirectory);
imwrite(mat2gray(resultWeightedPrior1),name);
name = sprintf('%s/resultWeightedPrior_test.mat',outDirectory);
save(name,'resultWeightedPrior1');

%% Irradiation
SE = strel('sphere',1);
weights2 = imdilate(weights,SE);
figure('Name','dilate_weights','NumberTitle','off');imshow(weights2);
projectionsW = radon(weights2,angleArr);
% figure('Name','projectionsW','NumberTitle','off');imshow(projectionsW,[]);
projectionsW_binary = imbinarize(projectionsW);
figure('Name','projectionsW_binary','NumberTitle','off');imshow(projectionsW_binary,[]);
projectionsW_binary_fbp = iradon(projectionsW_binary,angleArr,'linear','Cosine');
dim = size(weights2);
projectionsW_binary_fbp = projectionsW_binary_fbp(2:2+dim(1)-1,2:2+dim(2)-1);
figure('Name','projectionsW_binary_fbp','NumberTitle','off');imshow(projectionsW_binary_fbp,[]);

I_0_high = 5000;
projectionsHighDose = I_0_high*exp(-radon(testIm,angleArr));
combinedProjections = zeros(size(projectionsW_binary));
combinedProjections(projectionsW_binary==1) = projectionsHighDose(projectionsW_binary==1);
y_matrix = reshape(y_test,[(size(y_test,1))/numAngles numAngles]);
combinedProjections(projectionsW_binary==0) = y_matrix(projectionsW_binary==0);
% figure('Name','combinedProjections ','NumberTitle','off');imshow(combinedProjections,[]);
combinedProjections = combinedProjections(:);
I_0_matrix = zeros(size(projectionsW_binary));
I_0_matrix(projectionsW_binary==1) = I_0_high;
I_0_matrix(projectionsW_binary==0) = I_0;
% figure('Name','I0 matrix','NumberTitle','off');imshow(I_0_matrix,[]);

fistaMaxIter = 100; lambda0 = 0.01;
reconIrradiation = rescaled_non_linear_LSCS_I0_matrix(combinedProjections,dim,angleArr,I_0_matrix,...
                                                                                    fistaMaxIter,lambda0);
figure('Name','reconIrradiation','NumberTitle','off');imshow(reconIrradiation,[]);
name = sprintf('%s/resultWeightedPrior_irradiat_test_lamb_%f.png',outDirectory,lambda0);
imwrite(mat2gray(reconIrradiation),name);
name = sprintf('%s/resultWeightedPrior_irradiat_test_lamb_%f.mat',outDirectory,lambda0);
save(name,'reconIrradiation');

result_stitched = weights2.*reconIrradiation + (1-weights2).*pilot_testOnEigSpace;
figure('Name','Final_stitched','NumberTitle','off');
imshow(result_stitched,[]);
name = sprintf('%s/final_stitched_lamb_%f.png',outDirectory,lambda0);
imwrite(mat2gray(result_stitched),name);

%% ssim calculation
test = mat2gray(testIm);
testpilot = mat2gray(pilot_test);
test_wted_recon = mat2gray(resultWeightedPrior1);imshow(test_wted_recon);
test_irrad = mat2gray(reconIrradiation);imshow(test_irrad);

%overall ssim
ssim_pilot_ov = ssim(testpilot,test);
ssim_wtedrecon_ov = ssim(test_wted_recon,test);
ssim_irrad_ov = ssim(test_irrad,test);
%overall rmse
rmse_pilot_ov = sqrt(immse(testpilot,test));
rmse_wtedrecon_ov = sqrt(immse(test_wted_recon,test));
rmse_irrad_ov = sqrt(immse(test_irrad,test));
%ROI ssim
right = 128:143;  %55:72; horizontal pixels in image
left = 160:170;  %78:93; vertical pixels in image
% iptsetpref('ImshowBorder','tight');
fh = figure;imshow(residual_test_weiner,[]); hold on;
rectangle('Position',[128, 160, 16, 11],'EdgeColor','g','LineWidth',1);  %[52, 78, 21, 16]
name = sprintf('%s/roi2.png',outDirectory);
saveas(fh,name);

roi_test = test(left,right);%figure;imshow(roi_test,[]);
roi_pilot = testpilot(left,right);
roi_wtedrecon = test_wted_recon(left,right);
roi_irrad = test_irrad(left,right);

ssim_pilot_roi = ssim(roi_pilot,roi_test);
ssim_wtedrecon_roi= ssim(roi_wtedrecon,roi_test);
ssim_irrad_roi = ssim(roi_irrad,roi_test);
%ROI rmse
rmse_pilot_roi = sqrt(immse(roi_pilot,roi_test));
rmse_wtedrecon_roi = sqrt(immse(roi_wtedrecon,roi_test));
rmse_irrad_roi = sqrt(immse(roi_irrad,roi_test));
name = sprintf('%s/ssim.txt',outDirectory);
fileID = fopen(name, 'a');
fprintf(fileID,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',ssim_pilot_ov,...
    ssim_wtedrecon_ov,ssim_irrad_ov,rmse_pilot_ov,rmse_wtedrecon_ov,rmse_irrad_ov,...
    ssim_pilot_roi,ssim_wtedrecon_roi,ssim_irrad_roi,rmse_pilot_roi,...
    rmse_wtedrecon_roi,rmse_irrad_roi );
fclose(fileID);