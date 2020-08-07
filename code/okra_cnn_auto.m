clear all;
testImNo=3;
validImNo = 4;
templateNos = [5,6,7];
outDirectory = sprintf('./result/matnew_auto/cnn');
mkdir(outDirectory);

name = sprintf('%s/Xtrain_%d.mat',outDirectory,validImNo);
Xtrain = load(name);
Xtrain = Xtrain.Xtrain;
name = sprintf('%s/Ytrain_%d.mat',outDirectory,validImNo);
Ytrain = load(name);
Ytrain = Ytrain.Ytrain;
ind = randperm(size(Xtrain,1));
Xtrain = Xtrain(ind,:);
Ytrain = Ytrain(ind,:);
N = size(Xtrain,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(0.9*N)) = true;     
tf = tf(randperm(N));   % randomise order
Xtrain2 = Xtrain(tf,:); 
Xvalid = Xtrain(~tf,:);
Xtrain = reshape( Xtrain2', 10,10,1, []);
Xvalid = reshape( Xvalid', 10,10,1, []);
Ytrain2 = Ytrain(tf,:); 
Yvalid = Ytrain(~tf,:);
Yvalid = categorical(Yvalid);
Ytrain = categorical(Ytrain2);


layers = [
    imageInputLayer([10 10 1])
    
    convolution2dLayer(3,5,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,5,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2,'Padding','same')
    
%     convolution2dLayer(3,3,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
    
    fullyConnectedLayer(40)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',3, ...
    'Shuffle','every-epoch', ...
    'ValidationData', {Xvalid,Yvalid}, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(Xtrain,Ytrain,layers,options);

YPred = classify(net,Xvalid);
accuracy = sum(YPred == Yvalid)/numel(Yvalid);

name = sprintf('%s/Xtest_%d.mat',outDirectory,testImNo);
Xtest = load(name);
Xtest = Xtest.Xtest;
Xtest = reshape( Xtest', 10,10,1, []);

[Ytest,scores] = classify(net,Xtest);
x = [0,1];
Ytest = x(Ytest);
inlier = zeros(300,100);
dim = [300,100];
patchWidth =10;
patchHeight =10;
center =5;
patchNumber=0;
for i = 1:dim(1)-patchWidth+1  
    for j = 1:dim(2)-patchHeight+1 
        startWidth = j;
        endWidth = startWidth + patchWidth-1;
        startHeight = i;
        endHeight = startHeight + patchHeight-1;
        patchNumber = patchNumber + 1;
        inlier(startHeight+center,startWidth+center) = Ytest(patchNumber);
%         detectedInlier(startHeight:endHeight,startWidth:endWidth) = predict_label_L(patchNumber);
    end
end
figure('Name','Detected Inlier CNN','NumberTitle','off');imshow(inlier,[]);
full_inlier = zeros(300,300);
full_inlier(:,101:200) = inlier;
% SE =  strel('sphere',1);
% weights_closed = imclose(weights,SE);
% figure('Name','ClosedWeights','NumberTitle','off');imshow(weights_closed,[]);
% name = sprintf('%s/cnnresult_weightsmap_%d.mat',outDirectory,testImNo);
% save(name,'weights');

%% reconstruction part
SE = strel('sphere',5);
InlierClosed = imclose(full_inlier,SE);
figure('Name','InlierClosed','NumberTitle','off');imshow(InlierClosed,[]);
% SE = strel('sphere',1);
% InlierCloseddilate = imdilate(InlierClosed,SE);
% figure('Name','InlierCloseddilate','NumberTitle','off');imshow(InlierCloseddilate,[]);
residual_clust_test = bwareaopen(InlierClosed, 5); % remove clusters of size <5
figure('Name','residual_clust_test','NumberTitle','off');imshow(residual_clust_test);

weights = residual_clust_test;
name = sprintf('%s/cnnfinal_weights_detected.png',outDirectory);
imwrite(weights,name);
name = sprintf('%s/cnnmask_predicted_test_%d_template.mat',outDirectory,testImNo);
save(name,'weights');
weights= load(name); 
weights = weights.weights;
resultStitched_svm = weights.*pilot_test + (1-weights).*pilot_testOnEigSpace;
figure('Name','Test resultStitched_cnn','NumberTitle','off');
imshow(resultStitched_svm,[]);
name = sprintf('%s/resultStitched_cnn.png',outDirectory);
imwrite(mat2gray(resultStitched_svm),name);

%% Test on unseen Image 
ImAddr = sprintf('/home/yadnyesh/Desktop/seminar/fewviews/2D/okra/data_okra/okra6_okra%d_reg_450views_fdk.mat',testImNo); % test image
test_data = load(ImAddr);
test_data = test_data.FDK;
testIm= double(test_data(20:end-19,20:end-19,30));
testIm = mat2gray(testIm);
figure('Name','Test_Image','NumberTitle','off');imshow(testIm,[]);
name = sprintf('%s/testIm.png',outDirectory);
imwrite(mat2gray(testIm),name);
%% test measurements
ratio =100; numAngles=360;gaussianNoisePercent = 1; 
angleArr = linspace(0,180,numAngles+1);
angleArr = angleArr(1:numAngles);
testIm = testIm./ratio;
I_0 = 2000;
[y_test, y_test_noNoise] = generateMeasurements(testIm,I_0, angleArr, gaussianNoisePercent);
%% pilot recon test
name = sprintf('%s/pilot_test_360_angles_2000_I_0.mat',outDirectory);
pilot_test = load(name);
pilot_test = pilot_test.pilot_test;
%% project on HQ eigspace
dim = size(pilot_test);
[eigenVecsSpatial,meanTemplateSpatial,alpha] = genHighQualityPrior(pilot_test,ratio,templateNos,outDirectory);
% meanImage = reshape(meanTemplateSpatial,dim);
pilot_testOnEigSpace = meanTemplateSpatial + (eigenVecsSpatial*alpha);
pilot_testOnEigSpace = reshape(pilot_testOnEigSpace,dim);
pilot_testOnEigSpace_norm = mat2gray(pilot_testOnEigSpace);
figure('Name','Pilot_Test_ProjOnEigSpace','NumberTitle','off');imshow(pilot_testOnEigSpace_norm);
%% Using weights in objective function optimization

figure('Name','weights','NumberTitle','off');imshow(weights);
fistaMaxIter = 100;
numCycles = 5;
dim=size(weights);
% rel_tol = 0.001; % relative target duality gap
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
name = sprintf('%s/cnnresultWeightedPrior_test.png',outDirectory);
imwrite(mat2gray(resultWeightedPrior1),name);
name = sprintf('%s/cnnresultWeightedPrior_test.mat',outDirectory);
save(name,'resultWeightedPrior1');

%% Irradiation
SE = strel('sphere',5);
weights2 = imclose(weights,SE);
figure('Name','dilate_weights','NumberTitle','off');imshow(weights2);
projectionsW = radon(weights2,angleArr);
figure('Name','projectionsW','NumberTitle','off');imshow(projectionsW,[]);
projectionsW_binary = imbinarize(projectionsW);
figure('Name','projectionsW_binary','NumberTitle','off');imshow(projectionsW_binary,[]);
projectionsW_binary_fbp = iradon(projectionsW_binary,angleArr,'linear','Cosine');
dim = size(weights2);
projectionsW_binary_fbp = projectionsW_binary_fbp(2:2+dim(1)-1,2:2+dim(2)-1);
% figure('Name','projectionsW_binary_fbp','NumberTitle','off');imshow(projectionsW_binary_fbp,[]);

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

fistaMaxIter = 100; lambda0 = 0.001;
reconIrradiation = rescaled_non_linear_LSCS_I0_matrix(combinedProjections,dim,angleArr,I_0_matrix,...
                                                                                    fistaMaxIter,lambda0);
figure('Name','reconIrradiation','NumberTitle','off');imshow(reconIrradiation,[]);
name = sprintf('%s/cnnresultWeightedPrior_irradiat_test_lamb_%f.png',outDirectory,lambda0);
imwrite(mat2gray(reconIrradiation),name);
name = sprintf('%s/cnnresultWeightedPrior_irradiat_test_lamb_%f.mat',outDirectory,lambda0);
save(name,'reconIrradiation');

result_stitched = weights2.*reconIrradiation + (1-weights2).*pilot_testOnEigSpace;
figure('Name','Final_stitched','NumberTitle','off');
imshow(result_stitched,[]);
name = sprintf('%s/cnnfinal_stitched_lamb_%f.png',outDirectory,lambda0);
imwrite(mat2gray(result_stitched),name);

%% Load necessary images
name = sprintf('%s/testIm_template_%d.mat',outDirectory,testImNo);
testIm = load(name);
testIm = testIm.testIm;
name = sprintf('%s/pilot_test_360_angles_2000_I_0.mat',outDirectory);
pilot_test = load(name);
pilot_test = pilot_test.pilot_test;

%% ssim calculation
test = mat2gray(testIm);
testpilot = mat2gray(pilot_test);
test_wted_recon = mat2gray(resultWeightedPrior1);
test_irrad = mat2gray(reconIrradiation);

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
fh = figure;imshow(testpilot,[]); hold on;
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
name = sprintf('%s/cnnssim.txt',outDirectory);
fileID = fopen(name, 'a');
fprintf(fileID,'%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n','ssim_pilot_ov',...
    'ssim_wtedrecon_ov','ssim_irrad_ov','rmse_pilot_ov','rmse_wtedrecon_ov','rmse_irrad_ov',...
    'ssim_pilot_roi','ssim_wtedrecon_roi','ssim_irrad_roi','rmse_pilot_roi',...
    'rmse_wtedrecon_roi','rmse_irrad_roi');
fprintf(fileID,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',ssim_pilot_ov,...
    ssim_wtedrecon_ov,ssim_irrad_ov,rmse_pilot_ov,rmse_wtedrecon_ov,rmse_irrad_ov,...
    ssim_pilot_roi,ssim_wtedrecon_roi,ssim_irrad_roi,rmse_pilot_roi,...
    rmse_wtedrecon_roi,rmse_irrad_roi );
fclose(fileID);