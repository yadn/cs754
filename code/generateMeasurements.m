function [y, y_noNoise] = generateMeasurements(testIm,I_0, angleArr, gaussianNoisePercent)

proj = I_0*exp(-radon(testIm,angleArr));
proj_poisson = poissrnd(proj); % adding poisson noise
noiseMean = 0;
noiseSD = (gaussianNoisePercent/100)*mean(proj(:)); % should the sigma of the Gaussian noise be dependent on Poisson noise
noise = noiseMean + noiseSD*randn(size(proj_poisson(:)));
% sigma = noiseSD;
y = proj_poisson(:) +  noise; % adding gaussian noise
y_noNoise = radon(testIm,angleArr);
y_noNoise = y_noNoise(:);
end
% NSR = 1/sqrt(mean(proj_gauss(:)));

% name = sprintf('%s/measurements.mat',outDirectory);
% save(name,'y','y_noNoise','idx1','sigma','NSR');