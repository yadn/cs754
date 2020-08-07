function [positiveLabels,positivePatchVector,negativeLabels,negativePatchVector] = create_training_dataset(dim,hypothesisTestResultFBP,...
                                                                                    binaryInlier,patchWidth,patchHeight,center)


numPatches = 100;%(dim(1)/patchWidth) *(dim(2)/patchHeight);
patchVector = zeros(patchWidth*patchHeight,numPatches);
label = zeros(1,numPatches);
patchNumber = 0;
positiveLabelNumber = 0;
negativeLabelNumber = 0;
positiveLabels = zeros(1,30);
positivePatchVector = zeros(patchWidth*patchHeight,30);
negativeLabels = zeros(1,30);
negativePatchVector = zeros(patchWidth*patchHeight,30);


for i = 1:dim(1)-patchWidth+1                  %(dim(1)/patchWidth)-1
    for j = 1:dim(2)-patchHeight+1           %(dim(2)/patchHeight)-1
        startWidth = j;
        endWidth = startWidth + patchWidth-1;
        startHeight = i;
        endHeight = startHeight + patchHeight-1;
        patchNumber = patchNumber + 1;
        patch = hypothesisTestResultFBP(startHeight:endHeight,startWidth:endWidth);
        patchVector(:,patchNumber) = patch(:);
        label(patchNumber) = binaryInlier(startHeight + center,startWidth + center);
        if (label(patchNumber) ==1)% && positiveLabelNumber<30)
            positiveLabelNumber = positiveLabelNumber + 1;
            positiveLabels(positiveLabelNumber) = label(patchNumber);
            positivePatchVector(:,positiveLabelNumber) = patchVector(:,patchNumber);
        elseif (label(patchNumber) ==0)% && negativeLabelNumber<30)
            negativeLabelNumber = negativeLabelNumber + 1;
            negativeLabels(negativeLabelNumber) = label(patchNumber);
            negativePatchVector(:,negativeLabelNumber) = patchVector(:,patchNumber);
        end
%         if (positiveLabelNumber==30 && negativeLabelNumber==30)
%             fprintf('got 30 positive and 30 negative samples!\n');
%             return;
%         end
    end
end
end