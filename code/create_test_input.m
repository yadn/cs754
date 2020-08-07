function [patchVectorTest] = create_test_input(dim,pilot,...
                                                                                   patchWidth,patchHeight)
                                                                                
numPatches = 100;%(dim(1)/patchWidth) *(dim(2)/patchHeight); 
patchVectorTest = zeros(patchWidth*patchHeight,numPatches);
patchNumber = 0;

for i = 1:dim(1)-patchWidth+1 
    for j = 1:dim(2)-patchHeight+1 
        startWidth = j;
        endWidth = startWidth + patchWidth-1;
        startHeight = i;
        endHeight = startHeight + patchHeight-1;
        patchNumber = patchNumber + 1;
        patch = pilot(startHeight:endHeight,startWidth:endWidth);
        patchVectorTest(:,patchNumber) = patch(:);
    end
end
end