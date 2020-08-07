function fbp = filteredBackProjection2D(y,numAngles,idx1,dim)

proj = y;
%proj = reshape(y,[size(y,1)/numAngles numAngles]);
fbp = iradon(proj,idx1,'linear','cosine');
fbp = fbp(2:2+dim(1)-1,2:2+dim(2)-1);