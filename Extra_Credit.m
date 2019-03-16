clc;
clear all;
close all;
load ('data.mat');
 

face_copy(:,:,:)=face(:,:,:);

train_set= face_copy(:,:,1:400);  %separating data into training and testing
test_set= face_copy(:,:,401:600);

for z=1:400
    train_final(:,z)=reshape(train_set(:,:,z),[],1);
end                                                        %reshaping matrix
for p=1:200
    test_final(:,p)=reshape(test_set(:,:,p),[],1);    
end

count=0;
for z=1:200
    for c=1:400
        dist(:,c)=sqrt(sum((test_final(:,z) - train_final(:,c)).^2));  %eucledian distance
    end
    [distance,order]=sort(dist);
    k=200;
    index=order(1:k);
    
    for t=1:k
        if rem(index(t),3)==1   
            index(t)=1;
        elseif rem(index(t),3)==2   %classifying into 1. Neutral, 2. Happy, 3. Illuminated
            index(t)=2;
        elseif rem(index(t),3)==0
            index(t)=3;
        end
    end
    m=mode(index);
    
    if m==rem(z+199,3)
        count=count+1;
    end
end
accuracy=count*100/200