clc;
clear all;
close all;
tic
load('data.mat');
face_copy(:,:,:)=face(:,:,:);


for i=1:200
    N(:,:,i)=face_copy(:,:,3*i-2); %neutral images
    E(:,:,i)=face_copy(:,:,3*i-1); %happy images
    I(:,:,i)=face_copy(:,:,3*i);   %illuminated images
end
for n=1:200
    train_set(:,:,n)=N(:,:,n);
    train_set(:,:,n+200)=E(:,:,n);  %separating into training and testing data
    test_set(:,:,n)=I(:,:,n);
end
for z=1:400
    train_final(:,z)=reshape(train_set(:,:,z),[],1); %reshaping data
end
for p=1:200
    test_final(:,p)=reshape(test_set(:,:,p),[],1);
end
count=0;
for z=1:200
    for c=1:400
        dist(:,c)=sqrt(sum((test_final(:,z) - train_final(:,c)).^2)); %finding eucledian distance
    end
    [distance,order]=sort(dist); %ascending order of distances
    k=1;                         %change K value here (<400)
    index=order(1:k);
    
    for t=1:k
        if index(t)>200
            index(t)=index(t)-200;
        end
    end
    m=mode(index);  %most frequent class
    
    if m==z
        count=count+1;
    end
end
accuracy=count*100/200
t=toc;

