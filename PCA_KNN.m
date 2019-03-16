clc;
clear all;
close all;
load ('data.mat');
 
tic
face_copy(:,:,:)=face(:,:,:);
for i=1:200
    N(:,:,i)=face_copy(:,:,3*i-2); %neutral images
    E(:,:,i)=face_copy(:,:,3*i-1); %happy images
    I(:,:,i)=face_copy(:,:,3*i);   %illuminated images
end
for n=1:200
    train_set(:,:,n)=N(:,:,n);
    train_set(:,:,n+200)=E(:,:,n); %separating into training and tesing
    test_set(:,:,n)=I(:,:,n);
end
for z=1:400
    train_final(:,z)=reshape(train_set(:,:,z),[],1); %reshaping data
end
for p=1:200
    test_final(:,p)=reshape(test_set(:,:,p),[],1);
end
 


sumpca = zeros(504,400);

 
for i = 1:400
    sumpca = sumpca + train_final(:,i);
        
end
 
meanpca = sumpca/(400);
 
 
 scatM=zeros(504,504);
for i = 1:400
    
        scatM = scatM+(train_final(:,i) - meanpca)*(transpose(train_final(:,i) - meanpca)); %scatter matrix
    
end
 
[V,D]=eig(scatM);
 
[Dpca ,order] = sort(diag(D),'descend');  %# sort eigenvalues in descending order

dimensions=100;  %change dimensions here
DimRed = V(:,order(1:dimensions)); %dimension reduction
new_traindata=DimRed'*train_final; 
new_testdata=DimRed'*test_final;
count=0;
for z=1:200
    for c=1:400
        dist(:,c)=sqrt(sum((new_testdata(:,z) - new_traindata(:,c)).^2)); %eucledian distance
    end
    [distance,order]=sort(dist); %sorting distances
    k=1;  %change K value here
    index=order(1:k);
    
    for t=1:k
        if index(t)>200
            index(t)=index(t)-200;
        end
    end
    m=mode(index); %most frequent class
    
    if m==z
        count=count+1;
    end
end
accuracy=count*100/200

t=toc;



