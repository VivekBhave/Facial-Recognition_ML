clc;
clear all;
close all;
load('data.mat');
tic
face_copy(:,:,:)=face(:,:,:);

for i=1:200
    N(:,:,i)=face_copy(:,:,3*i-2);
    E(:,:,i)=face_copy(:,:,3*i-1);
    I(:,:,i)=face_copy(:,:,3*i);
end
for n=1:200
    train_set(:,:,n)=N(:,:,n);
    train_set(:,:,n+200)=E(:,:,n);
    test_set(:,:,n)=I(:,:,n);
end

for z=1:400
    train_final(:,z)=reshape(train_set(:,:,z),[],1);
end
for p=1:200
    test_final(:,p)=reshape(test_set(:,:,p),[],1);
end
 
    sumlda = zeros(504,400);
    avg = zeros(504,200);
 
 
for i = 1:400
    sumlda = sumlda + train_final(:,i);
        
end
    
totalmeanlda=sumlda/(400); %total mean
 
for a=1:200
    mean_vector(:,a)=(train_final(:,a)+train_final(:,a+200))/2; %mean per class
end

sw = zeros(504,504);

for i = 1:200
    
    si = train_final(:,i) - mean_vector(:,i) ; %Si matrix
   
    
    sw = sw+si; %Sw matrix
end
 
 
sb = zeros(504,504);
 
for i = 1:200
    d = mean_vector(:,i)-totalmeanlda;
    sb = sb+2*(d*transpose(d)); %Sb matrix
end

[V,D]=eig(sb,sw);
  
[Dlda ,order] = sort(diag(D),'descend');  %# sort eigenvalues in descending order

dimensions=100;  %change dimensions here
DimRed = V(:,order(1:dimensions));
new_traindata=DimRed'*train_final;
new_testdata=DimRed'*test_final;
count=0;
%KNN Classifier
for z=1:200
    for c=1:400
        dist(:,c)=sqrt(sum((new_testdata(:,z) - new_traindata(:,c)).^2));
    end
    [distance,order]=sort(dist);
    k=1;  %change K values here
    index=order(1:k);
    
    for t=1:k
        if index(t)>200
            index(t)=index(t)-200;
        end
    end
    m=mode(index);
    if m==z
        count=count+1;
    end
end
accuracy=count*100/200
t=toc;
