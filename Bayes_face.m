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
    train_final(:,z)=reshape(train_set(:,:,z),[],1); %reshaping of training data
end
for p=1:200
    test_final(:,p)=reshape(test_set(:,:,p),[],1); %reshaping of testing data
end

for a=1:200
    mean_vector(:,a)=(train_final(:,a)+train_final(:,a+200))/2; %MLE mean
end
for n=1:200
 cov_vector(:,:,n)= ((train_final(:,n)- mean_vector(:,n))*transpose(train_final(:,n)-mean_vector(:,n)))+((train_final(:,n+200)- mean_vector(:,n))*transpose(train_final(:,n+200)-mean_vector(:,n)))/2 ;  %MLE covariance matrix
end 
for n=1:200
    cov_final(:,:,n)=cov_vector(:,:,n)+eye(504); %adding identity matrix
end

p=zeros(200,200);
count=0;
for b=1:200
    for c=1:200
         p(b,c)=(1/(((2*pi)^252)*(det(cov_final(:,:,c))^0.5)))*exp(-0.5*transpose(test_final(:,b)-mean_vector(:,c))/(cov_final(:,:,c))*(test_final(:,b)-mean_vector(:,c))); %finding probability density of test point with each class
         
    end
   [maxprob,index]=max(p(b,:)); %finding class with maximum probabilty density
   if index==b
       count=count+1;
   end
   
    
end
accuracy=count/200*100
t=toc;
