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
   
    
    sw = sw+si;  %Sw matrix
end
 
 
sb = zeros(504,504);
 
for i = 1:200
    d = mean_vector(:,i)-totalmeanlda;
    sb = sb+2*(d*transpose(d));  %Sb matrix
end


[V,D]=eig(sb,sw);
  
[Dlda ,order] = sort(diag(D),'descend');  %# sort eigenvalues in descending order

dimensions=100;  %change dimensions here
DimRed = V(:,order(1:dimensions));
new_traindata=DimRed'*train_final;
new_testdata=DimRed'*test_final;

%Bayes' classifier
for a=1:200
    mean_inclass(:,a)=(new_traindata(:,a)+new_traindata(:,a+200))/2;
end
for n=1:200
 cov_vector(:,:,n)= ((new_traindata(:,n)- mean_inclass(:,n))*transpose(new_traindata(:,n)-mean_inclass(:,n)))+((new_traindata(:,n+200)- mean_inclass(:,n))*transpose(new_traindata(:,n+200)-mean_inclass(:,n)))/2 ;
end 
for n=1:200
    cov_final(:,:,n)=cov_vector(:,:,n)+eye(dimensions);
end

p=zeros(200,200);
count=0;
for b=1:200
    for c=1:200
        %p(c)= mvnpdf(test_final(:,b),mean_vector(:,c),cov_final(:,:,c));
         p(b,c)=(1/(((2*pi)^252)*(det(cov_final(:,:,c))^0.5)))*exp(-0.5*transpose(new_testdata(:,b)-mean_inclass(:,c))/(cov_final(:,:,c))*(new_testdata(:,b)-mean_inclass(:,c)));
         
    end
   [maxprob,index]=max(p(b,:));
   if index==b
       count=count+1;
   end
   
    
end
accuracy=count/200*100
t=toc;


