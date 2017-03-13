%assume every hiddent layer has the same nuron number
%
%
clear all
load SVHN
%=======hyperparameter ===========
learnRate=0.01;
totalDataNum=size(train_x,1);
classNum=size(train_label,2);
hiddenUnitNum=1000;
inputUnitNum1=size(train_x,2);
outptUnitNum=classNum;
hiddLayerNum=1;
layerNum=hiddLayerNum+1;%not include input layer

miniBatchSize=100;
if mod(totalDataNum,miniBatchSize)~=0, fprintf('miniBatchSize can not devide data Number!!');end
%===========training=====================

%===initial weighting matrix & bias
W{1}=rand(hiddenUnitNum,inputUnitNum1)*2-1;
b{1}=rand(hiddenUnitNum,1)*2-1;%initial bias
for i=2:hiddLayerNum
    W{i}=rand(hiddenUnitNum,hiddenUnitNum)*2-1;
    b{i}=rand(hiddenUnitNum,1)*2-1;%initial bias
end
W{layerNum}=rand(outptUnitNum,hiddenUnitNum)*2-1;
b{layerNum}=rand(outptUnitNum,1)*2-1;%initial bias
%=======================

epoch=0;
maxEpoch=100;
error_threshold=0.02;
error_record= costf(train_x,W,b,train_label);%record error in each epoch;
while(epoch < maxEpoch)
    epoch=epoch+1;
    %==========random data =================
    rand_index=randperm(size(train_x,1));%random data order
    rand_X=train_x(rand_index,:);
    rand_label=train_label(rand_index,:);
    %========== compute gradient for every mini-batch===========
    for i=1:miniBatchSize:totalDataNum        
        %===in a minibatch
        for t=1:layerNum % initial batch_gradient_W & batch_gradient_bias
            grad_W{t}=zeros(size(W{t}));
            grad_b{t}=zeros(size(b{t}));
        end%initial weight matrix
        
        for t=0:miniBatchSize-1
            [output,h]=fordProp(hiddLayerNum,W,b,rand_X(i+t,:));% h include all activation layer,h(:,end)=estimate_y
            [t_grad_W,t_grad_b]=backProp(rand_X(i+t,:),h,W,rand_label(i+t,:));
             for ii=1:layerNum  %sum all gradient for minibatch
                 grad_W{ii}=grad_W{ii}+t_grad_W{ii}; 
                 grad_b{ii}=grad_b{ii}+t_grad_b{ii};            
             end%assign weight matrix            
           
        end% one minibatch 
        
        %===  update gradient ========
        for t=1:layerNum 
            W{t}=W{t}-learnRate*grad_W{t}; 
            b{t}=b{t}-learnRate*grad_b{t}; 
        end
        
    end% one epoch
    %====================================  
        error_record=[error_record costf(train_x,W,b,train_label)];%record error in each epoch
        %fprintf('cost:%d \n',costf(train_x,W,b,train_label))  
        fprintf('Epoch:%d objective:%d \n',epoch,error_record(epoch+1));
end
%=========       ===============
err_rate=test(W,b,test_label,test_x);
figure(1);
plot(error_record);str=strcat('error rate:',num2str(err_rate));
title(str);