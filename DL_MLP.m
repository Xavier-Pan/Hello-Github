%assume every hiddent layer has the same nuron number
%
%
load SVHN
clear all
%=======hyperparameter ===========
learnRate=0.2;
totalDataNum=size(train_x);%#### correct =>  size(train_x,1);
classNum=size(train_label,2);
hiddenUnitNum=10;
inputUnitNum1=size(train_x,2);
outptUnitNum=classNum;
layerNum=5-1;%not include input layer
hiddLayerNum=3;
miniBatchSize=100;
if mod(totalDataNum,miniBatchSize)~=0 fprintf('miniBatchSize can not devide data Number!!');end
%===========training=====================
b=rand(hiddenUnitNum,layerNum)*2-1;%initial bias
%===initial weighting matrix
W{1}=rand(hiddenUnitNum,inputUnitNum1)*2-1;
for i=2:hiddLayerNum
    W{i}=rand(hiddenUnitNum,hiddenUnitNum)*2-1;
end
%W{3}=rand(hiddenUnitNum,hiddenUnitNum)*2-1;
W{layerNum}=rand(outptUnitNum,hiddenUnitNum)*2-1;
%===
epoch=0;
maxEpoch=10;
error_threshold=0.02;
error_record=0;
while(epoch < maxEpoch)
    epoch=epoch+1;
    %==========random data =================
    rand_index=randperm(size(train_x,1));%random data order
    rand_X=train_x(rand_index,:);
    rand_label=train_label(rand_index,:);
    %========== compute gradient for every mini-batch===========
    for i=1:miniBatchSize:totalDataNum
        %batch_X=rand_x(i:i+miniBatchSize-1,:);
        %batch_label=rand_label(i:i+miniBatchSize-1,:);
        %===in a minibatch
        for t=1:layerNum grad_W{t}=zeros(size(W{t}));end%initial weight matrix
        grad_b=zeros(size(b));
        for t=0:miniBatchSize-1
            [output,h]=fordProp(hiddLayerNum,W,b,rand_X(i+t,:));% h include all activation layer,h(:,end)=estimate_y
            [t_grad_W,t_grad_b]=backProp(rand_X(i+t,:),h,W,rand_label(i+t,:));
             for ii=1:layerNum grad_W{ii}=grad_W{ii}+t_grad_W{ii}; end%assign weight matrix            
            grad_b=grad_b+t_grad_b;            
        end
        %===  update gradient ========
        for t=1:layerNum W{t}=W{t}-learnRate*grad_W{t}; end
        b=b-learnRate*grad_b;
    end
    %====================================
    error_record=[error_record costf(train_x,W,b,train_label)];%record error in each epoch
end
%=========       ===============
error_record=error_record(2:end);
err_rate=test(W,b,test_label,test_x);
figure(1);
plot(error_record);str=strcat('error rate:',num2str(err_rate));
title(str);
