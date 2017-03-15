 %assume every hiddent layer has the same nuron number
%
%
clear all
load SVHN
%=======hyperparameter ===========
learnRate=0.0001;
totalDataNum=size(train_x,1);
classNum=size(train_label,2);
hiddenUnitNum=300;
inputUnitNum1=size(train_x,2);
outptUnitNum=classNum;
hiddLayerNum=1;
layerNum=hiddLayerNum+1;%not include input layer

miniBatchSize=100;
if mod(totalDataNum,miniBatchSize)~=0,
    fprintf('miniBatchSize can not devide data Number!!');
end
%===========training=====================

%===initial weighting matrix & bias
W{1}=rand(hiddenUnitNum,inputUnitNum1)*2-1;
b{1}=rand(hiddenUnitNum,1)*2-1;%initial bias

W{layerNum}=rand(outptUnitNum,hiddenUnitNum)*2-1;
b{layerNum}=rand(outptUnitNum,1)*2-1;%initial bias
%=======================

epoch=0;
maxEpoch=200;
%error_threshold=0.02;
error_record= costf_matrixVersion(train_x,W,b,train_label);%record error in each epoch;
fprintf('Epoch:%d objective:%d \n',epoch,error_record(epoch+1));
tic,
while(epoch < maxEpoch)
    epoch=epoch+1;
    %==========random data =================
    rand_index=randperm(size(train_x,1));%random data order
    rand_X=train_x(rand_index,:);
    rand_label=train_label(rand_index,:);
    %========== compute gradient for every mini-batch===========   
    for i=1:miniBatchSize:totalDataNum              
        %===in a minibatch              
        H=fordProp_batchVersion(W,b,rand_X(i:i+miniBatchSize-1,:));
        [G_W,G_bias]=backProp_batchVersion(rand_X(i:i+miniBatchSize-1,:),H,W,rand_label(i:i+miniBatchSize-1,:));
                     %===  update gradient ========
        for t=1:layerNum 
            W{t}=W{t}-learnRate*G_W{t}; 
            b{t}=b{t}-learnRate*G_bias{t}; 
        end   
    end% one epoch   
    error_record(epoch)=costf_matrixVersion(train_x,W,b,train_label);%record error in each epoch      
    %===========================================================          
   % fprintf('Epoch:%d objective:%d \n',epoch,error_record(epoch));  
end
toc;
plot(error_record);
%=========       ===============
err_rate=test(W,b,test_label,test_x);
figure(1);
plot(error_record);str=strcat('error rate:',num2str(err_rate),'  learning rate:',num2str(learnRate),'  BatchSize:',num2str(miniBatchSize),...
    ' hiddenUnitNum:',num2str(hiddenUnitNum));
title(str);