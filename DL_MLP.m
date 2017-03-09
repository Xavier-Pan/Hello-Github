%=======hyperparameter ===========
unitNum=20;
layerNum=3;
miniBatchSize=100;
%================================
b=rand(unitNum,layerNum);%initial bias
W=rand(unitNum,unitNum,layerNum);%initial weighting matrix
epoch=0;
maxEpoch=200;
error_threshold=0.2;
while(epoch < maxEpoch)
    epoch=epoch+1;
    rand_index=randperm(size(train_x,1));%random data order
    rand_x=train_x(rand_index,:);
    %========== compute gradient ===========
    for i=1:miniBatchSize-1:size(x,1)
        batch_X=rand_x(i:i+miniBatchSize-1,:);
        %===in a minibatch
        grad_W=zeros(size(W));
        grad_b=zeros(size(b));
        for t=1:miniBatchSize
            h=forwardProp(layerNum,W,b,batch_X);% h include all activation layer,h(:,end)=estimate_y
            [t_grad_W,t_grad_b]=backProp(h,W);
            grad_W=grad_W+t_grad_W;
            grad_b=grad_b+t_grad_b;            
        end
        W=W-alpha*grad_W;
        b=b-alpha*grad_b;
    end
    %====================================
end