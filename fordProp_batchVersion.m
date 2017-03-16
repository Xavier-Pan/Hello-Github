function H=fordProp_batchVersion(W,b,X,hiddentLayerNum)
% X:minibatch data  
% b:bias  
% W:weight matrix
%==================================================       
   X=X';%each row is a data
   H{1}=sigmoid(bsxfun(@plus,b{1},W{1}*X));
   for i=2:hiddentLayerNum
       H{i}=sigmoid(bsxfun(@plus,b{i},W{i}*H{i-1})); 
   end
   H{hiddentLayerNum+1}=softmax(bsxfun(@plus,b{hiddentLayerNum+1},W{hiddentLayerNum+1}*H{hiddentLayerNum})); 
end
