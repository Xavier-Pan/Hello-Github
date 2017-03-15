function H=fordProp_batchVersion(W,b,X)
% X:minibatch data  
% b:bias  
% W:weight matrix
%==================================================    
   X=X';%each row is a data
   H{1}=sigmoid(bsxfun(@plus,b{1},W{1}*X));
   H{2}=softmax(bsxfun(@plus,b{2},W{2}*H{1})); 
end
