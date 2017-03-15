function J=costf_matrixVersion(X,W,b,labels)
% X:minibatch data  
% b:bias  
% W:weight matrix
%==================================================    
   X=X';%each row is a data
   H{1}=sigmoid(bsxfun(@plus,b{1},W{1}*X));
   H{2}=softmax(bsxfun(@plus,b{2},W{2}*H{1})); 
   S=log(H{2})'.*labels(1:size(H{2},2),:);% each row is cross entropy for a data
   J=ones(1,size(S,1))*S*ones(size(S,2),1);%sum all the term in S
   J=-J;
end
