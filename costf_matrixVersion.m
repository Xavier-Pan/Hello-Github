function J=costf_matrixVersion(X,W,b,labels,hiddLayerNum)
% X:minibatch data  
% b:bias  
% W:weight matrix
%==================================================    
   X=X';%each column is a data
   N=size(X,2);
   H=fordProp_batchVersion(W,b,X,hiddentLayerNum)  
   S=log(H{end})'.*labels(1:size(H{end},2),:);% each row is cross entropy for a data
   J=ones(1,size(S,1))*S*ones(size(S,2),1);%sum all the term in S
   J=-J/N;
end
