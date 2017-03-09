function [grad_W,grad_b]=backProp(h,W,t)
%h:activation vector for all layer and input layer  W:weight matrix for all
%layer  t: label {0,1}^10
    g=t.*(1./h(:,end));  %derivative of cost function to output
    layerNum=size(h,2)-1;%except input layer
    grad_b=zeros(size(W,1),layerNum);%bias
    grad_W=zeros(size(W));
    for k=layerNum:-1:1
      %  k=layerNum+1-i;
        g=g.*sigmoid(h(:,k+1)).*(1-sigmoid(h(:,k+1)));% k+1 because index start from 1 ,then h^(k+1) in the same layer with W^(k)
        grad_b(:,k)=g;%==== lock normalize term
        grad_W(:,:,k)=g*h(:,k+1)';%%==== lock normalize term
        g=W(:,:,k)'*g;
    end
end