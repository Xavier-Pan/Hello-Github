function [grad_W,grad_b]=backProp(x,h,W,t)
%h:activation vector for all layer and input layer #### W:weight matrix for all
%layer  ##### t: label {0,1}^10 #####   x: one data,because I can't put it into h 
    if size(t,1)<size(t,2) t=t'; end %assure it is column 
    if size(x,1)<size(x,2) x=x'; end %assure it is column 
    g=h(:,end)-t;  %derivative of cost function to output
    layerNum=size(h,2);%except input layer
    for t=1:layerNum grad_W{t}=zeros(size(W{t}));end%initial weight matrix
    
    grad_b=zeros(size(h));%bias
   
    for k=layerNum:-1:1     
        g=g.*sigmoid(h(:,k)).*(1-sigmoid(h(:,k)));%
        grad_b(:,k)=g;%==== lock normalize term
        if k>1
            grad_W{k}=g*h(:,k-1)';%%==== lock normalize term
        else
            grad_W{k}=g*x';%%==== lock normalize term
        end
        g=W{k}'*g;
    end
    
end