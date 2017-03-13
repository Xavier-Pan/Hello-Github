function [grad_W,grad_b]=backProp(x,h,W,target)
%##### h:activation vector for all layer and input layer 
%##### W:weight matrix for all layer  
%##### t: label {0,1}^10 
%##### x: one data,because I can't put it into h 

    if size(target,1)<size(target,2) target=target'; end %assure it is column 
    if size(x,1)<size(x,2) x=x'; end %assure it is column     
   g=-target.*(1./h{end});%(partial J)/(partial y_hat)
    layerNum=size(h,2);%except input layer
    for t=1:layerNum
        grad_W{t}=zeros(size(W{t}));
        grad_b{t}=zeros(size(h{t}));%bias. fust for get the same size!!!!!!!!!!!!!!!
    end%initial weight matrix    
    
    for k=layerNum:-1:1     
        g=g.*sigmoid(h{k}).*(1-sigmoid(h{k}));
        %==for softmax
        if k==layerNum 
            g=h{end}-target;
        end
        %==============
        grad_b{k}=g;%==== lock normalize term
        if k>1
            grad_W{k}=g*h{k-1}';%%==== lock normalize term
        else
            grad_W{k}=g*x';%%==== lock normalize term
        end
        g=W{k}'*g;
    end    
end