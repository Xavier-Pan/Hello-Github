function [grad_W,grad_b]=backProp_newversion(x,h,W,target)
%##### h:activation vector for all layer and input layer 
%##### W:weight matrix for all layer  
%##### t: label {0,1}^10 
%##### x: one data,because I can't put it into h 

    if size(target,1)<size(target,2) target=target'; end %assure it is column 
    if size(x,1)<size(x,2) x=x'; end %assure it is column     
    
    %g=-target.*(1./h{end});%(partial J)/(partial y_hat)
    layerNum=size(h,2);%except input layer
    for t=1:layerNum
        grad_W{t}=zeros(size(W{t}));
        grad_b{t}=zeros(size(h{t}));%bias. fust for get the same size!!!!!!!!!!!!!!!
    end%initial weight matrix    
       
    %==ouput to hiddent       
    g=h{2}-target;            
    grad_b{2}=g;%==== lock normalize term    
    grad_W{2}=g*h{1}';%%==== lock normalize term        
    g=W{2}'*g;
    %===============    
    %======hiddent to input=========    
    g=g.*sigmoid(h{1}).*(1-sigmoid(h{1}));            
    grad_b{1}=g;%==== lock normalize term        
    grad_W{1}=g*x';%%==== lock normalize term        
end