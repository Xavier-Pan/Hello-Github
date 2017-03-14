function [output,h]=fordProp_newversion(hiddLayerNum,W,b,x)
% x:one data  
% b:bias  
% W:weight matrix
%==================================================    
    if size(x,1)<size(x,2)%turn x to colum if need
        x=x';
    end
    %=========initial===
    a{1}=zeros(size(b{1}))%initial Wx+b
    a{2}=zeros(size(b{2}))%initial Wx+b
    h{1}=zeros(size(b{1}));
    h{2}=zeros(size(b{1}));
    %==========feed forward==============
    a{1}=b{1}+W{1}*x;%for layer 1
    h{1}= 1./(1+exp(-a{1}));
    
    a{2}=b{2}+W{2}*h{1};%W{i} mean ith layer weight matrix      
    h{2}=softmax(a{2});         
       
    output=h{2};
end
