function [output,h]=fordProp(hiddLayerNum,W,b,x)%assume that ouput layer has the same activation function with hiddent layer
 %l:layerNum  x:one data  b:bias  W:weight matrix
 %==================================================
    a=zeros(size(b));%initial Wx+b
    if size(x,1)<size(x,2)%turn x to colum if need
        x=x';
    end
    %============
    h=zeros(size(a));%10*4
    %fprintf('W{1} size:%d,%d',size(W{1},1),size(W{1},2));
    a(:,1)=b(:,1)+W{1}*x;%for layer 1
    h(:,1)= 1./(1+exp(-a(:,1)));
    for i=2:hiddLayerNum+1
        a(:,i)=b(:,i)+W{i}*h(:,i-1);%W{i} mean ith layer weight matrix  
        if i<hiddLayerNum+1
            h(:,i)= 1./(1+exp(-a(:,i))); 
        else
            h(:,end)= softmax(a(:,end));%output activation function for softmax
        end
    end  
    output=h(:,end);
end
