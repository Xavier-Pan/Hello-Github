function [output,h]=fordProp(layerNum,W,b,x)%assume that ouput layer has the same activation function with hiddent layer
 %l:layerNum  x:one data  b:bias  W:weight matrix
 %==================================================
    a=zeros(size(W,1),1);
    if size(x,1)<size(x,2)%turn x to colum if need
        x=x';
    end
    %============
    h=x;%data784*1
    for i=1:layerNum
        a(:,i)=b(i)+W(:,:,i)*h(:,i);
        h=[h 1./(1+exp(-a(:,i)))];
    end
   % h=h(:,2:end);
    output=h(:,end);
end
