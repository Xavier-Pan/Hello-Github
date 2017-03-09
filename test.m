function err=test(W,b,test_label,test_x)
    layerNum=size(W,3);
    err=0;
    for i=1:size(test_x,1)
        y=fordProp(layerNum,W,b,test_x);
        [value index]=max(y);
        if index~=find(test_label(i,:)==1)
            err=err+1;
        end
    end
    err=err/size(test_x,1);%error rate
end