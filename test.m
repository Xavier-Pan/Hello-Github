function err=test(W,b,test_label,test_x)
    hiddlayerNum=size(W,2)-1;
    err=0;
    for i=1:size(test_x,1)
        y=fordProp(hiddlayerNum,W,b,test_x(i,:));
        [value,index]=max(y);
        if index~=find(test_label(i,:)==1)
            err=err+1;
        end
    end
    err=err/size(test_x,1);%error rate
end
