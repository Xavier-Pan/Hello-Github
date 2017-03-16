function err=testBatchVersion(W,b,test_label,test_x)
    hiddlayerNum=size(W,2)-1;
    err=0;
    Y=fordProp_batchVersion(W,b,test_x,hiddlayerNum);
    [V INDEX]=max(Y{end});
    [V2 INDEX2]=max(test_label');
    result=(INDEX~=INDEX2);
    err=sum(result)/size(test_x,1);
end
