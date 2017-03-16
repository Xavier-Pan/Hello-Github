 function plotData(W,b,X,label,error_record,learnRate,miniBatchSize,hiddenUnitNum,time,window)
    plot(error_record);
    %=========       ===============
    err_rate=test(W,b,label,X);
    figure(window);
    plot(error_record);str=strcat('error rate:',num2str(err_rate),'  learning rate:',num2str(learnRate),'  BatchSize:',num2str(miniBatchSize),...
        ' hiddenUnitNum:',num2str(hiddenUnitNum),' hiddenlayerNum:',num2str(size(W,2)-1),'  time:',num2str(time),'sec.');
    title(str);
end