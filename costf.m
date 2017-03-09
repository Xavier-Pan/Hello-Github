function J=costf(X,W,b,test_labels)
   layerNum=size(W,3);
   for i=1:size(X,1)
       score=fordProp(layerNum,W,b,X(i,:))
       J=J+log(score)*test_labels(i,:)';
   end
   J=-J;
end