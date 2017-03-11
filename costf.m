function J=costf(X,W,b,test_labels)
%assume each row in test_label is a data
   hiddlayerNum=size(W,2)-1;
   J=0;%initial
   for i=1:size(X,1)
       score=fordProp(hiddlayerNum,W,b,X(i,:));
       J=J+log(score)'*test_labels(i,:)';
   end
   J=-J;
end