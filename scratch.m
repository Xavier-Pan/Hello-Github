%i=5;
%A=test_x(i,:);
%B=reshape(A,28,28);
    %C=uint8(round(B));
%figure(1);
%surface(B);

%i=find(test_label(i,:)>0);
%fprintf('label=%d\n',i-1);
%X=reshape(A,28,28);
%colormap default
%figure(2);
%imagesc(X);
%for i=1:-1:-5
%    fprintf('in i=%d',i);
%end
%===gradient check
W{1}=rand(10,784);
W{1} = reshape(sin(1:numel(W{1})), size(W{1})) / 10;
W{2}=rand(10,10);
W{3}=rand(10,10);
W{4}=rand(10,10);
b=rand(10,4);

hiddLayerNum=3;hiddlayerNum=3;
delta=10^-4;

[output,h]=fordProp(hiddLayerNum,W,b,test_x(1,:));% h include all activation layer,h(:,end)=estimate_y
[t_grad_W,t_grad_b]=backProp(test_x(1,:),h,W,test_label(1,:));
%===numarical gradient====
num_W=W;
for k=1:size(W,2)
    for i=1:size(W{k},1)
        for j=1:size(W{k},2)
            delta1_W=W;
            delta2_W=W;
            delta1_W{k}(i,j)=W{k}(i,j)+delta;
            delta2_W{k}(i,j)=W{k}(i,j)-delta;
            %==compute J ===
            y=fordProp(hiddlayerNum,delta1_W,b,test_x(1,:));
            delta_J1=-log(y)'*test_label(1,:)';
            y=fordProp(hiddlayerNum,delta2_W,b,test_x(1,:));
            delta_J2=-log(y)'*test_label(1,:)';
            %====
            num_W{k}(i,j)=(delta_J1-delta_J2)/(2*delta);
        end
    end
end
%================
diff=t_grad_W{2}-num_W{2}
%===================