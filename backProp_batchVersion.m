function [G_W,G_bias]=backProp_batchVersion(X,H,W,T,hiddentLayerNum)
%##### H:activation vector for all layer and input layer. H{1}(:,k) mean kth data in layer one
%##### W:weight matrix for all layer  
%##### T: row is one data's label. {0,1}^10 
%##### X: batch data
    
    T=T';
    X=X';    
    G=H{end}-T;%G is error
    G_bias{hiddentLayerNum+1}=G*ones(size(G,2),1);%one minibatch's gradient sum
    G_W{hiddentLayerNum+1}=G*H{end-1}';%one minibatch's gradient sum
    G=W{hiddentLayerNum+1}'*G;
    %{
    G=G.*sigmoid(H{end-1}).*(1-sigmoid(H{end-1}));
    G_bias{end-1}=G*ones(size(G,2),1);
    G_W{end-1}=G*X';
    %}
    for i=hiddentLayerNum:-1:1
        G=G.*sigmoid(H{i}).*(1-sigmoid(H{i}));%G is error
        G_bias{i}=G*ones(size(G,2),1);
        if i==1
            G_W{i}=G*X';
        else
            G_W{i}=G*H{i-1}';
        end
        G=W{i}'*G;
    end   
end