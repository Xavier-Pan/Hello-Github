function [G_W,G_bias]=backProp_batchVersion(X,H,W,T)
%##### H:activation vector for all layer and input layer. H{1}(:,k) mean kth data in layer one
%##### W:weight matrix for all layer  
%##### T: row is one data's label. {0,1}^10 
%##### X: batch data
 
    T=T';
    X=X';
    G=H{2}-T;
    G_bias{2}=G*ones(size(G,2),1);
    G_W{2}=G*H{1}';
    G=W{2}'*G;
    G=G.*sigmoid(H{1}).*(1-sigmoid(H{1}));
    G_bias{1}=G*ones(size(G,2),1);
    G_W{1}=G*X';
    %{
    %==ouput to hiddent ========= 
    g=h{2}-target;        
    grad_b{2}=g;%==== lock normalize term    
    grad_W{2}=g*h{1}';%%==== lock normalize term        
    g=W{2}'*g;
    %===============    
    
    %======hiddent to input=========    
    g=g.*sigmoid(h{1}).*(1-sigmoid(h{1}));            
    grad_b{1}=g;%==== lock normalize term        
    grad_W{1}=g*x';%%==== lock normalize term  
    %}
end