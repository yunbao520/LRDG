%% Multi-label feature selection with latent representation learning and dynamic graph constraints 
%  
%
% Written by Yao Zhang (ayunxiaobao@163.com)
%
function [Fs] = LRDG(X,Y,k,alpha,beta,gamma,V,S)
%
%% Input:  
% X: data matrix (n x d)
% Y: label matrix (n x m)
% k: the number of the selected features
% alpha,beta,gamma: regularization parameter
% V: pseudo-label matrix (n x m)
% S: feature weight matrix (d x m)
%
%% Output:
% Fs: feature subset 

[~,AX,~] = cLLR_k(X,5);
[PY,AY,LY] = cLLR_k(Y,5);
[n,~] = size(X);
[~,m] = size(Y);
I=eye(n);
E=ones(n,1);
H=I-E*E'/n;
maxIte = 50;

% Initialization
for i=1:maxIte
    [PV,AV,LV] = cLLR_k(V',3);
    V=V.*((H*X*S+alpha*Y+2*beta*AX*V+gamma*AY*V)./(H*V+alpha*V+2*beta*(V*V')*V+gamma*PY*V));
    S=S.*((X'*H*V+gamma*S*AV)./(X'*H*X*S+gamma*S*PV));
    
    b=(V'*E-S'*X'*E)/n;
    
    % Objective function 
    J(i) = 1/(m*n)*(sum(sum((V-X*S-E*b').*(V-X*S-E*b')))+alpha*sum(sum((Y-V).*(Y-V)))+beta*sum(sum((AX-V*V').*(AX-V*V')))+gamma*(trace(S*LV*S')+trace(V'*LY*V)));
end
tempVector = sum(S.*S,2);
[~, value] = sort(tempVector, 'descend');
clear tempVector;
Fs= value(1:k);
end
    
   
    
