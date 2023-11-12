clear
clc
load('scene-data.mat')
[n,d] = size(train_data);
[~,m] = size(train_target');

%% feature selection
alpha=1; beta=1; gamma=1; k=round(d*0.2);
W=rand(d,m);
V=rand(n,m);
[Fs] = LRDG(train_data,train_target',k,alpha,beta,gamma,V,W);

%% test
Num=10;
Smooth=1;
train_data3=train_data(:,Fs);
test_data3=test_data(:,Fs);

% Invoking the training procedure
[Prior,PriorN,Cond,CondN]=MLKNN_train(train_data3,train_target,Num,Smooth); 

% Performing the test procedure
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1,~,~]=MLKNN_test(train_data3,train_target,test_data3,test_target,Num,Prior,PriorN,Cond,CondN); 
resultLRDG=[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,macrof1,microf1];