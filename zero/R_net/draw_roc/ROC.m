function [Tps, Fps] = ROC(scores, ground_truth)  
%probe = load('C:\Users\pc\Desktop\feature\Fea_120000.txt');
%gallery= load('C:\Users\pc\Desktop\feature\IDcardFea_120000.txt');
%lines= load('C:\Users\pc\Desktop\feature\test_2Dtexture.txt');
%scores=ones(2580,1);
%ground_truth=ones(2580,1);
m=length(scores);
pos_sum=0;
for i=1:m
   %scores(i)= dot(probe(i,:),gallery(i,:))/(norm(probe(i,:))*norm(gallery(i,:)));
   %ground_truth(i)=lines(i,2);
   if ground_truth(i)==1
       pos_sum=pos_sum+1;
   end
end
[pre,Index] = sort(scores,'descend');
neg_sum=m-pos_sum;
for i=1:m
        ground_truth(i)=ground_truth(Index(i));
end
    x=zeros(m+1,1);
    y=zeros(m+1,1);
    auc=0;
    x(1)=1;
    y(1)=1;
    
    for i=2:m
        %TP=sum(ground_truth(i:m)==1);
        %FP=sum(ground_truth(i:m)==0);
        TP=0;FP=0;
        for j=i:m     
            if ground_truth(j)==1
                TP=TP+1;
            else
                FP=FP+1;
            end
        end
        x(i)=FP/neg_sum;
        y(i)=TP/pos_sum;
        auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
    end
    x(m+1)=0;
    y(m+1)=0;
    auc=auc+y(m+1)*x(m+1)/2;   
    plot(x,y,'-r');
    xlabel('FPR');ylabel('TPR');
    title('ROC curve');