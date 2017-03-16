    function [Tps, Fps] = ROC(scores, labels,interval,img_name)  
       
    %% Sort Labels and Scores by Scores  
    sl = [scores; labels];  
    [d1 d2] = sort(sl(1,:));  
       
    sorted_sl = sl(:,d2);  
    s_scores = sorted_sl(1,:);  
    s_labels = round(sorted_sl(2,:));  
       
    %% Constants  
    counts = histc(s_labels, unique(s_labels));  
    negCount = counts(1);  
    posCount = counts(2);  
       
    %% Shift threshold to find the ROC  
    for thresIdx = 1:interval:(size(s_scores,2)+1)  
       
        % for each Threshold Index  
        tpCount = 0;  
        fpCount = 0;  
       
        for i = [1:size(s_scores,2)]  
       
            if (i >= thresIdx)           % We think it is positive  
                if (s_labels(i) == 1)   % Right!  
                    tpCount = tpCount + 1;  
                else                    % Wrong!  
                    fpCount = fpCount + 1;  
                end  
            end  
       
        end  
       
        Tps(thresIdx) = tpCount/posCount;  
        Fps(thresIdx) = fpCount/negCount;  
       
    end  
       
    %% Draw the Curve  
       
    % Sort [Tps;Fps]  
    x = Fps;  
    y = Tps;  

    plot(x,y,'.'); 
    hold on;
    f=getframe(gcf);
    imwrite(f.cdata,[img_name '.jpg']);