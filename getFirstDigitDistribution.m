function [dist] = getFirstDigitDistribution(img)

    [H,W,ch] = size(img);
         
    dist = zeros(1,9);
    
    for i=1:H
        for j=1:W
            for k=1:ch
                firstDigit = getFirstDigit(img(i,j,k));
                if(firstDigit~=0)
                    dist(firstDigit) = dist(firstDigit)+1;
                end
            end
        end
    end
    
    dist = dist ./ sum(dist(:));
end