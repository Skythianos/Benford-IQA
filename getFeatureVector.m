function [output] = getFeatureVector(imgRGB, extended, perceptual)
    
    if(isrgb(imgRGB))
        imgGray = im2double(rgb2gray(imgRGB));
    else
        imgGray = im2double(imgRGB);
    end     
    [~,cH,cV,cD] = dwt2(imgRGB,'sym4','mode','per');
    [DCT]        = dct2(imgGray);
    [~,S,~]      = svd(imgGray);
    sls = shearletSystem('ImageSize',[size(imgGray,1) size(imgGray,2)],'FilterBoundary','truncated');
    cfs = sheart2(sls, imgGray);
    if(extended==true)
        output = [getExtendedFDD(cH), getExtendedFDD(cV), getExtendedFDD(cD), ...
            getExtendedFDD(DCT), getExtendedFDD(S), getExtendedFDD(abs(cfs))];
    elseif(extended==false)
        output = [getFirstDigitDistribution(cH), getFirstDigitDistribution(cV), getFirstDigitDistribution(cD), ...
            getFirstDigitDistribution(DCT), getFirstDigitDistribution(S), getFirstDigitDistribution(abs(cfs))];
    else
        error('Unknown option'); 
    end
    
    if(perceptual==true)
        PC = getPhaseCongruencyImage(imgRGB);
        P = [getColorfulness(imgRGB),getGlobalContrastFactor(imgRGB),getDarkChannelFeature(imgRGB), entropy(imgRGB), mean(PC(:))];
        output = [output, P];
    elseif(perceptual==false)
        
    else
        error('Unknown option'); 
    end

end

