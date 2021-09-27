clear all
close all

load dmosESPL.mat

path = 'C:\Users\Public\QualityAssessment\ESPL_Synthetic_v2';

distImageList = imageList(26:end);
refImageList  = imageList(1:25);

numDist = size(distImageList, 2);
numRef  = size(refImageList, 2);

Constants.extended  = false;
Constants.perceptual= false;

if(Constants.extended)
    if(Constants.perceptual)
        Constants.length=95;
    else
        Constants.length=90;
    end
else
    if(Constants.perceptual)
        Constants.length=59;
    else
        Constants.length=54;
    end
end

Features = zeros(numDist, Constants.length);

dmos = dmos(26:end);

for i=1:numDist
    img = imread(strcat(path, filesep, distImageList{i}));
    Features(i,:) = getFeatureVector(img, Constants.extended, Constants.perceptual);
end

PLCC = zeros(1,1000);
SROCC= zeros(1,1000);
KROCC= zeros(1,1000);

for i=1:1000
    if(mod(i,50)==0)
        disp(i);
    end
    rng(i);
    p = randperm(numRef);
    train = p(1:numRef*0.8);
    Train = false(1,numDist);
    Test  = false(1,numDist);
    for j=1:numDist
        name = distImageList{j};
        tmp1 = str2num(name(4));
        tmp2 = str2num(name(5));
        if(isempty(tmp2))
            if(ismember(tmp1,train))
                Train(j)=true;
            else
                Test(j)=true;
            end
        else
            tmp = 10*tmp1+tmp2;
            if(ismember(tmp,train))
                Train(j)=true;
            else
                Test(j)=true;
            end
        end
    end
    trainFeatures = Features(Train,:);
    testFeatures  = Features(Test,:);
    trainLabel    = dmos(Train);
    testLabel     = dmos(Test);
    
    %Mdl = fitrsvm(trainFeatures, trainLabel, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    %Mdl = fitrsvm(trainFeatures, trainLabel, 'KernelFunction', 'linear', 'Standardize', true);
    Mdl = fitrgp(trainFeatures, trainLabel, 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    %Mdl = fitrtree(trainFeatures, trainLabel);
    %Mdl = fitrensemble(trainFeatures, trainLabel);
    
    Pred = predict(Mdl, testFeatures);
    
    beta(1) = max(testLabel); 
    beta(2) = min(testLabel); 
    beta(3) = mean(testLabel);
    beta(4) = 0.5;
    beta(5) = 0.1;
    
    [bayta,ehat,J] = nlinfit(Pred,testLabel',@logistic,beta);
    [pred_test_mos_align, ~] = nlpredci(@logistic,Pred,bayta,ehat,J);
    
    PLCC(i) = corr(pred_test_mos_align,testLabel');
    SROCC(i)= corr(Pred,testLabel','Type','Spearman');
    KROCC(i)= corr(Pred,testLabel','Type','Kendall');
end

disp(round(median(PLCC),3));
disp(round(median(SROCC),3));
disp(round(median(KROCC),3));

disp(round(std(PLCC),3));
disp(round(std(SROCC),3));
disp(round(std(KROCC),3));