clear all
close all

load DMOS_SIQAD.mat

pathRef = 'C:\Users\Public\QualityAssessment\SIQAD\references';
pathDist= 'C:\Users\Public\QualityAssessment\SIQAD\DistortedImages';

dmos = DMOS(:);

numRef = size(DMOS,2);
numDist= size(DMOS,1);

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

Features = zeros(numRef*numDist, Constants.length);
Train    = false(1, numRef*numDist);
Test     = false(1, numRef*numDist);

k=1;
for i=1:numRef
    disp(k);
    imgRef = imread(strcat(pathRef, filesep, 'cim', num2str(i), '.bmp'));
    for ii=1:7
        for jj=1:7
            imgDist = imread(strcat(pathDist, filesep, 'cim', num2str(i), '_', num2str(ii), '_', num2str(jj), '.bmp'));
            Features(k,:)=getFeatureVector(imgDist, Constants.extended, Constants.perceptual);
            k=k+1;
        end
    end   
end

PLCC = zeros(1,1000);
SROCC= zeros(1,1000);
KROCC= zeros(1,1000);

for ind=1:1000
    disp(ind);
    rng(ind);
    tmp = randperm(20);
    train = tmp(1:14);

    Train= false(1, numRef*49);
    Test = false(1, numRef*49);
    
    k=1;
    for i=1:numRef
        for j=1:49
            if(ismember(i,train))
                Train(k)=true;
            else
                Test(k)=true;
            end
            k=k+1;
        end
    end
    
    YTrain = dmos(Train);
    YTest  = dmos(Test);
    
    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    Mdl = fitrgp(TrainFeatures, YTrain, 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    Pred= predict(Mdl,TestFeatures);
    
    eval = metric_evaluation(Pred, YTest');
    PLCC(ind) = eval(1);
    SROCC(ind)= eval(2);
    KROCC(ind)= eval(3);
end

disp(round(median(PLCC),3));
disp(round(median(SROCC),3));
disp(round(median(KROCC),3));