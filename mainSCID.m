clear all
close all

path = 'C:\Users\Public\QualityAssessment\SCID\DistortedSCIs';

A = importdata('C:\Users\Public\QualityAssessment\SCID\MOS_SCID.txt');
Score    = A.data;
NameList = A.textdata;
DistNameList = NameList(:,2);

Regression = 'rqgpr';

numberOfDistImages = size(Score, 1);

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

Features = zeros(numberOfDistImages, Constants.length);

for i=1:numberOfDistImages
    if(mod(i,100)==0)
        disp(i);
    end
    imgDist = imread(strcat(path, filesep, DistNameList{i}, '.bmp'));
    Features(i,:) = getFeatureVector(imgDist, Constants.extended, Constants.perceptual);
end

numberOfSplits = 1000;
PLCC = zeros(1,numberOfSplits); SROCC = zeros(1,numberOfSplits); KROCC = zeros(1,numberOfSplits);

parfor i=1:numberOfSplits
    rng(i);
    disp(i);
    [Train, Test] = splitTrainTestSCID(DistNameList);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);    
   
    YTest = Score(Test);
    YTrain= Score(Train);

    if( strcmp(Regression, 'rbfsvr') )
        Mdl = fitrsvm(TrainFeatures, YTrain', 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    elseif( strcmp(Regression, 'linsvr') )
        Mdl = fitrsvm(TrainFeatures, YTrain', 'KernelFunction', 'linear', 'Standardize', true);
    elseif( strcmp(Regression, 'rqgpr') )
        Mdl = fitrgp(TrainFeatures, YTrain', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    elseif( strcmp(Regression, 'tree') )
        Mdl = fitrtree(TrainFeatures, YTrain');
    elseif( strcmp(Regression, 'forest') )
        Mdl = fitrensemble(TrainFeatures, YTrain');
    else
        error('Not defined regression algorithm');
    end
    
    Pred= predict(Mdl,TestFeatures);
    
    eval = metric_evaluation(Pred, YTest);
    PLCC(i) = eval(1);
    SROCC(i)= eval(2);
    KROCC(i)= eval(3);
end

disp('----------------------------------');
X = ['Median PLCC after 1000 random train-test splits: ', num2str(round(median(PLCC(:)),3))];
disp(X);
X = ['Median SROCC after 1000 random train-test splits: ', num2str(round(median(SROCC(:)),3))];
disp(X);
X = ['Median KROCC after 1000 random train-test splits: ', num2str(round(median(KROCC(:)),3))];
disp(X);

disp('----------------------------------');
X = ['Std PLCC after ', num2str(numberOfSplits), ' random train-test splits: ', num2str(round(std(PLCC(:)),3))];
disp(X);
X = ['Std SROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(std(SROCC(:)),3))];
disp(X);
X = ['Std KROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(std(KROCC(:)),3))];
disp(X);