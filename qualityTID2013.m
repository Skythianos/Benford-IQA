clear all
close all

load TID2013_Data.mat

pathDistorted = 'C:\Users\Public\QualityAssessment\tid2013\distorted_images'; % PATH TID2013
pathReference = 'C:\Users\Public\QualityAssessment\tid2013\reference_images'; % PATH TID2013

numberOfImages = size(dmos, 1);
Scores = zeros(numberOfImages, 1);

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

Features = zeros(numberOfImages, Constants.length);

for i=1:numberOfImages
    if(mod(i,100)==0)
        disp(i);
    end
    distortedImageName = moswithnames{i};
    distortedImagePath = strcat(pathDistorted, filesep, distortedImageName);
    
    tmp = char(distortedImageName);
    tmp = upper(tmp(1:3));
    tmp = string(tmp);
    
    referenceImageName = strcat(tmp,'.BMP');
    referenceImagePath = strcat(pathReference, filesep, referenceImageName);
    
    try
        imgDist = imread(distortedImagePath);
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
            distortedImageName(1) = 'I';
            distortedImagePath = strcat(pathDistorted, filesep, distortedImageName);
            imgDist = imread(distortedImagePath);
        end
    end
    
    try
        imgRef  = imread(referenceImagePath);
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist' ))
            referenceImagePath = strcat(pathReference, filesep, 'i25.bmp');
            imgRef  = imread(referenceImagePath);
        end
    end
    Features(i,:)=getFeatureVector(imgDist, Constants.extended, Constants.perceptual);
end

PLCC = zeros(1,1000); SROCC = zeros(1,1000); KROCC = zeros(1,1000);

parfor i=1:1000
    disp(i);
    rng(i);
    [Train, Test] = splitTrainTest_TID2013(moswithnames);

    TrainFeatures = Features(Train,:);
    TestFeatures  = Features(Test,:);
    
    YTest = dmos(Test);
    YTrain= dmos(Train);

    %Mdl = fitrsvm(TrainFeatures, YTrain, 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    Mdl = fitrgp(TrainFeatures, YTrain, 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    Pred= predict(Mdl,TestFeatures);
    
    eval = metric_evaluation(Pred, YTest');
    PLCC(i) = eval(1);
    SROCC(i)= eval(2);
    KROCC(i)= eval(3);
end

disp('----------------------------------');
X = ['Median PLCC after ', num2str(1000), ' random train-test splits: ', num2str(round(median(PLCC(:)),3))];
disp(X);
X = ['Median SROCC after ', num2str(1000),' random train-test splits: ', num2str(round(median(SROCC(:)),3))];
disp(X);
X = ['Median KROCC after ', num2str(1000),' random train-test splits: ', num2str(round(median(KROCC(:)),3))];
disp(X);

disp('----------------------------------');
X = ['Std PLCC after ', num2str(1000), ' random train-test splits: ', num2str(round(std(PLCC(:)),3))];
disp(X);
X = ['Std SROCC after ', num2str(1000),' random train-test splits: ', num2str(round(std(SROCC(:)),3))];
disp(X);
X = ['Std KROCC after ', num2str(1000),' random train-test splits: ', num2str(round(std(KROCC(:)),3))];
disp(X);


%save CurveletQA.mat PLCC SROCC KROCC

%figure;boxplot([PLCC',SROCC',KROCC'],{'PLCC','SROCC','KROCC'});
%saveas(gcf,'CurveletQA.png');

