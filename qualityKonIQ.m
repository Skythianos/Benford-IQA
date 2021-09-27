clear all
close all

load KonIQ10k.mat    % This mat file contains information about the database (KonIQ-10k)

Directory = 'C:\Users\Public\QualityAssessment\KonIQ-10k\1024x768';  % path to KonIQ-10k database 
numberOfImages = size(mos,1);   % number of images in KonIQ-10k database
numberOfTrainImages = round( 0.8*numberOfImages );   % appx. 80% of images is used for training
numberOfSplits = 1000;

Constants.Regression = 'rqgpr';

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

disp('Feature Extraction');
for i=1:numberOfImages
    img           = imread( strcat(Directory, filesep, names{i}) );
    tmp = getFeatureVector(img, Constants.extended, Constants.perceptual);
    Features(i,:) = tmp;
end

PLCC = zeros(1, numberOfSplits);
SROCC= zeros(1, numberOfSplits);
KROCC= zeros(1, numberOfSplits);

disp('Training and Testing');
for i=1:numberOfSplits
    rng(i);
    if(mod(i,10)==0)
        disp(i);
    end
    
    p = randperm(numberOfImages);
    
    Data_1 = Features(p,:);
    Target = mos(p);
    
    Train_1 = Data_1(1:round(numberOfImages*0.8),:);
    TrainLabel = Target(1:round(numberOfImages*0.8));
    
    Test_1  = Data_1(round(numberOfImages*0.8)+1:end,:);
    TestLabel = Target(round(numberOfImages*0.8)+1:end);
    
    if( strcmp(Constants.Regression, 'rbfsvr') )
        Mdl_1 = fitrsvm(Train_1, TrainLabel', 'KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);
    elseif( strcmp(Constants.Regression, 'linsvr') )
        Mdl_1 = fitrsvm(Train_1, TrainLabel', 'KernelFunction', 'linear', 'Standardize', true);
    elseif( strcmp(Constants.Regression, 'rqgpr') )
        Mdl_1 = fitrgp(Train_1, TrainLabel', 'KernelFunction', 'rationalquadratic', 'Standardize', true);
    elseif( strcmp(Constants.Regression, 'tree') )
        Mdl_1 = fitrtree(Train_1, TrainLabel');
    elseif( strcmp(Constants.Regression, 'forest') )
        Mdl_1 = fitrensemble(Train_1, TrainLabel');
    else
        error('Not defined regression algorithm');
    end
    
    Pred= predict(Mdl_1, Test_1);
    
    beta(1) = max(TestLabel); 
    beta(2) = min(TestLabel); 
    beta(3) = mean(TestLabel);
    beta(4) = 0.5;
    beta(5) = 0.1;
    
    [bayta,ehat,J] = nlinfit(Pred,TestLabel,@logistic,beta);
    [pred_test_mos_align, ~] = nlpredci(@logistic,Pred,bayta,ehat,J);
    
    PLCC(i) = corr(pred_test_mos_align,TestLabel);
    SROCC(i)= corr(Pred,TestLabel,'Type','Spearman');
    KROCC(i)= corr(Pred,TestLabel,'Type','Kendall');
end

disp('----------------------------------');
X = ['Median PLCC after ', num2str(numberOfSplits), ' random train-test splits: ', num2str(round(median(PLCC(:)),3))];
disp(X);
X = ['Median SROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(median(SROCC(:)),3))];
disp(X);
X = ['Median KROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(median(KROCC(:)),3))];
disp(X);

disp('----------------------------------');
X = ['Std PLCC after ', num2str(numberOfSplits), ' random train-test splits: ', num2str(round(std(PLCC(:)),3))];
disp(X);
X = ['Std SROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(std(SROCC(:)),3))];
disp(X);
X = ['Std KROCC after ', num2str(numberOfSplits),' random train-test splits: ', num2str(round(std(KROCC(:)),3))];
disp(X);
