function [output] = getExtendedFDD(in)

    benford = getBenfordsLaw();

    FDD = getFirstDigitDistribution(in);
    output = [FDD, getSymmetricKullbackLeibler(FDD, benford), skewness(FDD), kurtosis(FDD), ...
        entropy(FDD), median(FDD), std(FDD)];
    
end