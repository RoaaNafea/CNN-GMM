classdef GMMFullyConnectedLayer < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable
    
    properties
        OutputSize
        InputSize
    end

    properties(Learnable)
        MeanWeights
        RhoWeights
        MeanBias
        RhoBias

       
        Sigma1
        Sigma2
    end

    properties (State)
      
        LogPosterior
        LogPrior
    end

    methods
        function this = GMMFullyConnectedLayer(outputSize,nvargs)
            
            arguments
                outputSize
                nvargs.Name = ""
                nvargs.Sigma1 = 1.0
                nvargs.Sigma2 = 0.5
            end

            % Set output size.
            this.OutputSize = outputSize;

            % Set layer name.
            this.Name = nvargs.Name;

            % Set prior parameters.
            this.Sigma1 = nvargs.Sigma1;
            this.Sigma2 = nvargs.Sigma2;

            % Set number of layer inputs and outputs.
            this.NumInputs = 1;
            this.NumOutputs = 1;

            % Set initial values for log posterior and prior.
            this.LogPosterior = 0;
            this.LogPrior = 0;
        end

        function this = initialize(this, inputDataLayout)
            outputSize = this.OutputSize;

            sdims = finddim(inputDataLayout, 'S');
            cdim = finddim(inputDataLayout, 'C');
            inputSize = inputDataLayout.Size( [sdims cdim] );
            this.InputSize = inputSize;

            % initializeGlorot is attached as a supporting file.
            this.MeanWeights = initializeGlorot([outputSize inputSize], ...
                outputSize, prod(inputSize));
            this.RhoWeights = -2 + 1 .* rand([outputSize inputSize]);

            this.MeanBias = zeros(outputSize, 1);
            this.RhoBias = -2 .* ones(outputSize, 1);
        end

        function [Y, logPosterior, logPrior] = predict(this, X)
            [W, b] = this.samplePosterior();
            Y = fullyconnect(X, W, b);

            logPosterior = [];
            logPrior = [];
        end

        function [Y, logPosterior, logPrior] = forward(this, X)
            [W, b] = this.samplePosterior();
            Y = fullyconnect(X, W, b);

            logPosterior = this.estimateLogPosterior(W,b);
            logPrior = this.computeLogPrior(W,b);
        end
    end

    methods(Access = private)
        function logPrior = computeLogPrior(this, sampledWeights, sampledBias)
                        mixturePriorWeights = logScaleMixturePrior(sampledWeights, this.Sigma1, this.Sigma2) + eps;
            mixturePriorBias = logScaleMixturePrior(sampledBias, this.Sigma1, this.Sigma2) + eps;

            logMixturePrior = mixturePriorWeights + mixturePriorBias;
            logPrior = sum(logMixturePrior, 'all');
        end

        function logPosterior = estimateLogPosterior(this, sampledWeights, sampledBias)
            weightsSigma = iSigma(this.RhoWeights);
            logProbPosteriorWeights = sum(logProbabilityNormal(sampledWeights, this.MeanWeights, weightsSigma), 'all');

            biasSigma = iSigma(this.RhoBias);
            logProbPosteriorBias = sum(logProbabilityNormal(sampledBias, this.MeanBias, biasSigma));

            logPosterior = logProbPosteriorWeights + logProbPosteriorBias;
        end

        function [w,b] = samplePosterior(this)
            epsilonW = randn([this.OutputSize this.InputSize], "like", this.MeanWeights);
            epsilonB = randn([this.OutputSize 1], "like", this.MeanBias);

            w = this.MeanWeights + log(1 + exp(this.RhoWeights)) .* epsilonW;
            b = this.MeanBias + log(1 + exp(this.RhoBias)) .* epsilonB;
        end
    end
end

function sample = logScaleMixturePrior(x, sigma1, sigma2)
% This function produces a mixture distribution of Gaussian distributions
% with mixing proportion set by piScalar. Each distribution has mean 0 and
% variance defined by sigma1 and sigma2.
piScaler = 0.5;
p1 = piScaler .* logProbabilityNormal(x, 0, sigma1);
p2 = (1 - piScaler) .* logProbabilityNormal(x, 0, sigma2);
sample = p1 + p2;
end

function out = iSigma(rho)
out = log(1+exp(rho));
end
