#TradeshiftMain
#Ver 0.1

#Init---------------------------
rm(list=ls(all=TRUE))

#Libraries and extra functions----------------------
require('ggplot2')
require('h2o')
require('caret')

source(paste0(workingDirectory, 'TreeFinder.R'))
source(paste0(workingDirectory, 'Factor2Probability.R'))

#Set Working Directory----------
workingDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/Data/'

#Load Data----------------------
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE)
trainLabels <- read.csv(paste0(dataDirectory, 'trainLabels.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

#Files Transformations----------
#get factors to transform to probabilities
longFactors <- names(train[, (sapply(train[1:1000, ], class) == 'character') 
                           & (sapply(train[1, ], nchar) > 20)])

shortFactors <- names(train[, (sapply(train[1:1000, ], class) == 'character') 
                           & (sapply(train[1, ], nchar) < 20)])

#Mine all factors' probabilities from training set (this could be also done with the test
#set appended but this is not often the case in real life)
probsOfAFactor <- apply(train[, longFactors], 2, Factor2Probability, probsOnly = TRUE,
                        vectorOnly = FALSE)

#Transform to probabilities
for (i in 1:length(longFactors)){
  train[, longFactors[i]] <- Factor2Probability(train[, longFactors[i]], probsOnly = FALSE,
                                                vectorOnly = TRUE, probVectorInput = probsOfAFactor[[longFactors[i]]])
}
for (i in 1:length(longFactors)){
  test[, longFactors[i]] <- Factor2Probability(test[, longFactors[i]], probsOnly = FALSE,
                                                vectorOnly = TRUE, probVectorInput = probsOfAFactor[[longFactors[i]]])
}
for (i in 1:length(shortFactors)){
  train[, shortFactors[i]] <- as.factor(train[, shortFactors[i]])
}
for (i in 1:length(shortFactors)){
  test[, shortFactors[i]] <- as.factor(test[, shortFactors[i]])
}

#Make .csv files
write.csv(train, file = "trainProbs.csv", row.names = FALSE)
write.csv(test, file = "testProbs.csv", row.names = FALSE)

#EDA----------------------------
trainSmall <- read.csv(paste0(dataDirectory, 'train.csv'), nrows = 20000, header = TRUE, stringsAsFactors = FALSE)
trainLabelsSmall <- read.csv(paste0(dataDirectory, 'trainLabels.csv'), nrows = 20000, header = TRUE, stringsAsFactors = FALSE) 
testSmall <- read.csv(paste0(dataDirectory, 'test.csv'), nrows = 20000, header = TRUE, stringsAsFactors = FALSE)

#Hyperparameter Tuning----------
#Cross Validation Control Params
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = c(1, 3, 5),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 1500)

#Hiper parameter 5-fold Cross-validation "Ca"
set.seed(1005)
randomSubset <- sample.int(nrow(train), nrow(train)) #full data

gbmMODCa <- train(form = Ca~., 
                  data = train[randomSubset , c(allSpectralDataNoCO2, spatialPredictors, depthIx, 3586)],
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  trControl = GBMControl,
                  distribution = 'gaussian',
                  nTrain = floor(nrow(train) * 0.7),
                  verbose = TRUE)

#Modelling----------------------

## Launch H2O 
h2oServer <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '13g', startH2O = TRUE, nthreads = -1)

trainHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'train.csv'))
trainLabelHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'trainLabels.csv'))
testHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'test.csv'))


