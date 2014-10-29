#TradeshiftMain
#Ver 0.2

#Init-----------------------------------------------
rm(list=ls(all=TRUE))

#Libraries and extra functions----------------------
require('ggplot2')
require('caret')
require('Metrics')
require('RVowpalWabbit')

#Set Working Directory------------------------------
workingDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/'
setwd(workingDirectory)
dataDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/Data/'

source(paste0(workingDirectory, 'Factor2Probability.R'))
source(paste0(workingDirectory, 'csv2vw'))

#Load Data------------------------------------------
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE)
trainLabels <- read.csv(paste0(dataDirectory, 'trainLabels.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

#Files Transformations-----------------------------
#extract factors that'll be transformed to probabilities
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
for (i in 1:length(shortFactors)){
  train[, shortFactors[i]] <- as.factor(train[, shortFactors[i]])
}
for (i in 1:length(longFactors)){
  test[, longFactors[i]] <- Factor2Probability(test[, longFactors[i]], probsOnly = FALSE,
                                                vectorOnly = TRUE, probVectorInput = probsOfAFactor[[longFactors[i]]])
}
for (i in 1:length(shortFactors)){
  test[, shortFactors[i]] <- as.factor(test[, shortFactors[i]])
}

#Make .csv files
set.seed(1010101)
shuffledTrainIdx <- sample.int(nrow(train), nrow(train))
write.csv(train[shuffledTrainIdx , ], file = paste0(dataDirectory, 'trainProbs.csv'), row.names = FALSE)
write.csv(trainLabels[shuffledTrainIdx , ], file = paste0(dataDirectory, 'trainLabelsShuffled.csv'),
          row.names = FALSE)
write.csv(cbind(train[shuffledTrainIdx , ], trainLabels[shuffledTrainIdx , 2:dim(trainLabels)[2]]),
          file = paste0(dataDirectory, 'trainProbsFull.csv'), row.names = FALSE)
write.csv(test, file = paste0(dataDirectory, 'testProbs.csv'), row.names = FALSE)
rm(train)
rm(trainLabels)
rm(test)

#EDA-----------------------------------------------
train <- read.csv(paste0(dataDirectory, 'trainProbs.csv'), nrows = 10000, header = TRUE, stringsAsFactors = TRUE)
trainLabels <- read.csv(paste0(dataDirectory, 'trainLabelsShuffled.csv'), nrows = 10000, header = TRUE, stringsAsFactors = TRUE) 
test <- read.csv(paste0(dataDirectory, 'testProbs.csv'), nrows = 10000, header = TRUE, stringsAsFactors = TRUE)

#Hyperparameter Tuning-----------------------------
#Cross Validation Control Params

#Random Forrests h2o.ai
#possible targets
yMatrix <- names(trainLabels[1:10, ])[2:length(names(trainLabels[1:10, ]))]

require('h2o')
h2oServer <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '13g', startH2O = TRUE, nthreads = -1)
trainHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'trainProbsFull.csv'))

#new Idxs
set.seed(1010103)
shuffledTrainIdx <- sample.int(nrow(trainHEX), floor(nrow(trainHEX) * 0.8))
shuffledValIdx <- !(1:nrow(trainHEX) %in% shuffledTrainIdx)

hyperparametersRF <- sapply(yMatrix, function(target){  
  print(h2o.ls(h2oServer))
  cvmodel <- h2o.randomForest(x = seq(2, dim(trainHEX)[2] - 33),
                              y = as.character(target),
                              data = trainHEX[shuffledTrainIdx, ],
                              nfolds = 4,
                              classification = TRUE,
                              type = "fast",
                              mtries = c(floor(sqrt(dim(trainHEX)[2] - 33)), floor(sqrt(dim(trainHEX)[2] - 33)) * 2, 
                              floor(sqrt(dim(trainHEX)[2] - 33)) * 5))
  model <- cvmodel@model[[1]] #If cv model is a grid search model  
  predicted <- as.data.frame(h2o.predict(model, trainHEX[shuffledValIdx, ])[,3])
  logLossError <- logLoss(trainLabels[shuffledValIdx, ], predicted)
  print(logLoss)
  bestParameters <- cvmodel@model[[1]]@model$params
  print(h2o.ls(h2oServer))
  h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)$Key[1:(length(h2o.ls(h2oServer)$Key) - 2)])
  return(c(target, bestParameters, logLossError))
})
                             
#Vowpal Wabbit


#Modelling----------------------
#Random Forests
## Launch H2O 
require('h2o')
h2oServer <- h2o.init(ip = "localhost", port = 54321, max_mem_size = '13g', startH2O = TRUE, nthreads = -1)

#Load Data
trainHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'trainProbsFull.csv'))
testHEX <- h2o.importFile(h2oServer, path = paste0(dataDirectory, 'testProbs.csv'))

predictionsRF <- apply(hyperparametersRF, 1, function(hyperparameters, trainHex, 
                                                      testHex, labelsHex, nEnsembles){  
  
  if (as.character(hyperparameters[1]) == 'y14'){
    return(rep(0, dim(testHEX)[1]))
  }else{
    print(h2o.ls(h2oServer))
    RFModel <- h2o.randomForest(x = seq(2, dim(trainHex)[2] - 33),
                                y = as.character(hyperparameters[1]),
                                data = trainHex,
                                classification = TRUE,
                                type = "fast", 
                                mtries = floor(sqrt(dim(trainHEX)[2] - 33)))  
  
    GBMPrediction <- as.data.frame(h2o.predict(RFModel, newdata = testHex)[, 3])  
    print(h2o.ls(h2oServer))
    h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)$Key[1:(length(h2o.ls(h2oServer)$Key) - 2)])
    return(GBMPrediction)
  } 
}, trainHex = trainHEX, testHex = testHEX, labelsHex = trainLabelHEX)

#h2o shutdown WARNING, All data on the server will be lost!
h2o.shutdown(h2oServer, prompt = FALSE)

#Predictions Matrix
predictionsRF <- as.data.frame(predictionsRF)
names(predictionsRF) <- names(trainLabels)[2:length(trainLabels)]
#Write .csv 
submissionTemplate$pred <- as.vector(t(predictionsRF))
write.csv(submissionTemplate, file = "PredictionRFI.csv", row.names = FALSE)