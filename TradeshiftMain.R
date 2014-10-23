#TradeshiftMain
#Ver 0.1

#Init---------------------------
rm(list=ls(all=TRUE))

#Libraries----------------------
require('ggplot2')

#Set Working Directory----------
workingDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Tradeshift/Tradeshift/Data/'

#Load Data----------------------
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE)
trainLabels <- read.csv(paste0(dataDirectory, 'trainLabels.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

#Make a Unique training .csv file that will work on h2o
save(cbind(train, trainLabels), file = paste0(dataDirectory, 'trainWithLabels.csv'))

#EDA............................
