TreeFinder <- function(model, dataNew, maxIt = 10000, threshold = 0.005, returnModel = FALSE){
  #External Libraries
  require('gbm')
  
  #Find optimal number of trees
  trees <- gbm.perf(model, method = ifelse(length(model$valid.error)  > 0, 'test', 'OOB'))
  treesIterated <-  model$n.trees
  maxTrees <- model$n.trees
  
  while(trees >= treesIterated - 20 & (abs(model$valid.error[treesIterated] - model$valid.error[treesIterated - 90]) > threshold)){
    # do another n iterations  
    model <- gbm.more(model, maxTrees, data = dataNew, verbose = TRUE)
    trees <- gbm.perf(model, method = 'test')
    treesIterated <- treesIterated + maxTrees
    
    print(paste0('Run another ', maxTrees, ' iterations'))
    
    if(treesIterated >= maxIt){break}    
  }
  
  if (returnModel == TRUE){
    return(list(trees, model))
  }else{
    return(trees)
  }  
}


