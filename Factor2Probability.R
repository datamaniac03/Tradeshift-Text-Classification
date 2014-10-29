#Factor transformation to probabilities
Factor2Probability <- function(factorVector, probsOnly = FALSE, vectorOnly = FALSE, probVectorInput = NULL){
  if (length(probVectorInput) > 0){
    probabilities <- probVectorInput
  }else{
    probabilities <- as.data.frame(table(factorVector))
    probabilities$Freq <- probabilities$Freq / length(factorVector)        
  }
  if (probsOnly == TRUE){
    return(probabilities)
  } 
  idx <- match(factorVector, probabilities$factorVector)
  probVector <- probabilities[idx, 'Freq']
  probVector[idx == NA] <- min(probVector, na.rm = TRUE) #if a novel factor is found, round it to the lowest possible probability
  probVector[factorVector == ''] <- -999
  if (vectorOnly == TRUE){
    return(probVector)    
  }else{
    return(list(probabilities, probVector))
  }  
}
