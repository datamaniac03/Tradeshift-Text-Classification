#Factor transformation to probabilities
Factor2Probability <- function(factorVector, probsOnly = FALSE, vectorOnly = FALSE, probVectorInput = NULL){
  if (length(probVectorInput) > 0){
    probabilities <- probVectorInput
  }else{
    probabilities <- table(factorVector) / length(factorVector)        
  }
  if (probsOnly == TRUE){
    return(probabilities)
  } 
  idx <- match(factorVector, names(probabilities))
  probVector <- probabilities[idx]
  probVector[idx == NA] <- mean(probVector, na.rm = TRUE)  
  probVector[names(probVector) == ''] <- -999
  if (vectorOnly == TRUE){
    return(probVector)    
  }else{
    return(list(probabilities, probVector))
  }  
}
