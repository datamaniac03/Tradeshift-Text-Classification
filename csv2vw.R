csv2vw <- function(csvFile, Id, labels, outputFile = NULL,
                   commaSeparated = TRUE, fullTextInput = TRUE, inMemory = TRUE){
  

#  Features have to be in the form of:
#  [Label] [Importance [Tag]]|Namespace Features |Namespace Features ... |Namespace Features
  
  IdIdx <- which(names(csvFile[1:100, ]) == Id)
  LabelIdx <- which(names(csvFile[1:100, ]) %in% labels)
  #ImportanceIdx <- which(names(csvFile[1:10, ]) == Importance)
  #TagIdx <- which(names(csvFile[1:10, ]) == Tag)
  
  #determine column classes
  dataClasses <- sapply(csvFile[1:100, ], class)
  #determine numeric features
  numericIdx <- which(dataClasses == "numeric" | dataClasses == "integer")
  if (class(csvFile[1:100, IdIdx]) == "numeric" | class(csvFile[1:100, IdIdx]) == "integer"){
    numericIdx <- numericIdx[-which(names(numericIdx) == Id)]  
  }  
  if (class(csvFile[1:100, LabelIdx[1]]) == "numeric" | class(csvFile[1:100, LabelIdx[1]]) == "integer"){
    numericIdx <- numericIdx[!(names(numericIdx) %in% labels)]
  }
  #determine categorical features
  categoricalIdx <- which(dataClasses == "character" | dataClasses == "factor")
  if (class(csvFile[1:100, IdIdx]) == "character" | class(csvFile[1:100, IdIdx]) == "factor"){
    categoricalIdx <- categoricalIdx[-which(names(categoricalIdx) == Id)]  
  }  
  #Vector with all columns names
  dataNames <- names(csvFile[1:100, ])
  
  vwText <- apply(csvFile, 1, function(csvLine, numIdx, catIdx, namesVec, idVec, labelsCols){
    #numeric features    
    numericVector <- apply(cbind(names(csvLine[numIdx]), as.numeric(csvLine[numIdx])), 1, function(lin){
      return(paste0(lin[1], ":", lin[2]))
    })
    numericVector <- paste(numericVector, collapse = " ")
    
    #categorical features    
    catVector <- apply(cbind(names(csvLine[catIdx]), as.character(csvLine[catIdx])), 1, function(lin){
      return(paste0(lin[1], ":", lin[2]))
    })
    catVector <- paste(catVector, collapse = " ")
    
    #Labels
    labelVector <- sapply(which(as.character(csvLine[labelsCols]) == "1"), function(lin){
      return(paste0(lin, ":", 1))
      })
    
    labelVector <- paste(labelVector, collapse = " ")
        
    vwLine <- paste0(
      labelVector,
      " ",
      csvLine[idVec],
      " |i ",  
      numericVector, 
      " |c ",
      catVector
    )
    
  }, numIdx = numericIdx, catIdx = categoricalIdx, namesVec = dataNames, 
  idVec = IdIdx, labelsCols = LabelIdx)
  
  write.table(vwText, file = outputFile)
  #return(vwText)
}
