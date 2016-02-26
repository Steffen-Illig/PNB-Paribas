# Load Libraries - note that you might have to install the package first, e.g. enter "install.packages('randomForest')" in your R console
require(xgboost)
require(ggplot2)
require(readr)
require(caret)  # Not yet implemented, cv is done with the xgboost cv function xgb.cv
require(Matrix)

# Clear work space
rm(list = ls())

# Set working directory - save the Prudential train.csv and test.csv there
setwd("C:\\Users\\si278\\Desktop\\PNB_Paribas")

# Start the clock
start_time <- Sys.time()

# Load data
train = read.csv("train.csv")
test = read.csv("test.csv")
submission = read.csv("sample_submission.csv")

### Clean and Prepare Data ##################################################################################################

# Stripping off ID columns and Target Variable
train_tar <- train$target
train$ID = test$ID = train$target = NULL

# Bind train and test data
data <- rbind(train, test)

# Deal with NA values (set to some extreme value)
for(i in 1:ncol(data)){
  if(is.numeric(data[,i])){
    data[is.na(data[,i]),i] = min(-1, min(data[,1], na.rm=T) - 1)
  }else{
    data[,i] = as.character(data[,i])
    data[is.na(data[,i]),i] = "NAvalue"
    data[,i] = as.factor(data[,i])
  }
}

# Clean variables with too many categories; own level for empty categorial data 
for(i in 1:ncol(data)){
  if(!is.numeric(data[,i])){
    freq = data.frame(table(data[,i]))
    freq = freq[order(freq$Freq, decreasing = TRUE),]     # Clever ordering
    data[,i] = as.character(match(data[,i], freq$Var1[1:30]))
    data[is.na(data[,i]),i] = "RareValue"
    data[,i] = as.factor(data[,i])
    # If the following line pops up this is a sign that feature creation might be beneficial for this variable
    if(length(freq$Var1) > 30){cat("!!! Large no of levels in data column: ", i)} 
  }
}

# Transform Data in Sparse Matrix - transforms categorial data into one-hot-encoding
data_sparse <- sparse.model.matrix(~.-1, data = data)   # -1 in the formula removes the intercept
if (!nrow(data) == nrow(data_sparse)){cat("!!! Rows missing in sparse matrix - INVASTIGATE !!!")}

# Rewrite as train and test data sets
train <- data_sparse[1:nrow(train),]
test <- data_sparse[(nrow(train)+1):nrow(data_sparse),]
rm(data); rm(data_sparse); rm(freq); 

### Subsetting Data for CV ##################################################################################################

#train <- cbind(train_tar, train)

# Set a random seed for reproducibility
#set.seed(1)
#train$random <- runif(nrow(train))

# Apply changes to the original data set here

#train_tr <- train[train$random <= 0.75,] 
#train_cv <- train[train$random > 0.75,]

#train_cv_tar <- train_cv$target
#train_tr_tar <- train_tr$target

#train_cv$target = train_tr$target = train_cv$random = train_tr$random = NULL


### CrossValidation ##############################################################################################

# xgboost parameters
param0 <- list("objective" = "binary:logistic",    # multiclass classification 
               "eval_metric" = "logloss",          # evaluation metric 
               "max_depth" = 10,                   # maximum depth of tree 
               "eta" = 0.04,                       # step size shrinkage 
               "subsample" = 0.9,                  # part of data instances to grow tree 
               "colsample_bytree" = 0.9,           # subsample ratio of columns when constructing each tree 
               "min_child_weight" = 1              # minimum sum of instance weight needed in a child 
)

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
  model_cv = xgb.cv(params = param0, nrounds = iter, nfold = 4, data = xgtrain, early.stop.round = 20, maximize = FALSE, nthread = 8)
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1  # For some reason this gives the next index
}

xgtrain = xgb.DMatrix(as.matrix(train), label = train_tar)
cat("Training a XGBoost classifier with cross-validation\n")
print(difftime(Sys.time(), start_time, units = 'sec'))
set.seed(1)
cv_iter <- docv(param0, 500) 


### Model Training and Prediction ####################################################################################

# Train model on all trining data and predict test data
TrainModel <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(nrounds = iter, params = param0, data = xgtrain, watchlist = watchlist, print.every.n = 20, nthread = 8)
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

# Bagging xgboost
cat("Bagging final XGBoost classifieres and do prediction\n")
cat("Time: ", difftime( Sys.time(), start_time, units = 'sec'), "\n")
xgtest = xgb.DMatrix(as.matrix(test))
ensemble <- rep(0, nrow(test))
for (i in 1:10) {
  cat("Running round: ", i, "\n")
  set.seed(i)
  p <- TrainModel(param0, cv_iter) 
  ensemble <- ensemble + p
  cat("Time: ", difftime( Sys.time(), start_time, units = 'sec'), "\n")
}


### Understand the Model ##########################################################################################

# Compute feature importance matrix und plot (gives feature weight information)
# importance_matrix <- xgb.importance(names(train_70), model = my_classifier)
# xgb.plot.importance(importance_matrix[1:10,])


### Write Submission File #########################################################################################

submission$PredictedProb <- ensemble/i
write.csv(submission, "Submission_File_Bag10.csv", row.names=F, quote=F)
summary(submission$PredictedProb)


### LogLoss Function ##############################################################################################

MultiLogLoss <- function(act, pred) {
  eps = 1e-15;
  nr <- length(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(length(act))      
  return(ll);
}
