# Load packages

library(rpart)
library(bartMachine)
library(neuralnet)
library(brnn)

#### Code assumes that the data set is loaded as a dataframe into the  variable dat ####

### Preprocessing

# Normalize the data

maxs = apply(dat, 2, max)
mins = apply(dat, 2, min)
dat_scaled= as.data.frame(scale(dat,center = mins, scale = maxs - mins))


# Train-test split in the ratio 70:30

set.seed(2019)
testrows = sample(1:nrow(dat),replace=F,size=0.3*nrow(dat))
train_scaled = dat_scaled[-testrows,]
test_scaled = dat_scaled[testrows,]


# Create formula 

## Feature vector
feats = names(train)[-which(names(train)=='y')]
## Concatenate strings
f = paste(feats,collapse=' + ')
f = paste('y ~',f)
## Convert to formula
f = as.formula(f)


### BNT-1

set.seed(2019)

# Grow tree on training set and predict on both training and testing sets

treemod = rpart(f,data=train_scaled,method="anova",minsplit=0.1*nrow(train_scaled)) 
train.pred = predict(treemod,train_scaled)
test.pred = predict(treemod,test_scaled)

# Extract features from RT
frame = treemod$frame
leaves = frame$var == "<leaf>"
used = unique(frame$var[!leaves])

# Create a new data frame with selected features from RT, predictions from RT and output Y - both for training and testing sets

features.select = names(dat)[(names(dat) %in% used)]
train2 = train_scaled[, features.select]
test2 = test_scaled[, features.select]
train2$pred = train.pred; train2$y = train_scaled$y
test2$pred = test.pred; test2$y = test_scaled$y


## Create a new formula with new feature set

feats.rt = names(train2)[-which(names(train2)=='y')]
f.rt = paste(feats.rt,collapse=' + ')
f.rt = paste('y ~',f.rt)
## Convert to formula
f.rt = as.formula(f.rt)


## Fit a BNN model 

k = rgeom(1,0.3)
rtbnn.mod =  brnn(f.rt,data=train2,neurons=k,normalize=F) ## normalize=F as data is already normalized


# Prediction on test set 

rtbnn.pred.scaled = predict(rtbnn.mod,test2)
pred = rtbnn.pred.scaled * (maxs[names(maxs)=='y']-mins[names(mins)=='y']) + mins[names(mins)=='y'] ## bring back to original scale



### BNT-2


set.seed(2019)

# Grow BCART tree on training set and predict on both training and testing sets

Y.train = train_scaled$y
X.train = train_scaled[,-which(names(train_scaled)=='y')]
X.test = test_scaled[,-which(names(test_scaled)=='y')]
bcart.mod = bartMachine(X.train,Y.train,num_trees=1)
v=var_selection_by_permute(bcart.mod,
                           num_permute_samples = 10)$important_vars_local_names
train.pred = predict(bcart.mod,X.train)
test.pred = predict(bcart.mod,X.test)


## Create a new df with selected features from BCART, predictions from BCART and output Y - both for training and test

features.select.bcart = names(dat)[(names(dat) %in% v)]
train3 = train_scaled[, features.select.bcart]
test3 = test_scaled[, features.select.bcart]
train3$pred = train.pred; train3$y = train_scaled$y
test3$pred = test.pred; test3$y = test_scaled$y


## Create new formula
feats.bcart = names(train3)[-which(names(train3)=='y')]
f.bcart = paste(feats.bcart,collapse=' + ')
f.bcart = paste('y ~',f.bcart)
## Convert to formula
f.bcart = as.formula(f.bcart)


## Fit an ANN with optimal number of neurons

ntrain = nrow(train3)
dm = length(features.select.bcart)+1
neurons = sqrt(ntrain/(dm*log(ntrain)))
bcartnn.mod =  neuralnet(f.bcart,data=train3,hidden=neurons) 


## Predict on test set

bcartnn.pred.scaled  = neuralnet::compute(bcartnn.mod,test3[,-which(names(test3) %in% c("y"))])$net.result
pred = bcartnn.pred.scaled * (maxs[names(maxs)=='y']-mins[names(mins)=='y']) + mins[names(mins)=='y']




