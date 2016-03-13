####
#random forest example
#######

library(randomForest)
samp <- sample(nrow(df),.7*nrow(df))
test <- df[-samp,]
train <- df[samp,]
model<-randomForest(COL~. - quality, data = train)
pred<-predict(model, newdata = test)
table(pred,test$COL)
(418+197+571)/nrow(test)
