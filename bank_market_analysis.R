library(ggplot2)
library(dplyr)
library(gmodels)
library(caret)
library(class)
library(glmnet)

bank_df <- read.csv("../bank-full.csv", sep=";", strings = T)
colnames(bank_df)
# EDA
head(bank_df)
unique(pout)
unique(bank_df$poutcome)
unique(bank_df$y)
dim(bank_df)
str(bank_df)
unique(bank_df$loan)
unique(bank_df$housing)
unique(bank_df$pdays)
unique(bank_df$campaign)
unique(bank_df$education)
unique(bank_df$marital)

unique(bank_df$contact)

bank_df %>% group_by(y) %>% summarise(count = n())
bank_df %>%
  group_by(y) %>%
  summarise(count = n()) 
bank_df %>%
  group_by(marital, y) %>%
  summarise(count = n()) %>%
  ggplot() +
  geom_bar(aes(x=marital, y = count, fill = y),stat="identity", show.legend = TRUE)

ggplot(data = bank_df) +
  geom_histogram(mapping = aes(x=age))

bank_df %>% filter(bank_df$education == "unknown") %>% summarise(count = n())
bank_df %>% filter(bank_df$job == "unknown") %>% summarise(count = n())

min(bank_df$age)
max(bank_df$age)
bank_df$y <- as.factor(ifelse(bank_df$y == 'yes', 1, 0))

#JOB
CrossTable(x=bank_df$job, y = bank_df$y)
bank_df <- bank_df %>% filter(bank_df$job != "unknown")

#education
CrossTable(x=bank_df$education, y = bank_df$y)
bank_df <- bank_df %>% filter(bank_df$education != "unknown")

#default
CrossTable(x=bank_df$default, y = bank_df$y)

#Housing
CrossTable(x=bank_df$housing, y = bank_df$y)


CrossTable(x=bank_df$loan, y = bank_df$y)
CrossTable(x=bank_df$month, y = bank_df$y)
CrossTable(x=bank_df$contact, y = bank_df$y)
CrossTable(x=bank_df$marital, y = bank_df$y)
CrossTable(x=bank_df$poutcome, y = bank_df$y)


unique(bank_df$month)
View(bank_df)
unique(bank_df$previous)

unique((bank_df$month))
# removing column duration as this field will not help if marketing team should make a call to the customer.
bank_df <- subset(bank_df, select = -c(duration))
summary(bank_df)

set.seed(1)
X <- model.matrix( ~ age + job + marital + education + balance + default + housing + loan +
                     contact + month + campaign + pdays + previous + poutcome, data=bank_df)
#X <- subset(bank_df, select = -c(y))
Y <- bank_df$y
index <- createDataPartition(Y, p=0.70, list=FALSE)
X_train <- X[index, ]
X_test <- X[-index, ]
Y_train <- Y[index]
Y_test <- Y[-index]
head(X_train)
dim(X_train)
dim(X_test)

# K-NN algorithm
K1 <- knn(train=X_train, test=X_test, cl=Y_train, k=1)
K2 <- knn(train=X_train, test=X_test, cl=Y_train, k=2)
(K1)
K3 <- knn(train=X_train, test=X_test, cl=Y_train, k=3)
K5 <- knn(train=X_train, test=X_test, cl=Y_train, k=5)
K7 <- knn(train=X_train, test=X_test, cl=Y_train, k=7)
K10 <- knn(train=X_train, test=X_test, cl=Y_train, k=10)
K15 <- knn(train=X_train, test=X_test, cl=Y_train, k=15)
K50 <- knn(train=X_train, test=X_test, cl=Y_train, k=50)
K100 <- knn(train=X_train, test=X_test, cl=Y_train, k=100)

res <- data.frame(Y_test, K1, K2, K3, K5, K7, K10, K15, K50, K100 )
(K3)
apply(res[,-1], 2, function(c) mean(c==res[,1]))

i <- 1
cm = 1
prec = array(numeric(), c(10, 0))
recal = array(numeric(), c(10, 0))
speci = array(numeric(), c(10, 0))
accu = array(numeric(), c(10, 0))

for( i in 1:10) {
  knn.mod <- knn(train=X_train, test=X_test, cl=Y_train, k=i)
  cm <- table(pred=knn.mod, actual=Y_test)
  prec[i] <- precision(cm)
  recal[i] <-recall(cm)
  speci[i] <- specificity(cm)
  cm1 <- confusionMatrix(data=knn.mod, reference = Y_test)
  accu[i] <- cm1$overall['Accuracy']

}
plot(prec)
plot(accu)
plot(recal)
plot(speci)
# as we increase the value of K, we see that although accuracy is increasing but True Positives decreased.
# This is perhaps because we have a data that is biased towards customer not accepting the offer. 
# Hence for k model, large value of K, may be counter intuitive .


# applying Lasso Logistic regression using glmnet
netfit <- glmnet(x = X_train, y = Y_train, family="binomial")
plot(netfit, xvar = "lambda", label=TRUE)
print(coef(netfit))
(coef(netfit, s=0.0003))
(LMIN <- min(netfit$lambda))
print(netfit$lambda)
pnet <- predict(netfit, newx = data.matrix(X_test), s=70,type="class")
unique(pnet)
table(actual=Y_test, pred=pnet)

#applying simple logistic regression using 
model <- glm(y ~.,family=binomial(link='logit'),data=bank_df[index,])
results <- predict(model,newdata=bank_df[-index,],type='response')
unique(results)
y_pred <- ifelse(results > 0.1,1,0)
cm1 <- table(y_pred, Y_test)
sensitivity(cm1)
specificity(cm1)
precision(cm1)
recall(cm1)

coef(model)
