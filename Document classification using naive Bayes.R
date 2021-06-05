library(tm)
library(dplyr)
library(e1071)
library(caret)
library(caTools)
library(wordcloud)
library(stringr)

df <- read.csv('IMDB Dataset.csv', stringsAsFactors = FALSE)
View(df)
glimpse(df)

set.seed(1)
df <- df[sample(nrow(df)), ]
df <- df[sample(nrow(df)), ]
glimpse(df)

df$sentiment = factor(df$sentiment,
                      levels = c("negative", "positive"),
                      labels = c(0,1))

table(df$sentiment)

df$review =str_replace_all(df$review,"[\\.\\,\\;]+", " ")
df$review =str_replace_all(df$review,"http\\w+", "")
df$review =str_replace_all(df$review,"@\\w+", " ")
df$review =str_replace_all(df$review,"[[:punct:]]", " ")
df$review =str_replace_all(df$review,"[[:digit:]]", " ")
df$review =str_replace_all(df$review,"^ ", " ")
df$review =str_replace_all(df$review,"[<].*[>]", " ") 

df$review = gsub("<[^>]+>", "", df$review)
df$review = gsub('[[:punct:][:blank:]]+',' ', df$review)

corpus <- VCorpus(VectorSource(df$review))
inspect(corpus[1:3])

corpus.clean <- corpus %>% 
  tm_map(removeNumbers)

View(corpus.clean)

dtm <- DocumentTermMatrix(corpus.clean)
inspect(dtm[40:50, 10:15])

df.train <- df[1:4000,]
df.test <- df[4001:5000,]

dtm.train <- dtm[1:4000,]
dtm.test <- dtm[4001:5000,]

corpus.clean.train <- corpus.clean[1:4000]
corpus.clean.test <- corpus.clean[4001:5000]

dim(dtm.train)

positive <- subset(df.train, sentiment == 1)
negative  <- subset(df.train, sentiment == 0)

wordcloud(positive$review, max.words = 40, scale = c(3, 0.5))
wordcloud(negative$review, max.words = 45, scale = c(3, 0.5))


fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))

dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb)

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

system.time( classifier <- naiveBayes(trainNB, df.train$sentiment, laplace = 1) )

system.time( pred <- predict(classifier, newdata=testNB) )

table("Predictions"=pred, "Actual"=df.test$sentiment)

conf.mat <- confusionMatrix(data=pred, reference=df.test$sentiment)

conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']

#test a specific rows

naiveModel<- naiveBayes(sentiment~., df.train, laplace = 1)
system.time( pred <- predict(naiveModel, newdata = df.train[1:4,]))

table(pred)

cm = table(pred, df.train[1:4,]$sentiment)
conf_mat = confusionMatrix(cm)

conf_mat$byClass
conf_mat$overall
conf_mat$overall['Accuracy']
