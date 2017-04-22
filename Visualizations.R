library('ggplot2')

bots <- read.csv("Bots.csv")
rownames(bots) <- bots$X
bots$X = NULL
bots <- as.data.frame(t(bots[,-1]))
bots$Label = factor('bot')

users <- read.csv("Users.csv")
rownames(users) <- users$X
users$X = NULL
users <- as.data.frame(t(users[,-1]))
users$Label = factor('user')

TwitterData <- rbind(bots, users)
rm(bots, users)

TwitterData[, -8] = scale(TwitterData[,-8])

#lexical diversity
#lex_normalized <- (TwitterData$lexical_diversity-min(TwitterData$lexical_diversity))/(max(TwitterData$lexical_diversity)-min(TwitterData$lexical_diversity))
lex_bp <- ggplot(TwitterData, aes(Label, lexical_diversity)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
lex_Scatter <- qplot(seq_along(TwitterData$lexical_diversity), TwitterData$lexical_diversity)
lex_StackedBar <- qplot(x=TwitterData$lexical_diversity, fill=TwitterData$Label, geom="histogram") 
lex_heatBar <- qplot(x=TwitterData$lexical_diversity, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#tweet frequency
#tf_normalized <- (TwitterData$tweet_freq-min(TwitterData$tweet_freq))/(max(TwitterData$tweet_freq)-min(TwitterData$tweet_freq))
tf_bp <- ggplot(TwitterData, aes(Label, tweet_freq)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
tf_Scatter <- qplot(seq_along(TwitterData$tweet_freq), TwitterData$tweet_freq)
tf_StackedBar <- qplot(x=TwitterData$tweet_freq, fill=TwitterData$Label, geom="histogram") 
tf_heatBar <- qplot(x=TwitterData$tweet_freq, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#statuses count
#sc_normalized <- (TwitterData$statuses_count-min(TwitterData$statuses_count))/(max(TwitterData$statuses_count)-min(TwitterData$statuses_count))
sc_bp <- ggplot(TwitterData, aes(Label, statuses_count)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
sc_Scatter <- qplot(seq_along(TwitterData$statuses_count), TwitterData$statuses_count)
sc_StackedBar <- qplot(x=TwitterData$statuses_count, fill=TwitterData$Label, geom="histogram") 
sc_heatBar <- qplot(x=TwitterData$statuses_count, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#follower ratio
#fr_normalized <- (TwitterData$follower_ratio-min(TwitterData$follower_ratio))/(max(TwitterData$follower_ratio)-min(TwitterData$follower_ratio))
fr_bp <- ggplot(TwitterData, aes(Label, follower_ratio)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
fr_Scatter <- qplot(seq_along(TwitterData$follower_ratio), TwitterData$follower_ratio)
fr_StackedBar <- qplot(x=TwitterData$follower_ratio, fill=TwitterData$Label, geom="histogram") 
fr_heatBar <- qplot(x=TwitterData$follower_ratio, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#listed count
#lc_normalized <- (TwitterData$listed_count-min(TwitterData$listed_count))/(max(TwitterData$listed_count)-min(TwitterData$listed_count))
lc_bp <- ggplot(TwitterData, aes(Label, listed_count)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
lc_Scatter <- qplot(seq_along(TwitterData$listed_count), TwitterData$listed_count)
lc_StackedBar <- qplot(x=TwitterData$listed_count, fill=TwitterData$Label, geom="histogram") 
lc_heatBar <- qplot(x=TwitterData$listed_count, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#quotes
#q_normalized <- (TwitterData$quotes-min(TwitterData$quotes))/(max(TwitterData$quotes)-min(TwitterData$quotes))
q_bp <- ggplot(TwitterData, aes(Label, quotes)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
q_Scatter <- qplot(seq_along(TwitterData$quotes), TwitterData$quotes)
q_StackedBar <- qplot(x=TwitterData$quotes, fill=TwitterData$Label, geom="histogram") 
q_heatBar <- qplot(x=TwitterData$quotes, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#replies
#r_normalized <- (TwitterData$replies-min(TwitterData$replies))/(max(TwitterData$replies)-min(TwitterData$replies))
r_bp <- ggplot(TwitterData, aes(Label, replies)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
r_Scatter <- qplot(seq_along(TwitterData$replies), TwitterData$replies)
r_StackedBar <- qplot(x=TwitterData$replies, fill=TwitterData$Label, geom="histogram") 
r_heatBar <- qplot(x=TwitterData$replies, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

lex_bp
lex_Scatter
lex_StackedBar
lex_heatBar

tf_bp
tf_Scatter
tf_StackedBar
tf_heatBar

fr_bp
fr_Scatter
fr_StackedBar
fr_heatBar

sc_bp
sc_Scatter
sc_StackedBar
sc_heatBar

lc_bp
lc_Scatter
lc_StackedBar
lc_heatBar

q_bp
q_Scatter
q_StackedBar
q_heatBar

r_bp
r_Scatter
r_StackedBar
r_heatBar
