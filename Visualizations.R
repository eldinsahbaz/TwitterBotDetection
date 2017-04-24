library('ggplot2')
library(arules)
library(scales)

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

usersCopy  = users
botsCopy = bots

TwitterData <- rbind(bots, users)
rm(bots, users)

TwitterData[, -8] = scale(TwitterData[,-8])

#lexical diversity
#lex_normalized <- (TwitterData$lexical_diversity-min(TwitterData$lexical_diversity))/(max(TwitterData$lexical_diversity)-min(TwitterData$lexical_diversity))
lex_bp <- ggplot(TwitterData, aes(Label, lexical_diversity)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
lex_Scatter <- qplot(seq_along(TwitterData$lexical_diversity), TwitterData$lexical_diversity, color= TwitterData$Label)
lex_StackedBar <- qplot(x=TwitterData$lexical_diversity, fill=TwitterData$Label, geom="histogram") 
lex_heatBar <- qplot(x=TwitterData$lexical_diversity, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#tweet frequency
#tf_normalized <- (TwitterData$tweet_freq-min(TwitterData$tweet_freq))/(max(TwitterData$tweet_freq)-min(TwitterData$tweet_freq))
tf_bp <- ggplot(TwitterData, aes(Label, tweet_freq)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
tf_Scatter <- qplot(seq_along(TwitterData$tweet_freq), TwitterData$tweet_freq, color= TwitterData$Label)
tf_StackedBar <- qplot(x=TwitterData$tweet_freq, fill=TwitterData$Label, geom="histogram") 
tf_heatBar <- qplot(x=TwitterData$tweet_freq, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#statuses count
#sc_normalized <- (TwitterData$statuses_count-min(TwitterData$statuses_count))/(max(TwitterData$statuses_count)-min(TwitterData$statuses_count))
sc_bp <- ggplot(TwitterData, aes(Label, statuses_count)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
sc_Scatter <- qplot(seq_along(TwitterData$statuses_count), TwitterData$statuses_count, color= TwitterData$Label)
sc_StackedBar <- qplot(x=TwitterData$statuses_count, fill=TwitterData$Label, geom="histogram") 
sc_heatBar <- qplot(x=TwitterData$statuses_count, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#follower ratio
#fr_normalized <- (TwitterData$follower_ratio-min(TwitterData$follower_ratio))/(max(TwitterData$follower_ratio)-min(TwitterData$follower_ratio))
fr_bp <- ggplot(TwitterData, aes(Label, follower_ratio)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
fr_Scatter <- qplot(seq_along(TwitterData$follower_ratio), TwitterData$follower_ratio, color= TwitterData$Label)
fr_StackedBar <- qplot(x=TwitterData$follower_ratio, fill=TwitterData$Label, geom="histogram") 
fr_heatBar <- qplot(x=TwitterData$follower_ratio, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#listed count
#lc_normalized <- (TwitterData$listed_count-min(TwitterData$listed_count))/(max(TwitterData$listed_count)-min(TwitterData$listed_count))
lc_bp <- ggplot(TwitterData, aes(Label, listed_count)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
lc_Scatter <- qplot(seq_along(TwitterData$listed_count), TwitterData$listed_count, color= TwitterData$Label)
lc_StackedBar <- qplot(x=TwitterData$listed_count, fill=TwitterData$Label, geom="histogram") 
lc_heatBar <- qplot(x=TwitterData$listed_count, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#quotes
#q_normalized <- (TwitterData$quotes-min(TwitterData$quotes))/(max(TwitterData$quotes)-min(TwitterData$quotes))
q_bp <- ggplot(TwitterData, aes(Label, quotes)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
q_Scatter <- qplot(seq_along(TwitterData$quotes), TwitterData$quotes, color= TwitterData$Label)
q_StackedBar <- qplot(x=TwitterData$quotes, fill=TwitterData$Label, geom="histogram") 
q_heatBar <- qplot(x=TwitterData$quotes, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

#replies
#r_normalized <- (TwitterData$replies-min(TwitterData$replies))/(max(TwitterData$replies)-min(TwitterData$replies))
r_bp <- ggplot(TwitterData, aes(Label, replies)) + geom_boxplot(outlier.colour = "red", outlier.shape = 1)
r_Scatter <- qplot(seq_along(TwitterData$replies), TwitterData$replies, color= TwitterData$Label)
r_StackedBar <- qplot(x=TwitterData$replies, fill=TwitterData$Label, geom="histogram") 
r_heatBar <- qplot(x=TwitterData$replies, fill=..count.., geom="histogram") + scale_fill_gradient(low="blue", high="red")

###############################################################################################################
#Discretizing the Data for each individual dataset of Bots and Users

#Lexical Diversity For User
lex_cut_user = discretize(usersCopy$lexical_diversity,method = "interval", categories = 5, onlycuts = TRUE)
user_discretized = cut(usersCopy$lexical_diversity, lex_cut_user, labels = c("Low","Low-Mod","Moderate","Mod-High","High") )
usersCopy$lexical_diversity <- user_discretized

Lexical_Diversity_Users <- as.vector(table(usersCopy$lexical_diversity))
lbls <- c("Low","Low-Mod","Moderate","Mod-High","High")
pct <- round(Lexical_Diversity_Users/sum(Lexical_Diversity_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(Lexical_Diversity_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Lexical Diversity of Users")

#Lexical Diversity For Bot
lex_cut_bot = discretize(botsCopy$lexical_diversity,method = "interval", categories = 5, onlycuts = TRUE)
bot_discretized = cut(botsCopy$lexical_diversity, lex_cut_bot, labels = c("Low","Low-Mod","Moderate","Mod-High","High") )
botsCopy$lexical_diversity <- bot_discretized

Lexical_Diversity_Bots <- as.vector(table(botsCopy$lexical_diversity))
lbls <- c("Low","Low-Mod","Moderate","Mod-High","High")
pct <- round(Lexical_Diversity_Bots/sum(Lexical_Diversity_Bots)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(Lexical_Diversity_Bots,labels = lbls, col=rainbow(length(lbls)),
    main="Lexical Diversity of Bots") 

#Tweet Freq For User
tf_cut = discretize(usersCopy$tweet_freq,method = "cluster", categories = 2, onlycuts = TRUE)
tf_discretized = cut(usersCopy$tweet_freq, tf_cut, labels = c("Low","High") )
usersCopy$tweet_freq <- tf_discretized

tf_Users <- as.vector(table(usersCopy$tweet_freq))
lbls <- c("Low","High")
pct <- round(tf_Users/sum(tf_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(tf_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Tweet Freq of Users")

#Tweet Freq for Bot
tf_cut_bot = discretize(botsCopy$tweet_freq,method = "cluster", categories = 3, onlycuts = TRUE)
tf_discretized_bot = cut(botsCopy$tweet_freq, tf_cut_bot, labels = c("Low","Moderate","High") )
botsCopy$tweet_freq <- tf_discretized_bot

tf_bot <- as.vector(table(botsCopy$tweet_freq))
lbls <- c("Low","Moderate","High")
pct <- round(tf_bot/sum(tf_bot)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(tf_bot,labels = lbls, col=rainbow(length(lbls)),
    main="Tweet Freq of bot")

#Listed Count For Users
ls_cut_user = discretize(usersCopy$listed_count,method = "cluster", categories = 4, onlycuts = TRUE)
ls_discretized = cut(usersCopy$listed_count, ls_cut_user, labels = c("Low","Low-Mod","Moderate","High") )
usersCopy$listed_count <- ls_discretized

ls_Users <- as.vector(table(usersCopy$listed_count))
lbls <- c("Low","Low-Mod","Moderate","High")
pct <- round(ls_Users/sum(ls_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(ls_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Listed Count of Users")


#Listed Count For Bots
ls_cut_bot = discretize(botsCopy$listed_count,method = "cluster", categories = 3, onlycuts = TRUE)
ls_discretized_bot = cut(botsCopy$listed_count, ls_cut_bot, labels = c("Low","Moderate","High") )
botsCopy$listed_count <- ls_discretized_bot

ls_bot <- as.vector(table(botsCopy$listed_count))
lbls <- c("Low","Moderate","High")
pct <- round(ls_bot/sum(ls_bot)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(ls_bot,labels = lbls, col=rainbow(length(lbls)),
    main="Listed Count of bots")

#quotes for Users
qs_cut_user = discretize(usersCopy$quotes,method = "cluster", categories = 4, onlycuts = TRUE)
qs_discretized = cut(usersCopy$quotes, qs_cut_user, labels = c("Low","Low-Mod","Moderate","High") )
usersCopy$quotes <- qs_discretized

qs_Users <- as.vector(table(usersCopy$quotes))
lbls <- c("Low","Low-Mod","Moderate","High")
pct <- round(ls_Users/sum(ls_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(qs_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Users that Quote")

#quotes for Bots

qs_cut_bot = discretize(botsCopy$quotes,method = "cluster", categories = 2, onlycuts = TRUE)
qs_discretized_bot = cut(botsCopy$quotes, qs_cut_bot, labels = c("Low","High") )
botsCopy$quotes <- qs_discretized_bot

qs_bot <- as.vector(table(botsCopy$quotes))
lbls <- c("Low","High")
pct <- round(qs_bot/sum(qs_bot)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(qs_bot,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Bots that Quote")


#Replies For Users
r_cut_user = discretize(usersCopy$replies,method = "interval", categories = 3, onlycuts = TRUE)
r_discretized = cut(usersCopy$replies, r_cut_user, labels = c("Low","Moderate","High") )
usersCopy$replies <- r_discretized

r_Users <- as.vector(table(usersCopy$replies))
lbls <- c("Low","Moderate","High")
pct <- round(ls_Users/sum(ls_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(r_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Users that Reply")



#Replies For bots
r_cut_bot = discretize(botsCopy$replies,method = "interval", categories = 3, onlycuts = TRUE)
r_discretized_bot = cut(botsCopy$replies,  r_cut_bot, labels = c("Low","Moderate","High") )
botsCopy$replies <- r_discretized_bot

r_bot <- as.vector(table(botsCopy$replies))
lbls <- c("Low","Moderate","High")
pct <- round(r_bot/sum(r_bot)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(r_bot,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Bots That Reply ")


#Status Count For Users
sc_cut_user = discretize(usersCopy$statuses_count,method = "cluster", categories = 3, onlycuts = TRUE)
sc_discretized = cut(usersCopy$statuses_count, sc_cut_user, labels = c("Low","Moderate","High") )
usersCopy$statuses_count <- sc_discretized

sc_Users <- as.vector(table(usersCopy$statuses_count))
lbls <- c("Low","Moderate","High")
pct <- round(sc_Users/sum(sc_Users)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(sc_Users,labels = lbls, col=rainbow(length(lbls)),
    main="Pie Chart of Users's Status Counts")




#Status Count For bots
sc_cut_bot = discretize(botsCopy$statuses_count,method = "fixed", categories = c(0,1000,10000,Inf), onlycuts = TRUE)
sc_discretized_bot = cut(botsCopy$statuses_count, sc_cut_bot, labels =  c("Low","Moderate","High"))
botsCopy$statuses_count <- sc_discretized_bot

sc_Bots <- as.vector(table(botsCopy$statuses_count))
lbls <- c("Low","Moderate","High")
pct <- round(sc_Bots/sum(sc_Bots)*100)
lbls <- paste(lbls, pct)
lbls <- paste(lbls,"%",sep="") 
pie(sc_Bots,labels = lbls, col=rainbow(length(lbls)),
    main="Status Count of Bots") 

