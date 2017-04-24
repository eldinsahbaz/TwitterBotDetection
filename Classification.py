import numpy as np
import json, random, twitter, datetime, re, sys, string
from sys import maxint
from dateutil import parser
from bs4 import BeautifulSoup as BSHTML
from urllib2 import URLError
from httplib import BadStatusLine
from functools import partial
from collections import OrderedDict
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold

'''
Data Collection...The code from this section comes from Python Twitter cookbook
'''
def login():
	CONSUMER_KEY = "4kwvbpYNFKLuJlGOCsz2tdpV8"
	CONSUMER_SECRET = "MBJtx5gO6PdhFkWreCYeS9ivr93eAnlO4uGVikAU0pJRwAYL96"
	OAUTH_TOKEN = "825037296655335425-FFlFTVM8uDgboAK9YfBO5j3cfhHwgT7"
	OAUTH_TOKEN_SECRET = "9LpuESuyI37Y1vrtUPSvnEuZaw1U91MHtbP83GnRy9tM7"
	auth = twitter.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
	return twitter.Twitter(auth=auth)

#make a request to twitter
def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):

    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600: # Seconds
            print >> sys.stderr, 'Too many retries. Quitting.'
            raise e

        # See https://dev.twitter.com/docs/error-codes-responses for common codes

        if e.e.code == 401:
            print >> sys.stderr, 'Encountered 401 Error (Not Authorized)'
            return None
        elif e.e.code == 404:
            print >> sys.stderr, 'Encountered 404 Error (Not Found)'
            return None
        elif e.e.code == 429:
            print >> sys.stderr, 'Encountered 429 Error (Rate Limit Exceeded)'
            if sleep_when_rate_limited:
                print >> sys.stderr, "Retrying in 15 minutes...ZzZ..."
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print >> sys.stderr, '...ZzZ...Awake now and trying again.'
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print >> sys.stderr, 'Encountered %i Error. Retrying in %i seconds' %                 (e.e.code, wait_period)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError, e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError, e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print >> sys.stderr, "URLError encountered. Continuing."
            if error_count > max_errors:
                print >> sys.stderr, "Too many consecutive errors...bailing out."
                raise
        except BadStatusLine, e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print >> sys.stderr, "BadStatusLine encountered. Continuing."
            if error_count > max_errors:
                print >> sys.stderr, "Too many consecutive errors...bailing out."
                raise

#get a desired user's profile
def get_user_profile(twitter_api, screen_names=None, user_ids=None):

    # Must have either screen_name or user_id (logical xor)
    assert (screen_names != None) != (user_ids != None),     "Must have screen_names or user_ids, but not both"

    items_to_info = {}

    items = screen_names or user_ids

    while len(items) > 0:

        # Process 100 items at a time per the API specifications for /users/lookup.
        # See https://dev.twitter.com/docs/api/1.1/get/users/lookup for details.

        items_str = ','.join([str(item) for item in items[:100]])
        items = items[100:]

        if screen_names:
            response = make_twitter_request(twitter_api.users.lookup,
                                            screen_name=items_str)
        else: # user_ids
            response = make_twitter_request(twitter_api.users.lookup,
                                            user_id=items_str)

        for user_info in response:
            if screen_names:
                items_to_info[user_info['screen_name']] = user_info
            else: # user_ids
                items_to_info[user_info['id']] = user_info

    return items_to_info

#get a desiered user's timeline
def harvest_user_timeline(twitter_api, screen_name=None, user_id=None, max_results=1000):

    assert (screen_name != None) != (user_id != None),     "Must have screen_name or user_id, but not both"

    kw = {  # Keyword args for the Twitter API call
        'count': 200,
        'trim_user': 'true',
        'include_rts' : 'true',
        'since_id' : 1
        }

    if screen_name:
        kw['screen_name'] = screen_name
    else:
        kw['user_id'] = user_id

    max_pages = 16
    results = []

    tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)

    if tweets is None: # 401 (Not Authorized) - Need to bail out on loop entry
        tweets = []

    results += tweets

    print >> sys.stderr, 'Fetched %i tweets' % len(tweets)

    page_num = 1

    # Many Twitter accounts have fewer than 200 tweets so you don't want to enter
    # the loop and waste a precious request if max_results = 200.

    # Note: Analogous optimizations could be applied inside the loop to try and
    # save requests. e.g. Don't make a third request if you have 287 tweets out of
    # a possible 400 tweets after your second request. Twitter does do some
    # post-filtering on censored and deleted tweets out of batches of 'count', though,
    # so you can't strictly check for the number of results being 200. You might get
    # back 198, for example, and still have many more tweets to go. If you have the
    # total number of tweets for an account (by GET /users/lookup/), then you could
    # simply use this value as a guide.

    if max_results == kw['count']:
        page_num = max_pages # Prevent loop entry

    while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:

        # Necessary for traversing the timeline in Twitter's v1.1 API:
        # get the next query's max-id parameter to pass in.
        # See https://dev.twitter.com/docs/working-with-timelines.
        kw['max_id'] = min([ tweet['id'] for tweet in tweets]) - 1

        tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)
        results += tweets

        print >> sys.stderr, 'Fetched %i tweets' % (len(tweets),)

        page_num += 1

    print >> sys.stderr, 'Done fetching tweets'

    return results[:max_results]

'''
Data Processing
'''
#function for filtering raw user timline data from Twitter
def extractTweets(tweets):
	#holds the reducded feature set
    newTweets = dict()

	#list of stopwords from NLTK
    stop = stopwords.words('english')

	#iterate over each user's timeline
    for usr, tweet in tweets.iteritems():

		#temporary variables for computation
        newUsr = dict()
        replyCount = 0
        quoteCount = 0
        tweetDates = list()
        #sources = list()
        text = list()

		#iterate over each tweet for this user
        for t in tweet:
			#compute the reply count and the quote count
            if t['in_reply_to_status_id'] is not None:
                replyCount = replyCount + 1
            if t['is_quote_status']:
                quoteCount = quoteCount + 1

			#filter the tweets of all symbols and unicode text (i.e. emojis)
            intermediate = re.sub(r'[^\w]', ' ', t['text']).encode('utf8').decode('unicode_escape').encode('ascii','ignore').lower()
            intermediate = ''.join([w for w in intermediate if w not in string.punctuation])
            intermediate = (' '.join([word for word in intermediate.split() if word not in stop]))

			#if the string isn't empty, the add it to text
            if intermediate != '':
                text.append(intermediate)

			#parse the date at which the tweet was created
            tweetDates.append(parser.parse(t['created_at']).date())

			#We planned to incorporate source as a feature but did not get to it in time.
			#Leaving the code commented out now so that we can just uncomment it in the future
            #if BSHTML(t['source']).a is not None:
            #    sources.append(re.sub(r'[^\w]', ' ', (BSHTML(t['source']).a.contents[0].strip())).encode('utf8').lower())

		#put the information in a temporary dictionary
        newUsr['tweets'] = text
        newUsr['replies'] = replyCount
        newUsr['quotes'] = quoteCount
        newUsr['tweet_freq'] = avgDailyTweets(tweetDates)

		#this was for adding sources to the dictionary
		#Leaving the code commented out now so that we can just uncomment it in the future
        #if len(sources) == 0:
        #    newUsr['num_sources'] = 0
        #else:
        #    newUsr['num_sources'] = len(set(sources))/float(len(sources))

		#add the temporary dictionary to the new dictionary being returned
        newTweets[usr] = newUsr

    return newTweets

#function for filtering raw user profile data from Twitter
def extractProfile(profile):
	#a list of the features that we are keeping
    ks = ['listed_count', 'statuses_count']
    cleaned = dict()
    temp = dict()

	#iterate over each user
    for usr in profile.keys():
        temp = dict()

		#add each of the listed features to a temporary dictionary
        for k in ks:
            temp[k] = profile[usr][k]

		#derive friend-follower ration
        temp['follower_ratio'] = profile[usr]['friends_count']/float(profile[usr]['followers_count'])

		#add the temporary dictionary with the extracted features to the 'cleaned' dictionary
        cleaned[usr] = temp

    return cleaned

#compute the average number of daily tweets for each user
def avgDailyTweets(dates):
	#list of average daily tweet rates
    DailyTweetCount = list()
    count = 0
    currentDate = None

	#iterate over each tweet date
    for date in dates:
		#if the date hasn't changed then increate the
		#tweet frequency. Otherwise, if the date is different
		#store the tweet frequency and reset the frequency counter
        if currentDate == None:
            currentDate = date
            count = count + 1
        elif date == currentDate:
            count = count + 1
        else:
            DailyTweetCount.append(count)
            count = 1
            currentDate = date

	#add the last tweet frequency if it hasn't been added yet
    if len(DailyTweetCount) != len(set(dates)):
        DailyTweetCount.append(count)

	#compute the average or return 0 (avoiding divide by 0 error)
    if len(set(dates)) != 0:
        return sum(DailyTweetCount)/float(len(set(dates)))
    else:
        return 0

#merge keys from dictionary b into a
def combine(a, b):
    if (sorted(a.keys()) == sorted(b.keys())):
        for user in a.keys():
			#add keys from b to a
            a[user].update(b[user])

#computes lexical diversity for a set of tweets
def lexicalDiversity(data):
    lexicalDiversities = dict()
    totalUniqueWords = list()
    marked = list()

	#iterate over each user in the user timeline
    for user in data.keys():
        uniqueWords = list()
		#for each tweet, add each word (space delimited) to the uniqueWords list
		#this will be converted to a set below to get the true unique word count
        for tweet in data[user]['tweets']:
            uniqueWords.extend(tweet.split(" "))

		#add each word from these tweets to a the list of all words accross all users
        totalUniqueWords.extend(uniqueWords)
        if len(uniqueWords) > 0:
			#compute number of unique words over number of toal words for this user
            lexicalDiversities[user] = {'lexical_diversity': len(set(uniqueWords))/float(len(uniqueWords))}
        else:
			#if the user doesnt have any tweets, we'll mark them for now and
			#replace their lexical diversity with the average lexical diversity to
			#avoid outliers in our dataset (i.e. estimate the missing value by averaging)
            marked.append(user)

	#for each marked user, replace their lexical diversity with the average lexical diversity
	#if the number of total unique words is 0 (i.e. no body tweeted anything), then manually set
	#the lexical diversity to 0
    for mark in marked:
        if len(totalUniqueWords) > 0:
            lexicalDiversities[mark] = {'lexical_diversity': len(set(totalUniqueWords))/float(len(totalUniqueWords))}
        else:
            lexicalDiversities[mark] = {'lexical_diversity': 0}

    return lexicalDiversities

#remove the desired key from the data
def remove(key, data):
	#pops the key from the dictionary for each user
    for user in data.keys():
        if key in data[user]:
            data[user].pop(key, None)

'''
Classification
'''
#train the classifiers on our dataset
def trainClassifiers():
	#read in bot data from our data set
    f = open('Bots.txt', 'r')
    bots = json.loads((f.readlines())[0])
    f.close()

	#read in use data from our dataset
    f = open('Users.txt', 'r')
    users = json.loads((f.readlines())[0])
    f.close()

	#make sure that all the keys for each user is ordered the same way
	#then convert the dictionary to a list of values. This is for classification.
	#having data in the wrong index can throw-off the classifiers during training
    for bot in bots.keys():
        bots[bot] = OrderedDict(sorted(bots[bot].items())).values()

    for user in users.keys():
        users[user] = OrderedDict(sorted(users[user].items())).values()

	#convert the dictionaries of bots and users to lists
    bots = bots.values()
    users = users.values()

	#shuffle the data
    random.shuffle(bots)
    random.shuffle(users)

	#conver the bot and user lists to numpy arrays
    bots = np.array(bots)
    users = np.array(users)

	#split the data into 10 folds (for cross validation)
    folds = KFold(n_splits=10, random_state=None, shuffle=False)
    folds.get_n_splits(bots)
    folds.get_n_splits(users)

	#place holders for accuracy values for each of the classifiers we use
	#(to be averaged after iterating over each of the 10 folds)
    accuracies = {'NeuralNetwork': [], 'LogisticRegression': [], 'KNearestNeighbors': [], 'SupportVectorMachine': [], 'RandomForest': [], 'DecisionTree': [], 'NaiveBayes': []}

	#iterate over each fold
    f = 1
    for train_index, test_index in folds.split(bots):
        print('fold {0}'.format(f))
        f = f + 1

		#segment the data into training and testing sets
        trainBots, testBots = bots[train_index], bots[test_index]
        trainUsers, testUsers = users[train_index], users[test_index]

		#merge the training bot and user data into a training sets
        trainData = list()
        trainData.extend(trainBots)
        trainData.extend(trainUsers)

		#generate training labels
        trainLabels = [0]*len(trainBots)
        trainLabels.extend([1]*len(trainUsers))

		#merge the testing bot and user data into a testing set
        testData = list()
        testData.extend(testBots)
        testData.extend(testUsers)

		#generate testing labels
        testLabels = [0]*len(testBots)
        testLabels.extend([1]*len(testUsers))

		#train and classify with a random forest
	    #https://www.kaggle.com/c/digit-recognizer/discussion/2299
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(trainData, trainLabels)
        rfOut = rf.predict(testData)

		#train and classify with a decision tree using bagging
		#http://scikit-learn.org/stable/modules/tree.html
        dt = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
        dt = dt.fit(trainData, trainLabels)
        dtOut = dt.predict(testData)

		#train and classify with a logisitic regressor using bagging
	    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        lr = BaggingClassifier(linear_model.LogisticRegression(C=1e5), max_samples=0.5, max_features=0.5)
        lr.fit(trainData, trainLabels)
        lrOut = lr.predict(testData)

		#train and classify with a support vector machine with a linear kernel using bagging
	    #http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        svm = BaggingClassifier(LinearSVC(), max_samples=0.5, max_features=0.5)
        svm.fit(trainData, trainLabels)
        svmOut = svm.predict(testData)

		#train and classify with a Naive Bayes for multivariate Bernoulli models using bagging
	    #http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        nb = BaggingClassifier(BernoulliNB(), max_samples=0.5, max_features=0.5)
        nb.fit(trainData, trainLabels)
        nbOut = nb.predict(testData)

		#train and classify with a k nearest keighbors classifier with k=4 using bagging
	    #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=4), max_samples=0.5, max_features=0.5)
        knn.fit(trainData, trainLabels)
        knnOut = knn.predict(testData)

		#train and classify with a multi layer perceptron using bagging
	    #http://scikit-learn.org/stable/modules/neural_networks_supervised.html
        ann = BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1), max_samples=0.5, max_features=0.5)
        ann.fit(trainData, trainLabels)
        annOut = ann.predict(testData)

		#conver the testLabels to a numpy array and compute the accuracy for this
		#iteration. Then add the accuracy to the list of accuracies for each classifier
        testLabels = np.array(testLabels)
        accuracies['NeuralNetwork'].append(accuracy(annOut, testLabels))
        accuracies['LogisticRegression'].append(accuracy(lrOut, testLabels))
        accuracies['KNearestNeighbors'].append(accuracy(knnOut, testLabels))
        accuracies['SupportVectorMachine'].append(accuracy(svmOut, testLabels))
        accuracies['RandomForest'].append(accuracy(rfOut, testLabels))
        accuracies['DecisionTree'].append(accuracy(dtOut, testLabels))
        accuracies['NaiveBayes'].append(accuracy(nbOut, testLabels))

    return (ann, rf, dt, svm, nb, knn, lr, accuracies)

#compute the accuracy of the output from the classifiers
def accuracy(output, testLabels):
	#find the instances that match in the output and the test lables and sum them up
	#then divide by the toal number of instances and multiply by 100 (to get percentage)
	return (100*np.count_nonzero(np.array(output) == testLabels)/float(len(testLabels)))

#classify a new instance
def classify(name, twitter_api, ann, rf, dt, svm, nb, knn, lr):
	#get the user's profile from Twitter
    profile = get_user_profile(twitter_api, screen_names=[name])

	#only classify if the user not a verified user
    if profile[name]['verified']:
        print('verified user')
        return({})
    else:
		#get Tweets for this user (i.e. get user timeline)
	    tweets = dict()
	    tweets[name] = harvest_user_timeline(twitter_api, screen_name=name,  max_results=maxint)

		#extract the features from user timeline
	    tweets = extractTweets(tweets)
		#combine user timeline with lexical diversity
	    combine(tweets, lexicalDiversity(tweets))
		#remove the text (actual tweet data) from user timeline data
	    remove('tweets', tweets)
		#extract features from user profile
	    profile = extractProfile(profile)
		#combine profile and timeline
	    combine(profile, tweets)
		#order the keys and convert to list for classification
	    user = [OrderedDict(sorted(profile[name].items())).values()]

		#return rsults of classification
	    return({'ann': ann.predict(user).tolist(), 'rf': rf.predict(user).tolist(), 'dt': dt.predict(user).tolist(), 'lr': lr.predict(user).tolist(), 'svm': svm.predict(user).tolist(), 'nb': nb.predict(user).tolist(), 'knn': knn.predict(user).tolist()})


#takes in usernames for classification via command line arguments
if __name__ == "__main__":
	#get authenticated to make twitter calls
	twitter_api = login()
	#train and get classifiers
	(ann, rf, dt, svm, nb, knn, lr, rates) = trainClassifiers()

	#average the accuracy rates (for each classifier) and convert the accuracy rates to percent
	for i in rates.keys():
		rates[i] = '%.2f%%' % (sum(rates[i])/float(len(rates[i])))
	print('Classification Accuracy:', rates)
	f = open('KFoldCV.txt', 'w')
	json.dump(rates, f)
	f.close()

	#for each screen_name passed in via command line argument, classify as either
	#bot (i.e. 0) or user (i.e. 1)
	for name in sys.argv[1:]:
		print('screen_name:' + name)
		result = classify(name, twitter_api, ann, rf, dt, svm, nb, knn, lr)
		print('Classification Results for {0}:'.format(name), result)
        f = open('result.txt', 'w')
        json.dump(result, f)
        f.close()
