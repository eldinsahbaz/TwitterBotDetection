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

#Data Collection
def login():
	CONSUMER_KEY = "4kwvbpYNFKLuJlGOCsz2tdpV8"
	CONSUMER_SECRET = "MBJtx5gO6PdhFkWreCYeS9ivr93eAnlO4uGVikAU0pJRwAYL96"
	OAUTH_TOKEN = "825037296655335425-FFlFTVM8uDgboAK9YfBO5j3cfhHwgT7"
	OAUTH_TOKEN_SECRET = "9LpuESuyI37Y1vrtUPSvnEuZaw1U91MHtbP83GnRy9tM7"
	auth = twitter.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
	return twitter.Twitter(auth=auth)

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


#Data Processing
def extractTweets(tweets):
    print('extracting tweets\n')
    newTweets = dict()
    stop = stopwords.words('english')

    for usr, tweet in tweets.iteritems():
        print("tweets: " + str(usr) + '\n')
        newUsr = dict()
        replyCount = 0
        quoteCount = 0
        tweetDates = list()
        #sources = list()
        text = list()

        for t in tweet:
            if t['in_reply_to_status_id'] is not None:
                replyCount = replyCount + 1
            if t['is_quote_status']:
                quoteCount = quoteCount + 1

            intermediate = re.sub(r'[^\w]', ' ', t['text']).encode('utf8').decode('unicode_escape').encode('ascii','ignore').lower()
            intermediate = ''.join([w for w in intermediate if w not in string.punctuation])
            intermediate = (' '.join([word for word in intermediate.split() if word not in stop]))
            if intermediate != '':
                text.append(intermediate)

            tweetDates.append(parser.parse(t['created_at']).date())
            #if BSHTML(t['source']).a is not None:
            #    sources.append(re.sub(r'[^\w]', ' ', (BSHTML(t['source']).a.contents[0].strip())).encode('utf8').lower())

        newUsr['tweets'] = text
        newUsr['replies'] = replyCount
        newUsr['quotes'] = quoteCount
        newUsr['tweet_freq'] = avgDailyTweets(tweetDates)
        #if len(sources) == 0:
        #    newUsr['num_sources'] = 0
        #else:
        #    newUsr['num_sources'] = len(set(sources))/float(len(sources))
        newTweets[usr] = newUsr

    return newTweets

def extractProfile(profile):
    print('extracting profile\n')

    ks = ['listed_count', 'statuses_count']
    cleaned = dict()
    temp = dict()

    for usr in profile.keys():
        temp = dict()
        for k in ks:
            temp[k] = profile[usr][k]

        temp['follower_ratio'] = profile[usr]['friends_count']/float(profile[usr]['followers_count'])
        cleaned[usr] = temp

    return cleaned

def avgDailyTweets(dates):
    print('averaging daily tweets\n')
    DailyTweetCount = list()
    count = 0
    currentDate = None
    for date in dates:
        if currentDate == None:
            currentDate = date
            count = count + 1
        elif date == currentDate:
            count = count + 1
        else:
            DailyTweetCount.append(count)
            count = 1
            currentDate = date
    if len(DailyTweetCount) != len(set(dates)):
        DailyTweetCount.append(count)

    if len(set(dates)) != 0:
        return sum(DailyTweetCount)/float(len(set(dates)))
    else:
        return 0

def combine(a, b):
    print('combining profile and tweets\n')
    for user in a.keys():
        a[user].update(b[user])

def lexicalDiversity(data):
    print('computing lexical Diversity')
    lexicalDiversities = dict()
    totalUniqueWords = list()
    marked = list()

    for user in data.keys():
        uniqueWords = list()
        for tweet in data[user]['tweets']:
            uniqueWords.extend(tweet.split(" "))

        totalUniqueWords.extend(uniqueWords)
        if len(uniqueWords) > 0:
            lexicalDiversities[user] = {'lexical_diversity': len(set(uniqueWords))/float(len(uniqueWords))}
        else:
            marked.append(user)

    for mark in marked:
        if len(totalUniqueWords) > 0:
            lexicalDiversities[mark] = {'lexical_diversity': len(set(totalUniqueWords))/float(len(totalUniqueWords))}
        else:
            lexicalDiversities[mark] = {'lexical_diversity': 0}

    return lexicalDiversities

def remove(key, data):
    for user in data.keys():
        data[user].pop(key, None)


#Classification
def trainClassifiers():
    f = open('Bots.txt', 'r')
    bots = json.loads((f.readlines())[0])
    f.close()

    f = open('Users.txt', 'r')
    users = json.loads((f.readlines())[0])
    f.close()

    for bot in bots.keys():
        bots[bot] = OrderedDict(sorted(bots[bot].items())).values()

    for user in users.keys():
        users[user] = OrderedDict(sorted(users[user].items())).values()

    bots = bots.values()
    users = users.values()

    random.shuffle(bots)
    random.shuffle(users)

    botSplit = int(len(bots)*0.8)
    userSplit = int(len(users)*0.8)

    trainData = bots[:botSplit]
    trainData.extend(users[:userSplit])
    trainLabels = [0]*botSplit
    trainLabels.extend([1]*userSplit)

    testData = bots[botSplit:]
    testData.extend(users[userSplit:])
    testLabels = [0]*(len(bots)-botSplit)
    testLabels.extend([1]*(len(users)-userSplit))

    #https://www.kaggle.com/c/digit-recognizer/discussion/2299
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(trainData, trainLabels)
    rfOut = rf.predict(testData)

	#http://scikit-learn.org/stable/modules/tree.html
    dt = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
    dt = dt.fit(trainData, trainLabels)
    dtOut = dt.predict(testData)

    #http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    lr = BaggingClassifier(linear_model.LogisticRegression(C=1e5), max_samples=0.5, max_features=0.5)
    lr.fit(trainData, trainLabels)
    lrOut = lr.predict(testData)

    #http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    svm = BaggingClassifier(LinearSVC(), max_samples=0.5, max_features=0.5)
    svm.fit(trainData, trainLabels)
    svmOut = svm.predict(testData)

    #http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    nb = BaggingClassifier(BernoulliNB(), max_samples=0.5, max_features=0.5)
    nb.fit(trainData, trainLabels)
    nbOut = nb.predict(testData)

    #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=4), max_samples=0.5, max_features=0.5)
    knn.fit(trainData, trainLabels)
    knnOut = knn.predict(testData)

    #http://scikit-learn.org/stable/modules/neural_networks_supervised.html
    ann = BaggingClassifier(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 4), random_state=1), max_samples=0.5, max_features=0.5)
    ann.fit(trainData, trainLabels)
    annOut = ann.predict(testData)

    testLabels = np.array(testLabels)
    rates = {'ann': computeRate(annOut, testLabels),'rf': computeRate(rfOut, testLabels), 'dt': computeRate(dtOut, testLabels), 'lr': computeRate(lrOut, testLabels), 'svm': computeRate(svmOut, testLabels), 'nb': computeRate(nbOut, testLabels), 'knn': computeRate(knnOut, testLabels)}
    return (ann, rf, dt, svm, nb, knn, lr, rates)

def computeRate(output, testLabels):
	return ('%.2f%%' % (100*np.count_nonzero(np.array(output) == testLabels)/float(len(testLabels))))

def classify(name):
    twitter_api = login()
    (ann, rf, dt, svm, nb, knn, lr, rates) = trainClassifiers()
    profile = get_user_profile(twitter_api, screen_names=[name])

    if profile[name]['verified']:
        print('verified user')
    else:
        tweets = dict()
        tweets[name] = harvest_user_timeline(twitter_api, screen_name=name,  max_results=maxint)

        tweets = extractTweets(tweets)
        combine(tweets, lexicalDiversity(tweets))
        remove('tweets', tweets)
        profile = extractProfile(profile)
        combine(profile, tweets)
        user = [OrderedDict(sorted(profile[name].items())).values()]

        result = {'ann': ann.predict(user), 'rf': rf.predict(user), 'dt': dt.predict(user), 'lr': lr.predict(user), 'svm': svm.predict(user), 'nb': nb.predict(user), 'knn': knn.predict(user)}
        return (rates, result)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        (rates, result) = classify(name)
        print('Classification Accuracy:', rates)
        print('Classification Results for {0}:'.format(name), result)
