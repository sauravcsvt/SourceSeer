import cPickle as pickle
import os
import numpy
from numpy import *
from datetime import datetime, date, timedelta
import tarfile
from embers import geocode, utils
import glob
import re
from urllib2 import urlparse, urlopen
import json
from dateutil.parser import parse
import webarticle2text
import nltk
import time

locationcls = geocode.Geo()
wordlemmatizer = nltk.WordNetLemmatizer()


def contains_digits(word):

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    count_digit = 0
    for element in word:
        if element in digits:
            count_digit += 1
    if count_digit == len(word):
        return True
    else:
        return False


def tensor(json_data, id_list, timestamp, stopwords_list, starttimeindex):

    country_list = ["Argentina", "Brazil", "Chile", "Paraguay", "Ecuador", "Colombia", "El Salvador", "Mexico", "Uruguay", "Venezuela"]
    metric_dictionary = {}
    for q in xrange(len(json_data)):

        idcurrentlist = []
        if "embersId" in json_data[q]:
            idcurrentlist.append(json_data[q]["embersId"])
        else:
            idcurrentlist.append(json_data[q]["embers_id"])
        if idcurrentlist[0] in id_list:
            id_list.remove(idcurrentlist[0])
        else:
            continue

        if json_data[q][u'location'][u'country'] not in country_list:
            continue
        date_obj = parse(json_data[q]["date"])
        latitude_num = float(json_data[q][u'location'][u'lat'])
        longitude_num = float(json_data[q][u'location'][u'lng'])
        d1 = date(date_obj.year, date_obj.month, date_obj.day)
        if d1 < timestamp[0] or d1 >= timestamp[len(timestamp) - 1]:
            continue
        try:
            finalURL = (urlopen(json_data[q]['link'])).geturl()
            u = urlparse.urlparse(finalURL)
            article_source = u.netloc
            articleprovince = list(locationcls.lookup_city(latitude_num, longitude_num, 360.)[0])[2]
            articlecountry = list(locationcls.lookup_city(latitude_num, longitude_num, 360.)[0])[1]
        except Exception:
            continue
        try:
            sentence_1 = webarticle2text.extractFromURL(finalURL)
        except Exception:
            sentence_1 = ""
        sentence_2 = json_data[q]['descr']
        tokens = nltk.word_tokenize(sentence_2)
        article_dictionary = {}
        token_normalized = []
        token_minimized = []
        token_filtered = []
        ultimate_token = []
        token_minimized_list = []
        for a in xrange(len(timestamp) - 1):
            if timestamp[a] <= d1 < timestamp[a + 1]:
                timestampindex = starttimeindex + a
                break
        if sentence_1 != "":
            tokens_1 = nltk.word_tokenize(sentence_1)
            for word in tokens_1:
                tokens.append(word)
        for word in tokens:
            word_split = re.split('(\W+)', word)
            if len(word_split) == 1:
                if len(word_split[0]) > 2 and len(word_split[0]) < 15:
                    token_filtered.append(word)
                else:
                    continue
            elif (len(word_split) == 3 and word_split[2] == '' and len(word_split[0]) > 2 and len(word_split[0]) < 15):
                token_filtered.append(word_split[0])
        for word in token_filtered:
            if not contains_digits(word):
                token_minimized.append(word)
        for word in token_minimized:
            try:
                token_normalized.append(utils.normalize_str(word))
            except Exception:
                continue
        for word in token_normalized:
            if stopwords_list.count(word) == 0:
                ultimate_token.append(word)
        token_minimized_list = list(set(ultimate_token))
        for word in token_minimized_list:
            article_dictionary[(word, (articleprovince, articlecountry), article_source, timestampindex)] = ultimate_token.count(word)
        for countkey in article_dictionary.keys():
            if metric_dictionary.has_key(countkey):
                metric_dictionary[countkey] = (metric_dictionary[countkey] + article_dictionary[countkey])
            else:
                metric_dictionary[countkey] = article_dictionary[countkey]

    return {"4D_dict": metric_dictionary}


def New_Topic_Model(starttimeindex, dataweek, metricdict, uniquewordlist, uniquelocationlist, uniquesourcelist, timeindex, K, alpha, beta, gamma, max_iter, burn_in_iter, sampling_lag):
    print 'Initialization....'
    U = len(uniquewordlist)
    V = len(uniquelocationlist)
    S = len(uniquesourcelist)
    N = len(timeindex)
    print N
    worddict = {}
    locationdict = {}
    sourcedict = {}
    for u in xrange(U):
        worddict[uniquewordlist[u]] = u
    for v in xrange(V):
        locationdict[uniquelocationlist[v]] = v
    for s in xrange(S):
        sourcedict[uniquesourcelist[s]] = s
    wordtopiclist = beta * ones((K, U))
    topiclocationmetric = alpha * ones((V, K))
    topictimemetric = gamma * ones((K, N))
    wordmetric = (beta * U) * ones((K, 1))
    locationmetric = (alpha * K) * ones((V, 1))
    timemetric = (gamma * N) * ones((K, 1))
    estimatedword = zeros((K, U))
    estimatedlocation = zeros((V, K))
    estimatedtime = zeros((K, N))
    topicdict = {}
    for key in metricdict:
        word = key[0]
        location = key[1]
        source = key[2]
        t = key[3]
        topicdict[(word, location, source, t)] = []
        for m in xrange(metricdict[(word, location, source, t)]):
            randomsample = numpy.random.multinomial(1, [1 / float(K)] * K, size=1)[0]
            randomsampleindex = nonzero(randomsample == 1)[0][0]
            topicdict[(word, location, source, t)].append(randomsampleindex)
            wordtopiclist[topicdict[(word, location, source, t)][m]][worddict[word]] += 1
            topiclocationmetric[locationdict[location]][topicdict[(word, location, source, t)][m]] += 1
            topictimemetric[topicdict[(word, location, source, t)][m]][t] += 1
            wordmetric[topicdict[(word, location, source, t)][m]] += 1
            locationmetric[locationdict[location]] += 1
            timemetric[topicdict[(word, location, source, t)][m]] += 1
    print 'Fast Collapsed Gibbs sampling starts'
    read_out_word = zeros((K, U))
    read_out_location = zeros((V, K))
    read_out_time = zeros((K, N))
    read_out_sampling_num = 0
    for iter in xrange(1, max_iter + 1):
        print '.',
        for key in metricdict:
            word = key[0]
            location = key[1]
            source = key[2]
            t = key[3]
            for m in xrange(metricdict[(word, location, source, t)]):
                wordtopiclist[topicdict[(word, location, source, t)][m]][worddict[word]] -= 1
                topiclocationmetric[locationdict[location]][topicdict[(word, location, source, t)][m]] -= 1
                topictimemetric[topicdict[(word, location, source, t)][m]][t] -= 1
                wordmetric[topicdict[(word, location, source, t)][m]] -= 1
                locationmetric[locationdict[location]] -= 1
                timemetric[topicdict[(word, location, source, t)][m]] -= 1
                topicprobs = numpy.zeros((1, K))
                for rr in xrange(K):
                    topicprobs[0][rr] = (wordtopiclist[rr][worddict[word]] / wordmetric[rr]) * (topiclocationmetric[locationdict[location]][rr] / locationmetric[locationdict[location]]) * (topictimemetric[rr][t] / timemetric[rr])
                topicprobs[0, :] = topicprobs[0, :] / sum(topicprobs[0, :])
                topic_sample = numpy.random.multinomial(1, topicprobs[0, :], size=1)[0] 
                topicdict[(word, location, source, t)][m] = nonzero(topic_sample == 1)[0][0]
                wordtopiclist[topicdict[(word, location, source, t)][m]][worddict[word]] += 1
                topiclocationmetric[locationdict[location]][topicdict[(word, location, source, t)][m]] += 1
                topictimemetric[topicdict[(word, location, source, t)][m]][t] += 1
                wordmetric[topicdict[(word, location, source, t)][m]] += 1
                locationmetric[locationdict[location]] += 1
                timemetric[topicdict[(word, location, source, t)][m]] += 1
        if iter % sampling_lag == 0 or iter == 1:
            if iter >= burn_in_iter:
                read_out_sampling_num += 1
                for kk in range(K):
                        read_out_word[kk, :] += wordtopiclist[kk, :] / wordmetric[kk]
                for mm in range(V):
                    read_out_location[mm, :] += topiclocationmetric[mm, :] / locationmetric[mm]
                for nn in range(K):
                    read_out_time[nn, :] += topictimemetric[nn, :] / timemetric[nn]
            print '\n',
    estimatedword = read_out_word / read_out_sampling_num
    estimatedlocation = read_out_location / read_out_sampling_num
    estimatedtime = read_out_time / read_out_sampling_num
    return {'worddict': worddict, 'locationdict': locationdict, 'sourcedict': sourcedict, 'word': estimatedword, 'wordmetric': wordtopiclist, 'location': estimatedlocation, 'locationmetric': topiclocationmetric, 'timemetric': topictimemetric, 'time': estimatedtime}


def parse_args():

    '''

    Reads the command line options and parses the appropriate commands

    '''
    import argparse
    ap = argparse.ArgumentParser('New Topic Model')

    # Required Program Arguments
    ap.add_argument("-s", "--start", type=str, default=date.today().strftime('%Y-%m-%d'), required=False, help="End Date for topic modeling, should be a Sunday. E.g. 2012-10-07. Default=Today")
    ap.add_argument("-i", "--inputfolder", type=str, help="Input folder containing the tar files", default="~/saurav/source/data")
    ap.add_argument("-o", "--outputfolder", type=str, help="outputfolder where the new tar files will be dumped", default="~/saurav/source/data")
    arg = ap.parse_args()
    return arg


def main():

    _arg = parse_args()
    start_date = date(2012, 06, 03)
    today_date = datetime.strptime(_arg.start, '%Y-%m-%d').date()
    from_from_date = start_date
    to_date = today_date + timedelta(days=6)
    inputfolder = _arg.inputfolder
    outputfolder = _arg.outputfolder
    print today_date
    json_data = []
    filelist = []
    totalwordlist = []
    totalprovincestate = []
    totalsourcelist = []
    stopwords_list = pickle.load(open(inputfolder + "/stop_list", "rb"))
    stopwords_list += ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'IX', 'X', 'VIII']

    for f1 in sorted(glob.glob(inputfolder + "/hmdata*")):
        filename = os.path.basename(f1)
        filedate = date(int(re.split('\W+', filename)[1]), int(re.split('\W+', filename)[2]), int(re.split('\W+', filename)[3]))
        if start_date <= filedate <= today_date - timedelta(days=1):
            print f1
            filelist.append(f1)

    for f2 in filelist:
        f = open(f2, "r")
        for line in f:
            try:
                json_data.append(json.loads(line, encoding="utf-8"))
            except Exception:
                continue
        f.close()

    id_list = []

    for element in json_data:
        if "embers_id" in element:
            id_list.append(element["embers_id"])
        else:
            id_list.append(element["embersId"])
    id_list = list(set(id_list))
    from_date = from_from_date
    prevtimestamp = []
    timestamp = []
    while from_date <= to_date:
        timestamp.append(from_date)
        from_date = from_date + timedelta(days=7)
    from_date = from_from_date
    while start_date < from_date:
        prevtimestamp.append(start_date)
        start_date = start_date + timedelta(days=7)
    starttimeindex = len(prevtimestamp)
    dataweek = len(timestamp) - 1
    timeindex = range(starttimeindex + dataweek)
    print dataweek
    print starttimeindex

    result_1 = tensor(json_data, id_list, timestamp, stopwords_list, starttimeindex)
    totalmetricdict = result_1["4D_dict"]
    print len(totalmetricdict)
    for metric_key in totalmetricdict:
        if metric_key[0] == "hanta":
            print metric_key, totalmetricdict[metric_key]
        totalwordlist.append(metric_key[0])
        totalprovincestate.append(metric_key[1])
        totalsourcelist.append(metric_key[2])
    totalwordlist = list(set(totalwordlist))
    totalprovincestate = list(set(totalprovincestate))
    totalsourcelist = list(set(totalsourcelist))
    start_t = time.time()
    result = New_Topic_Model(starttimeindex, dataweek, totalmetricdict, totalwordlist, totalprovincestate, totalsourcelist, timeindex, 12, 0.16, 0.01, 0.01, 500, 250, 50)
    end_t = time.time()
    print (end_t - start_t) / 3600., "hours"
    os.chdir(outputfolder)
    pickle.dump(result['word'], open('newestimatedword.pkl', 'wb'))
    pickle.dump(result['location'], open('newestimatedlocation.pkl', 'wb'))
    pickle.dump(result['time'], open('newestimatedtime.pkl', 'wb'))
    pickle.dump(result['wordmetric'], open('newestimatedwordmetric.pkl', 'wb'))
    pickle.dump(result['locationmetric'], open('newestimatedlocationmetric.pkl', 'wb'))
    pickle.dump(result['timemetric'], open('newestimatedtimemetric.pkl', 'wb'))
    pickle.dump(result['worddict'], open('newworddict.pkl', 'wb'))
    pickle.dump(result['locationdict'], open('newlocationdict.pkl', 'wb'))
    pickle.dump(result['sourcedict'], open('newsourcedict.pkl', 'wb'))
    pickle.dump(totalwordlist, open('totalword.pkl', 'wb'))
    pickle.dump(totalprovincestate, open('totalprovince.pkl', 'wb'))
    pickle.dump(totalsourcelist, open('totalsource.pkl', 'wb'))
    pickle.dump(totalmetricdict, open('totaldictionary.pkl', 'wb'))
    pickle.dump(id_list, open("id_list.pkl", "wb"))

    with tarfile.open("new_state_pkl_file-" + date.isoformat(today_date) + ".tar", "w") as write_tar:
        for name in ['totalword.pkl', 'totalprovince.pkl', 'totalsource.pkl',
                     'totaldictionary.pkl', 'newestimatedword.pkl',
                     'newestimatedlocation.pkl', 'newestimatedtime.pkl',
                     'newestimatedwordmetric.pkl',
                     'newestimatedlocationmetric.pkl', 'newestimatedtimemetric.pkl',
                     'newworddict.pkl', 'newlocationdict.pkl', 'newsourcedict.pkl', 'id_list.pkl']:
            write_tar.add(name)
    write_tar.close()
    os.system('rm *.pkl')

if __name__ == "__main__":

    main()

