#!/usr/bin/python
# -*- coding: utf-8 -*-


__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import cPickle as pickle
import os
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.parser import parse as dtparse
import tarfile
import time
from tuples_gen import fourD_tuples
from pymongo import MongoClient


class spatio_TM(object):
    '''
    Class for executing the spatio-temporal topic model on the 4D tuples (word, location, source, time)
    '''
    
    def __init__(self, **kwargs):
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.max_iter = kwargs['max_iter']
        self.burn_in_iter = kwargs['burn_in_iter']
        self.sampling_lag = kwargs['sampling_lag']
        self.K = kwargs['num_topic']
        return

    def New_Topic_Model(self, starttimeindex, dataweek, timeindex, metricdict, 
                        uniquewordlist, uniquelocationlist, uniquesourcelist):
        print 'Initialization....'
        U = len(uniquewordlist)
        V = len(uniquelocationlist)
        S = len(uniquesourcelist)
        N = len(timeindex)
        worddict = {}
        locationdict = {}
        sourcedict = {}
        for u in xrange(U):
            worddict[uniquewordlist[u]] = u
        for v in xrange(V):
            locationdict[uniquelocationlist[v]] = v
        for s in xrange(S):
            sourcedict[uniquesourcelist[s]] = s
        wordtopiclist = self.beta * np.ones((self.K, U))
        topiclocationmetric = self.alpha * np.ones((V, self.K))
        topictimemetric = self.gamma * np.ones((self.K, N))
        wordmetric = (self.beta * U) * np.ones((self.K, 1))
        locationmetric = (self.alpha * self.K) * np.ones((V, 1))
        timemetric = (self.gamma * N) * np.ones((self.K, 1))
        estimatedword = np.zeros((self.K, U))
        estimatedlocation = np.zeros((V, self.K))
        estimatedtime = np.zeros((self.K, N))
        topicdict = {}
        for key in metricdict:
            word = key[0]
            location = key[1]
            source = key[2]
            t = key[3]
            topicdict[(word, location, source, t)] = []
            for m in xrange(metricdict[(word, location, source, t)]):
                randomsample = np.random.multinomial(1, [1 / float(self.K)] * self.K, size=1)[0]
                randomsampleindex = np.nonzero(randomsample == 1)[0][0]
                topicdict[(word, location, source, t)].append(randomsampleindex)
                wordtopiclist[topicdict[(word, location, source, t)][m]][worddict[word]] += 1
                topiclocationmetric[locationdict[location]][topicdict[(word, location, source, t)][m]] += 1
                topictimemetric[topicdict[(word, location, source, t)][m]][t] += 1
                wordmetric[topicdict[(word, location, source, t)][m]] += 1
                locationmetric[locationdict[location]] += 1
                timemetric[topicdict[(word, location, source, t)][m]] += 1
        print 'Gibbs sampling starts'
        read_out_word = np.zeros((self.K, U))
        read_out_location = np.zeros((V, self.K))
        read_out_time = np.zeros((self.K, N))
        read_out_sampling_num = 0
        for iter in xrange(1, self.max_iter + 1):
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
                    topicprobs = np.zeros((1, self.K))
                    # Calculation the probability of each topic to be assigned
                    # for this tuple
                    for rr in xrange(self.K):
                        topicprobs[0][rr] = (wordtopiclist[rr][worddict[word]] / wordmetric[rr]) * \
                                            (topiclocationmetric[locationdict[location]][rr] / locationmetric[locationdict[location]]) * \
                                            (topictimemetric[rr][t] / timemetric[rr])
                    # Normalizing the topic probabilities
                    topicprobs[0, :] = topicprobs[0, :] / sum(topicprobs[0, :])
                    # Sampling the topic to be assigned to this tuple
                    topic_sample = np.random.multinomial(1, topicprobs[0, :], size=1)[0] 
                    topicdict[(word, location, source, t)][m] = np.nonzero(topic_sample == 1)[0][0]
                    wordtopiclist[topicdict[(word, location, source, t)][m]][worddict[word]] += 1
                    topiclocationmetric[locationdict[location]][topicdict[(word, location, source, t)][m]] += 1
                    topictimemetric[topicdict[(word, location, source, t)][m]][t] += 1
                    wordmetric[topicdict[(word, location, source, t)][m]] += 1
                    locationmetric[locationdict[location]] += 1
                    timemetric[topicdict[(word, location, source, t)][m]] += 1
            if iter % self.sampling_lag == 0 or iter == 1:
                if iter >= self.burn_in_iter:
                    read_out_sampling_num += 1
                    for kk in range(self.K):
                        read_out_word[kk, :] += wordtopiclist[kk, :] / wordmetric[kk]
                    for mm in range(V):
                        read_out_location[mm, :] += topiclocationmetric[mm, :] / locationmetric[mm]
                    for nn in range(self.K):
                        read_out_time[nn, :] += topictimemetric[nn, :] / timemetric[nn]
                print '\n',
        estimatedword = read_out_word / read_out_sampling_num
        estimatedlocation = read_out_location / read_out_sampling_num
        estimatedtime = read_out_time / read_out_sampling_num
        return {'worddict': worddict, 'locationdict': locationdict, 'sourcedict': sourcedict, 
                'word': estimatedword, 'wordmetric': wordtopiclist, 'location': estimatedlocation, 
                'locationmetric': topiclocationmetric, 'timemetric': topictimemetric, 'time': estimatedtime}


def parse_args():

    '''

    Reads the command line options and parses the appropriate commands

    '''
    import argparse
    ap = argparse.ArgumentParser('New Topic Model')

    # Required Program Arguments
    ap.add_argument("-s", "--start", type=str, required=True, help="End Date for topic modeling, should be a Sunday. E.g. 2012-10-07. Default=Today")
    ap.add_argument("-o", "--outputfolder", type=str, required=True, help="outputfolder where the topic-specific distributions will be dumped")
    arg = ap.parse_args()
    return arg


def main():

    _arg = parse_args()
    start_date = date(2012, 06, 03)
    today_date = datetime.strptime(_arg.start, '%Y-%m-%d').date()
    from_from_date = start_date
    to_date = today_date + timedelta(days=6)
    json_data = []

    # Read input HealthMap articles from MongoDB
    
    db, collection = 'Rare', 'HealthMap'
    _client = MongoClient()
    _collection = _client[db][collection]
    for artl in _collection.find():
        if dtparse(artl['date']).date() < today_date:
            json_data.append(artl)
    
    outputfolder = _arg.outputfolder
    print today_date
    print len(json_data)
    totalwordlist = []
    totalprovincestate = []
    totalsourcelist = []

    # Reading the list of stopwords from MongoDB

    stop_dict = _client[db]['STOPWORDS'].find_one()
    stop_dict.pop("_id")
    stopwords_list = stop_dict.values()
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
    print "Start Epiweek Index for topic modeling is : {}".format(starttimeindex)
    print "Total number of Epiweeks is : {}".format(dataweek)
    print "End Epiweek Index for topic modeling is : {}".format(timeindex[-1])

    tuples_obj = fourD_tuples(json_data, timestamp, stopwords_list, starttimeindex)
    tuples_obj.final_tuples()
    totalmetricdict = tuples_obj.final_dict
    print "Total number of 4D tuples upto current timepoint is: {}".format(len(totalmetricdict))
    for metric_key in totalmetricdict:
        totalwordlist.append(metric_key[0])
        totalprovincestate.append(metric_key[1])
        totalsourcelist.append(metric_key[2])
    totalwordlist = list(set(totalwordlist))
    totalprovincestate = list(set(totalprovincestate))
    totalsourcelist = list(set(totalsourcelist))
    start_t = time.time()
    topic_params = {'alpha': 0.16, 'beta': 0.01, 'gamma': 0.01, 'num_topic': 12,
                    'max_iter': 500, 'burn_in_iter': 250, 'sampling_lag': 50}
    spatio_tpobj = spatio_TM(**topic_params)
    result_topic = spatio_tpobj.New_Topic_Model(starttimeindex, dataweek, timeindex, 
                                                totalmetricdict, totalwordlist, 
                                                totalprovincestate, totalsourcelist)
    end_t = time.time()
    print "Total time (in hours) taken for topic modeling is: {}".format((end_t - start_t) / 3600.)
    os.chdir(outputfolder)
    pickle.dump(result_topic['word'], open('newestimatedword.pkl', 'wb'))
    pickle.dump(result_topic['location'], open('newestimatedlocation.pkl', 'wb'))
    pickle.dump(result_topic['time'], open('newestimatedtime.pkl', 'wb'))
    pickle.dump(result_topic['wordmetric'], open('newestimatedwordmetric.pkl', 'wb'))
    pickle.dump(result_topic['locationmetric'], open('newestimatedlocationmetric.pkl', 'wb'))
    pickle.dump(result_topic['timemetric'], open('newestimatedtimemetric.pkl', 'wb'))
    pickle.dump(result_topic['worddict'], open('newworddict.pkl', 'wb'))
    pickle.dump(result_topic['locationdict'], open('newlocationdict.pkl', 'wb'))
    pickle.dump(result_topic['sourcedict'], open('newsourcedict.pkl', 'wb'))
    pickle.dump(totalwordlist, open('totalword.pkl', 'wb'))
    pickle.dump(totalprovincestate, open('totalprovince.pkl', 'wb'))
    pickle.dump(totalsourcelist, open('totalsource.pkl', 'wb'))
    pickle.dump(totalmetricdict, open('totaldictionary.pkl', 'wb'))

    with tarfile.open("new_state_pkl_file-" + date.isoformat(today_date) + ".tar", "w") as write_tar:
        for name in ['totalword.pkl', 'totalprovince.pkl', 'totalsource.pkl',
                     'totaldictionary.pkl', 'newestimatedword.pkl',
                     'newestimatedlocation.pkl', 'newestimatedtime.pkl',
                     'newestimatedwordmetric.pkl',
                     'newestimatedlocationmetric.pkl', 'newestimatedtimemetric.pkl',
                     'newworddict.pkl', 'newlocationdict.pkl', 'newsourcedict.pkl']:
            write_tar.add(name)
    write_tar.close()
    os.system('rm *.pkl')

if __name__ == "__main__":

    main()

