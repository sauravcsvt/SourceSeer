#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

This script calculates the topic coverage of each source of Healthmap news articles. The topic coverage is used as a feature in multiplicative weights algorithm

"""

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import os
import numpy as np
import argparse
import tarfile
import cPickle as pickle
from datetime import datetime, date
import statsmodels.tsa.ar_model as ts
import multiprocessing
import time
import math


def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n)]

# This function calculates the topic coverage for each source by multiprocessing


def predict(result_queue, estimatedtime, estimatedlocation, sourcedict, locationdict, worddict, estimatedword, totalword, totalprovince, totalsource, totaldictionary):

    numtopic = np.shape(estimatedword)[0]
    timelength = np.shape(estimatedtime)[1]
    validationweeks = int(round(timelength / 3.))
    predtopic = np.zeros((1, numtopic))
    tsldict = {}
    for k in xrange(numtopic):
        predvalues = np.zeros((1, 6))
        prederror = np.zeros((1, 6))
        for b in xrange(6):
            validationerror = []
            sumerror = 0
            for j in xrange(validationweeks + 1):
                model = ts.AR(estimatedtime[k][0:timelength - validationweeks + j], freq='W')
                try:
                    result = model.fit(maxlag=b + 1, method='cmle', trend='c', maxiter=35, disp=-1)
                    model = result.model
                    if j == 0:
                        params = result.params
                    predictions = model.predict(params=params, start=timelength - validationweeks + j - 1, end=timelength - validationweeks + j)
                    if j != validationweeks:
                        if predictions[1] < 0:
                            predictions[1] = 0
                        validationerror.append(pow(predictions[1] - estimatedtime[k][timelength - validationweeks + j], 2))
                    elif j == validationweeks:
                        if predictions[1] < 0:
                            predictions[1] = 0
                        predvalues[0][b] = predictions[1]
                except Exception:
                    if j != validationweeks:
                        continue
                    else:
                        predvalues[0][b] = 0
            if len(validationerror) != 0:
                sumerror = sum(validationerror)
                prederror[0][b] = sumerror / len(validationerror)
            else:
                prederror[0][b] = 100
        minerrorindex = np.argsort(prederror[0])[0]
        predtopic[0][k] = predvalues[0][minerrorindex]

    sum_deno = 0
    for i in xrange(timelength):
        sum_deno += 1 / (float(i) + 1)
    source_word = {}
    source_loc = {}
    source_word_loc = {}
    for metrickey in totaldictionary:
        word = metrickey[0]
        location = metrickey[1]
        source = metrickey[2]
        if (word, source) in source_word:
            source_word[(word, source)].append(metrickey[3])
        else:
            source_word[(word, source)] = []
            source_word[(word, source)].append(metrickey[3])
        if (source, location) in source_loc:
            source_loc[(source, location)] += totaldictionary[metrickey]
        else:
            source_loc[(source, location)] = 0
            source_loc[(source, location)] += totaldictionary[metrickey]
        if (word, source, location) in source_word_loc:
            source_word_loc[(word, source, location)] += totaldictionary[metrickey]
        else:
            source_word_loc[(word, source, location)] = 0
            source_word_loc[(word, source, location)] += totaldictionary[metrickey]

    for key_element in source_word:
        source_word[key_element] = list(set(source_word[key_element]))

    wordcountdict = {}
    sumwordavecount = {}

    for word in totalword:
        wordcountdict[word] = 0

    for wordkey in totaldictionary:
        wordcountdict[wordkey[0]] += totaldictionary[wordkey]
    for word in totalword:
        sumwordavecount[word] = wordcountdict[word] / float(timelength)

    topicwordcontent = np.zeros((numtopic, len(totalword)))
    for k in xrange(numtopic):
        for word in totalword:
            topicwordcontent[k][worddict[word]] = sumwordavecount[word] * estimatedword[k][worddict[word]]
    for location in totalprovince:
        for source in totalsource:
            singleworddict = np.zeros((1, len(totalword)))
            for word in totalword:
                source_pub = 0
                if (word, source) in source_word:
                    for element in source_word[(word, source)]:
                        source_pub += (1 / (float(timelength) - float(element)))
                    source_pub = source_pub / float(sum_deno)
                if (word, source, location) in source_word_loc:
                    sl_prob = source_word_loc[(word, source, location)] / float(source_loc[(source, location)])
                else:
                    sl_prob = 0
                sumtopic = 0
                for k in xrange(numtopic):
                    sumtopic += (estimatedword[k][worddict[word]] * estimatedlocation[locationdict[location]][k] * predtopic[0][k])
                singleworddict[0][worddict[word]] = (sumwordavecount[word] * source_pub * sumtopic * sl_prob)
            KL_dist_topic = similarity(topicwordcontent, singleworddict, numtopic, worddict, totalword)
            for k in xrange(numtopic):
                tsldict[(k, source, location, timelength)] = KL_dist_topic[0][k]
    result_queue.put(tsldict)

    return


def similarity(topicwordcontent, word_count_vec, numtopic, worddict, totalword):

    KL_dist_topic = np.zeros((1, numtopic))
    for k in xrange(numtopic):
        sumxx = 0
        sumxy = 0
        sumyy = 0
        for word in totalword:
            x = topicwordcontent[k][worddict[word]]
            y = word_count_vec[0][worddict[word]]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        KL_dist_topic[0][k] = sumxy / math.sqrt(sumxx * sumyy)
    return KL_dist_topic


def parse_args():

    '''

    Reads the command line options and parses the appropriate commands

    '''

    ap = argparse.ArgumentParser('Topic coverage calculation code')

    # Required Program arguments

    ap.add_argument("-s", "--start", type=str,
                    default=date.today().strftime("%Y-%m-%d"), required=False,
                    help="Date for source relevances are to be calculated. e.g. 2013-01-06. Default=Today")
    ap.add_argument("-i", "--inputfolder", type=str, help="Input Folder containing the output of the tensor decomposition, i.e. the .tar state pkl file for the current week",
                    default="~/saurav/source/data")

    arg = ap.parse_args()

    return arg


def main():

    start_time = time.time()

    _arg = parse_args()

    start_pred_date = datetime.strptime(_arg.start, "%Y-%m-%d").date()
    print start_pred_date
    datapath = _arg.inputfolder   # folder containing the output of the tensor decomposition
    os.chdir(datapath)

    topsourceloctimedict = {}
    read_tar = tarfile.open("new_state_pkl_file-" + date.isoformat(start_pred_date) + ".tar", 'r')

    estimatedword = pickle.load(read_tar.extractfile('newestimatedword.pkl'))
    estimatedtime = pickle.load(read_tar.extractfile('newestimatedtime.pkl'))
    estimatedlocation = pickle.load(read_tar.extractfile('newestimatedlocation.pkl'))
    locationdict = pickle.load(read_tar.extractfile('newlocationdict.pkl'))
    sourcedict = pickle.load(read_tar.extractfile('newsourcedict.pkl'))
    worddict = pickle.load(read_tar.extractfile('newworddict.pkl'))
    totalwordlist = pickle.load(read_tar.extractfile('totalword.pkl'))
    totalprovincelist = pickle.load(read_tar.extractfile('totalprovince.pkl'))
    totalsourcelist = pickle.load(read_tar.extractfile('totalsource.pkl'))
    totalmetricdict = pickle.load(read_tar.extractfile('totaldictionary.pkl'))
    read_tar.close()
    print len(totalsourcelist)
    print len(totalprovincelist)
    truncated_loc = []
    for location in totalprovincelist:
        if location[1] in ["Chile", "Brazil", "Argentina", "Uruguay"]:
            truncated_loc.append(location)
    print len(truncated_loc)
    source_partition = partition(totalsourcelist, 80)

    # implementing multiprocessing to run the code faster

    result_queue = multiprocessing.Queue()

    jobs = [multiprocessing.Process(target=predict, args=(result_queue, estimatedtime, estimatedlocation, sourcedict, locationdict, worddict, estimatedword, totalwordlist, truncated_loc, totalsource, totalmetricdict))
            for totalsource in source_partition]

    # Starting the jobs

    for job in jobs:
        job.start()

    all_data = [result_queue.get() for job in jobs]

    for job in jobs:
        job.join()

    print "multiprocessing completed"

    for dictionary in all_data:
        for key in dictionary:
            topsourceloctimedict[key] = dictionary[key]

    pickle.dump(topsourceloctimedict, open("topsourceloctimedict", "wb"))

    with tarfile.open("new_state_pkl_file-" + date.isoformat(start_pred_date) + ".tar", 'a') as write_tar:
        write_tar.add("topsourceloctimedict")

    os.system("rm topsourceloctimedict")

    print (time.time() - start_time) / 3600., "hours"

if __name__ == "__main__":

    main()
