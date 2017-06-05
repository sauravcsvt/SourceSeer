#!/usr/bin/python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

__author__ = "Theodoros Rekatsinas"
__email__ = "thodrek@cs.umd.edu"
import sys
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import glob
import os
import tarfile
import cPickle as pickle
import xlrd
import multiprocessing
import warnings


def validate(X_train, X_excluded, clf):
    errorNormal = 0.0
    errorAnomalous = 0.0
    # evaluate for normal points
    clf.fit(X_train)
    pred_normal = clf.predict(X_train)
    for p in pred_normal:
        if p == -1.0:
            errorNormal += 1.0
    
    # iterate over anomalous points
    # train svm with all X_train points
    clf.fit(X_excluded)
    pred_anomalous = clf.predict(X_excluded)
    for p in pred_anomalous:
        if p == 1.0:
            errorAnomalous += 1.0
    
    # Overall performance
    avgErrorNormal = float(errorNormal) / float(len(X_train))
    avgErrorAnomalous = 0.0
    wNormal = 1.0
    wAnomalous = 0.0
    if len(X_excluded) > 0:
        avgErrorAnomalous = float(errorAnomalous) / float(len(X_excluded))
        wNormal = 0.5
        wAnomalous = 0.5
    overallError = wNormal * (avgErrorNormal) + wAnomalous * (avgErrorAnomalous)
    return overallError


def leaveOneOutCrossValidation(X_train, X_excluded, clf, c):
    errorNormal = 0.0
    errorAnomalous = 0.0
    # iterate over normal points
    for k in range(len(X_train)):
        # leave one out and concatenate
        cur_X_train = np.concatenate((X_train[:k], X_train[(k + 1):]), axis=0)
        # train svm
        clf.fit(cur_X_train)
        # evaluate
        point = [X_train[k]]
        point_ar = np.array(point)
        prediction = clf.predict(point_ar)[0]
        if prediction == -1.0:
            errorNormal += 1.0
    # iterate over anomalous points
    # train svm with all X_train points
    clf.fit(X_train)
    for k in X_excluded:
        # evaluate
        point = [k]
        point_ar = np.array(point)
        prediction = clf.predict(point_ar)[0]
        if prediction == 1.0:
            errorAnomalous += 1.0
    
    # Overall performance
    avgErrorNormal = float(errorNormal) / float(len(X_train))
    avgErrorAnomalous = 0.0
    wNormal = 1.0
    wAnomalous = 0.0
    if len(X_excluded) > 0:
        avgErrorAnomalous = float(errorAnomalous) / float(len(X_excluded))
        wNormal = 0.1
        wAnomalous = 0.9
    firstRoundError = wNormal * (avgErrorNormal) + wAnomalous * (avgErrorAnomalous)
    overallError = 0.5 * firstRoundError + 0.5 * validate(X_train, X_excluded, clf)
    return overallError


def findSVMForSource(args):
    curr_proc = multiprocessing.current_process()
    # uncomment following line to get this to work
    curr_proc.daemon = False
    c, s, src, training_data, hanta_topic_dict = args
    # iterate over training data for source
    X_train = []
    X_excluded = []
    X_total = []
    for w in sorted(training_data):
        dataPoint = []
        for t in hanta_topic_dict[(s, c)]:
            dataPoint.append(round(training_data[w][t], 2))
        if sum(dataPoint) == 0.0:
            continue
        X_total.append(dataPoint)
        
        if (training_data[w]["_actValue"] == -1):
            X_excluded.append(dataPoint)
        else:
            X_train.append(dataPoint)
    
    if (len(X_train) >= 10):
        # find best OCSVM
        try:
            clf_info = findBestOCSVM(X_train, X_excluded, c, s, src)
            return (src, clf_info)
        except Exception:
            return (src, None)
    else:
        return (src, None)


def computeCLFScore(args):
    config, X_train_ar, X_excluded_ar, c = args
    if config['kernel'] == "rbf":
        clf = svm.OneClassSVM(nu=config['nuValue'], kernel="rbf", gamma=config['gValue'])
    elif config['kernel'] == "linear":
        clf = svm.OneClassSVM(nu=config['nuValue'], kernel="linear")
    else:
        clf = svm.OneClassSVM(nu=config['nuValue'], kernel=config['kernel'], gamma=config['gValue'], degree=config['degree'], coef0=config['coef0'])
    score = leaveOneOutCrossValidation(X_train_ar, X_excluded_ar, clf, c)
    return score


def findBestOCSVM(X_train, X_excluded, c, s, src):
    # Initialize labels
    X_total = []
    for e in X_train:
        X_total.append(e)
    for e in X_excluded:
        X_total.append(e)
    
    # Format lists to arrays
    X_train_ar = np.array(X_train)
    X_excluded_ar = np.array(X_excluded)
    X_total_ar = np.array(X_total)
    # Scale arrays to zero mean unit std but not used
    scaler = StandardScaler().fit(X_total_ar)   
    X_train_ar_scaled = X_train_ar
    X_excluded_ar_scaled = X_excluded_ar
    X_total_ar_scaled = X_total_ar
   
    # Initialize parameter options
    nurbf = [0.0001, 0.001, 0.01, 0.1]
    nurbf.reverse()
    nupoly = [0.0001, 0.001, 0.01, 0.1, 0.5]
    nupoly.reverse()
    gammarbf = [0.1, 0.5, 1.0, 2.0]
    gammapoly = [0.1, 0.5, 1.0, 3.0]
    coef0_values = [0.1, 1.0, -0.5, -1.0, -2.0]
    degreepoly = [3, 5, 7]
    
    # Iterate over parameters and run leave-one-out cross validation
    clf_config_options = []
    for nuValue in nurbf:
        for gValue in gammarbf:
            newConfig = {}
            newConfig['nuValue'] = nuValue
            newConfig['kernel'] = "rbf"
            newConfig['gValue'] = gValue
            clf_config_options.append(newConfig)
    for nuValue in nupoly:
        for gValue in gammapoly:
            for dValue in degreepoly:
                for co_val in coef0_values:
                    newConfig = {}
                    newConfig['nuValue'] = nuValue
                    newConfig['kernel'] = "poly"
                    newConfig['gValue'] = gValue
                    newConfig['degree'] = dValue
                    newConfig['coef0'] = co_val
                    clf_config_options.append(newConfig)
    for nuValue in nupoly:
        for gValue in gammapoly:
            for co_val in coef0_values:
                newConfig = {}
                newConfig['nuValue'] = nuValue
                newConfig['kernel'] = "sigmoid"
                newConfig['gValue'] = gValue
                newConfig['degree'] = dValue
                newConfig['coef0'] = co_val
                clf_config_options.append(newConfig)
    
    for nuValue in nupoly:
        newConfig = {}
        newConfig['nuValue'] = nuValue
        newConfig['kernel'] = "linear"
        clf_config_options.append(newConfig)
    
    p = multiprocessing.Pool(30)
    clf_Scores = p.map(computeCLFScore, [(config, X_train_ar_scaled, X_excluded_ar_scaled, c) for config in clf_config_options])
    best_clf_score = 10.0
    best_conf_index = -1
    min_error = min(clf_Scores)
    min_models = []
    for i in range(len(clf_Scores)):
        if clf_Scores[i] < best_clf_score:
            best_clf_score = clf_Scores[i]
            best_conf_index = i
        if clf_Scores[i] == min_error:
            min_models.append(i)
    
    # Return best classfier
    if clf_config_options[best_conf_index]['kernel'] == "rbf":
        best_clf = svm.OneClassSVM(nu=clf_config_options[best_conf_index]['nuValue'], kernel="rbf", gamma=clf_config_options[best_conf_index]['gValue'])
    elif clf_config_options[best_conf_index]['kernel'] == "linear":
        best_clf = svm.OneClassSVM(nu=clf_config_options[best_conf_index]['nuValue'], kernel="linear")
    else:
        best_clf = svm.OneClassSVM(nu=clf_config_options[best_conf_index]['nuValue'], kernel=clf_config_options[best_conf_index]['kernel'], gamma=clf_config_options[best_conf_index]['gValue'], degree=clf_config_options[best_conf_index]['degree'], coef0=clf_config_options[best_conf_index]['coef0'])
    best_clf.fit(X_train_ar_scaled)
    # grab distance of X_total
    distances = best_clf.decision_function(X_total_ar_scaled)
    maxPositiveDistance = max(distances)
    maxNegativeDistance = min(-0.001, min(distances))
    return (best_clf, scaler, maxNegativeDistance, maxPositiveDistance)


def topicdtm(estimatedword, worddict):
    
    hanta_keywords = ["hanta", "rural", "hantavirus", "roedores", "ratones", "cardiopulmonar"]
    general_keywords = ["gripe", "dengue", "influenza", "aedes", "aegypti", "aviar"]
    
    hanta_topic = []
    general_topic = []
    
    for word in hanta_keywords:
        hanta_topic.append(np.argmax(estimatedword[:, worddict[word]]))
    
    hanta_topic = list(set(hanta_topic))
    
    for word in general_keywords:
        general_topic.append(np.argmax(estimatedword[:, worddict[word]]))
    
    general_topic = list(set(general_topic))
    
    common_topic = []
    
    for element in hanta_topic:
        if element in general_topic:
            common_topic.append(element)
    
    if len(common_topic) != 0:
        for element in common_topic:
            hanta_topic.remove(element)
    
    return hanta_topic


# read command line input
warnings.simplefilter('ignore', DeprecationWarning)
if len(sys.argv) < 6:
    print "Wrong input. Run the script as: python runSVMOneClass.py <prediction_date> <past_timewindow_duration (in months)> <input_data_dir> <output_data_dir> <warnings_dir><GSR_file>. Example: python runSVMOneClass.py 02/28/2013 5 /Users/thodoris/inputData /Users/thodoris/outputData /Users/thodoris/disease_warnings /Users/thodoris/GSR.xlxs . Format of prediction date should be: %m/%d/%Y."
else:
    pred_date = datetime.strptime(sys.argv[1], "%d/%m/%Y").date()
    warning_date = pred_date + timedelta(days=4)
    time_window_size = int(sys.argv[2])  # in months
    earliest_pred_date = pred_date + relativedelta(months=-time_window_size)
    inputData_dir_prefix = sys.argv[3]
    outputData_dir_prefix = sys.argv[4]
    warnings_dir_prefix = sys.argv[5]
    GSR_file = sys.argv[6]

# find all relevant tensor decomposition files in the input data dir
fDir = inputData_dir_prefix + '/new_state_pkl_file-*.tar'
historicalFiles = []
currentFile = ''
for f in glob.glob(fDir):
    fName = os.path.basename(f)
    fDate_str = fName.rstrip('.tar')
    fDate_str = fDate_str.split('-')
    fDate_str = fDate_str[2] + '/' + fDate_str[3] + '/' + fDate_str[1]  # format: %m/%d/%Y
    fDate = datetime.strptime(fDate_str, "%m/%d/%Y").date()
    if (earliest_pred_date <= fDate < pred_date):
        historicalFiles.append((fDate, os.path.basename(f)))
    elif (fDate == pred_date):
        currentFile = os.path.basename(f)
    else:
        continue
print pred_date
print GSR_file
# validate files
if (len(historicalFiles) == 1 or currentFile == ''):
    print "No relevant data found. Exiting..."
    sys.exit(-1)

# extract data for current week and predict disease outbreaks
# initialize country, state, topic distribution training data dictionary
testing_data_dict = {}

# iterate over prediction file and extract test data
# open file
fileName = inputData_dir_prefix + '/' + currentFile
read_tar = tarfile.open(fileName, 'r')

# extract data
estimatedword = pickle.load(read_tar.extractfile('newestimatedword.pkl'))
estimatedtime = pickle.load(read_tar.extractfile('newestimatedtime.pkl'))
estimatedlocation = pickle.load(read_tar.extractfile('newestimatedlocation.pkl'))
estimatedtimemetric = pickle.load(read_tar.extractfile('newestimatedtimemetric.pkl'))
estimatedlocationmetric = pickle.load(read_tar.extractfile('newestimatedlocationmetric.pkl'))
estimatedwordmetric = pickle.load(read_tar.extractfile('newestimatedwordmetric.pkl'))
locationdict = pickle.load(read_tar.extractfile('newlocationdict.pkl'))
sourcedict = pickle.load(read_tar.extractfile('newsourcedict.pkl'))
worddict = pickle.load(read_tar.extractfile('newworddict.pkl'))
totalwordlist = pickle.load(read_tar.extractfile('totalword.pkl'))
totalprovincelist = pickle.load(read_tar.extractfile('totalprovince.pkl'))
totalsourcelist = pickle.load(read_tar.extractfile('totalsource.pkl'))
totalmetricdict = pickle.load(read_tar.extractfile('totaldictionary.pkl'))
topsourceloctimedict = pickle.load(read_tar.extractfile('topsourceloctimedict'))
read_tar.close()

rel_src_dict = {}

for key in totalmetricdict:
    location = key[1]
    if location in rel_src_dict:
        rel_src_dict[location].append(key[2])
    else:
        rel_src_dict[location] = []
        rel_src_dict[location].append(key[2])

for location in rel_src_dict:
    rel_src_dict[location] = list(set(rel_src_dict[location]))

hanta_topic = topicdtm(estimatedword, worddict)
print hanta_topic
hanta_topic_dict = {}
for location in totalprovincelist:
    hanta_topic_dict[location] = hanta_topic

timepoint = np.shape(estimatedtime)[1]

# iterate over source information and populate testing dictionary
for entryKey in topsourceloctimedict:
    topicIndex = entryKey[0]
    if topicIndex not in hanta_topic:
        continue
    source = entryKey[1]
    state = entryKey[2][0]
    country = entryKey[2][1]
    weekIndex = entryKey[3]
    
    if source not in rel_src_dict[(state, country)]:
        continue
    
    if (country in testing_data_dict):
        if (state in testing_data_dict[country]):
            if (source in testing_data_dict[country][state]):
                testing_data_dict[country][state][source][topicIndex] = topsourceloctimedict[entryKey]
            else:
                testing_data_dict[country][state][source] = {}
                testing_data_dict[country][state][source][topicIndex] = topsourceloctimedict[entryKey]
        else:
            testing_data_dict[country][state] = {}
            testing_data_dict[country][state][source] = {}
            testing_data_dict[country][state][source][topicIndex] = topsourceloctimedict[entryKey]
    else:
        testing_data_dict[country] = {}
        testing_data_dict[country][state] = {}
        testing_data_dict[country][state][source] = {}
        testing_data_dict[country][state][source][topicIndex] = topsourceloctimedict[entryKey]

# load positive examples from GSR
positive_examples_state = {}

GSR_wb = xlrd.open_workbook(GSR_file)
GSR_sh = GSR_wb.sheet_by_name("CLEAN V1")
for i in xrange(1, GSR_sh.nrows):
    if isinstance(GSR_sh.row_values(i)[7], float):
        event_type = GSR_sh.row_values(i)[7]
    else:
        event_type = GSR_sh.row_values(i)[7].encode("UTF-8")
    
    if isinstance(event_type, float):
        event_type = int(event_type)
    
    if event_type == "0313" or event_type == "313" or event_type == 0313 or event_type == 313:
        
        country = (GSR_sh.row_values(i)[4]).encode('UTF-8')
        state = (GSR_sh.row_values(i)[5]).encode('UTF-8')
        if state == "Santiago":
            state = "Metropolitana"
        if state[0] == " ":
            state = state[1:len(state)]
        elif state[len(state) - 1] == " ":
            state = state[1:len(state) - 1]
        if state == "os Lagos":
            state = "Los Lagos"
        elif state == "aule":
            state = "Maule"
        elif state == "uenos Aires":
            state = "Buenos Aires"
        elif state == "ntre R\xc3\xados":
            state = "Entre R\xc3\xados"
        year, month, day, hour, minute, second = xlrd.xldate_as_tuple(GSR_sh.row_values(i)[9], GSR_wb.datemode)
        event_date = date(year, month, day)
        if state == "Valpara\xc3\xadso" and event_date == date(2013, 2, 22):
            state = "Los Lagos"
        
        if (country in positive_examples_state):
            if (state in positive_examples_state[country]):
                positive_examples_state[country][state].append(event_date)
            else:
                positive_examples_state[country][state] = []
                positive_examples_state[country][state].append(event_date)
        else:
            positive_examples_state[country] = {}
            positive_examples_state[country][state] = []
            positive_examples_state[country][state].append(event_date)


# initialize country, state, topic distribution training data dictionary
training_data_dict = {}
training_data_global_dict = {}

# iterate over historical files and extract training data
for fileInfo in historicalFiles:
    # open file
    fileName = inputData_dir_prefix + '/' + fileInfo[1]
    read_tar = tarfile.open(fileName, 'r')
    print fileName
    # extract data
    
    estimatedword = pickle.load(read_tar.extractfile('newestimatedword.pkl'))
    estimatedtime = pickle.load(read_tar.extractfile('newestimatedtime.pkl'))
    estimatedlocation = pickle.load(read_tar.extractfile('newestimatedlocation.pkl'))
    estimatedtimemetric = pickle.load(read_tar.extractfile('newestimatedtimemetric.pkl'))
    estimatedlocationmetric = pickle.load(read_tar.extractfile('newestimatedlocationmetric.pkl'))
    estimatedwordmetric = pickle.load(read_tar.extractfile('newestimatedwordmetric.pkl'))
    locationdict = pickle.load(read_tar.extractfile('newlocationdict.pkl'))
    sourcedict = pickle.load(read_tar.extractfile('newsourcedict.pkl'))
    worddict = pickle.load(read_tar.extractfile('newworddict.pkl'))
    totalwordlist = pickle.load(read_tar.extractfile('totalword.pkl'))
    totalprovincelist = pickle.load(read_tar.extractfile('totalprovince.pkl'))
    totalsourcelist = pickle.load(read_tar.extractfile('totalsource.pkl'))
    totalmetricdict = pickle.load(read_tar.extractfile('totaldictionary.pkl'))
    topsourceloctimedict = pickle.load(read_tar.extractfile('topsourceloctimedict'))
    
    # close file
    read_tar.close()
    
    # iterate over source information and populate training dictionary
    for entryKey in topsourceloctimedict:
        topicIndex = entryKey[0]
        if topicIndex not in hanta_topic:
            continue
        source = entryKey[1]
        state = entryKey[2][0]
        country = entryKey[2][1]
        weekIndex = entryKey[3]
        
        if source not in rel_src_dict[(state, country)]:
            continue
        if np.isnan(topsourceloctimedict[entryKey]):
            continue
        excludeStateExample = 1
        
        if (country in positive_examples_state):
            if (state in positive_examples_state[country]):
                for element in positive_examples_state[country][state]:
                    if fileInfo[0] <= element <= fileInfo[0] + timedelta(days=6):
                        excludeStateExample = -1
                
        if (country in training_data_global_dict):
            if (state in training_data_global_dict[country]):
                training_data_global_dict[country][state]["_weeks"][weekIndex] = excludeStateExample
            else:
                training_data_global_dict[country][state] = {}
                training_data_global_dict[country][state]["_weeks"] = {}
                training_data_global_dict[country][state]["_weeks"][weekIndex] = excludeStateExample
                training_data_global_dict[country][state]["_sources"] = {}
        else:
            training_data_global_dict[country] = {}
            training_data_global_dict[country][state] = {}
            training_data_global_dict[country][state]["_weeks"] = {}
            training_data_global_dict[country][state]["_weeks"][weekIndex] = excludeStateExample
            training_data_global_dict[country][state]["_sources"] = {}
        
        if (country in training_data_dict):
            if (state in training_data_dict[country]):
                if (source in training_data_dict[country][state]):
                    if (weekIndex in training_data_dict[country][state][source]):
                        training_data_dict[country][state][source][weekIndex][topicIndex] = topsourceloctimedict[entryKey]
                        training_data_dict[country][state][source][weekIndex]["_actValue"] = excludeStateExample
                    else:
                        training_data_dict[country][state][source][weekIndex] = {}
                        training_data_dict[country][state][source][weekIndex]["_actValue"] = excludeStateExample
                        training_data_dict[country][state][source][weekIndex][topicIndex] = topsourceloctimedict[entryKey]
                else:
                    training_data_dict[country][state][source] = {}
                    training_data_dict[country][state][source][weekIndex] = {}
                    training_data_dict[country][state][source][weekIndex]["_actValue"] = excludeStateExample
                    training_data_dict[country][state][source][weekIndex][topicIndex] = topsourceloctimedict[entryKey]
            else:
                training_data_dict[country][state] = {}
                training_data_dict[country][state][source] = {}
                training_data_dict[country][state][source][weekIndex] = {}
                training_data_dict[country][state][source][weekIndex]["_actValue"] = excludeStateExample
                training_data_dict[country][state][source][weekIndex][topicIndex] = topsourceloctimedict[entryKey]
        else:
            training_data_dict[country] = {}
            training_data_dict[country][state] = {}
            training_data_dict[country][state][source] = {}
            training_data_dict[country][state][source][weekIndex] = {}
            training_data_dict[country][state][source][weekIndex]["_actValue"] = excludeStateExample
            training_data_dict[country][state][source][weekIndex][topicIndex] = topsourceloctimedict[entryKey]


# train one-class svm for each country, state, source
# get total entries

total_entries = 0.0

for c in training_data_dict:
    for s in training_data_dict[c]:
        total_entries += float(len(training_data_dict[c][s]))

country_state_source_svms = {}
country_state_source_scalers = {}
country_state_source_distances = {}
processed = 0.0
for c in training_data_dict:
    country_state_source_svms[c] = {}
    country_state_source_scalers[c] = {}
    country_state_source_distances[c] = {}
    for s in training_data_dict[c]:
        country_state_source_svms[c][s] = {}
        country_state_source_scalers[c][s] = {}
        country_state_source_distances[c][s] = {}

        p = multiprocessing.Pool(30)
        clf_src_result = p.map(findSVMForSource, [(c, s, src, training_data_dict[c][s][src], hanta_topic_dict) for src in training_data_dict[c][s]])
        for e in clf_src_result:
            src = e[0]
            clf_info = e[1]
            if clf_info is not None:
                clf = clf_info[0]
                scaler = clf_info[1]
                maxNegativeDistance = clf_info[2]
                maxPositiveDistance = clf_info[3]
                country_state_source_svms[c][s][src] = clf
                country_state_source_scalers[c][s][src] = scaler
                country_state_source_distances[c][s][src] = {}
                country_state_source_distances[c][s][src]["_negD"] = maxNegativeDistance
                country_state_source_distances[c][s][src]["_posD"] = maxPositiveDistance
                training_data_global_dict[c][s]["_sources"][src] = 1
        processed += 1.0
        progress = 100 * round(processed / total_entries, 2)
        sys.stdout.write("Training progress: %d%% (%d/%d)  \r" % (progress, processed, total_entries))
        sys.stdout.flush()


# iterate over training dataset and perform multiplicative weights algorithm
country_state_source_weights = {}
country_state_source_acc_dict = {}
for c in training_data_global_dict:
    country_state_source_weights[c] = {}
    country_state_source_acc_dict[c] = {}
    for s in training_data_global_dict[c]:
        country_state_source_weights[c][s] = {}
        country_state_source_acc_dict[c][s] = {}
        epsilon = 0.1
        NumSources = len(training_data_global_dict[c][s]["_sources"])
        if NumSources == 0:
            continue
        for src in training_data_global_dict[c][s]["_sources"]:
            country_state_source_weights[c][s][src] = 1.0 / float(NumSources)
        for w in sorted(training_data_global_dict[c][s]["_weeks"]):
            actualValue = training_data_global_dict[c][s]["_weeks"][w]
            for src in training_data_global_dict[c][s]["_sources"]:
                if (w in training_data_dict[c][s][src]):
                    X_pred = []
                    dataPoint = []
                    for t in hanta_topic_dict[(s, c)]:
                        dataPoint.append(round(training_data_dict[c][s][src][w][t], 2))
                    X_pred.append(dataPoint)
                    X_pred_ar = np.array(dataPoint)
                    X_pred_ar_scaled = X_pred_ar
                    src_prediction = country_state_source_svms[c][s][src].predict(X_pred_ar_scaled)[0]
                else:
                    src_prediction = 0
                if src not in country_state_source_acc_dict[c][s]:
                    country_state_source_acc_dict[c][s][src] = {}
                    country_state_source_acc_dict[c][s][src]["_total"] = 0.0
                    country_state_source_acc_dict[c][s][src]["_correct"] = 0.0
                country_state_source_acc_dict[c][s][src]["_total"] += 1.0
                if src_prediction != actualValue:
                    country_state_source_weights[c][s][src] = float(math.exp(-epsilon)) * country_state_source_weights[c][s][src]
                else:
                    country_state_source_acc_dict[c][s][src]["_correct"] += 1.0
                    country_state_source_weights[c][s][src] = float(math.exp(epsilon)) * country_state_source_weights[c][s][src]
        totalWeight = 0.0
        for src in country_state_source_weights[c][s]:
            totalWeight += country_state_source_weights[c][s][src]
        for src in country_state_source_weights[c][s]:
            country_state_source_weights[c][s][src] = float(country_state_source_weights[c][s][src]) / float(totalWeight)


pickle.dump(country_state_source_weights, open("css_weights-" + date.isoformat(pred_date) + ".pkl", "wb"))


# generate predictions based on weighted majority for new timepoint
out_filename = warnings_dir_prefix + '/disease_warnings-' + date.isoformat(warning_date) + '.txt'
out = open(out_filename, 'w')
pred_dict = {}
pred_weight_dict = {}
negative_count = {}
pred_conf_dict = {}
pred_acc_dict = {}
for c in testing_data_dict:
    if (c not in country_state_source_weights):
        continue
    negative_count[c] = {}
    pred_conf_dict[c] = {}
    pred_acc_dict[c] = {}
    for s in testing_data_dict[c]:
        if (s not in country_state_source_weights[c]):
            continue
        negative_count[c][s] = 0
        pred_conf_dict[c][s] = []
        pred_acc_dict[c][s] = {}
        pred_acc_dict[c][s]["_pos"] = []
        pred_acc_dict[c][s]["_neg"] = []
        majorityPrediction = 0
        src_pred = {}
        X_pred_to_print = {}
        for src in testing_data_dict[c][s]:
            if (src in country_state_source_svms[c][s] and src in country_state_source_weights[c][s] and src in country_state_source_acc_dict[c][s]):
                X_pred = []
                dataPoint = []
                for t in hanta_topic_dict[(s, c)]:
                    dataPoint.append(round(training_data_dict[c][s][src][w][t], 2))
                X_pred.append(dataPoint)
                X_pred_ar = np.array(X_pred)
                X_pred_ar_scaled = X_pred_ar
                src_pred[src] = country_state_source_svms[c][s][src].predict(X_pred_ar_scaled)[0]
                if src_pred[src] == -1.0:
                    negative_count[c][s] += 1
                    src_conf = country_state_source_svms[c][s][src].decision_function(X_pred_ar_scaled) / country_state_source_distances[c][s][src]["_negD"]
                    pred_conf_dict[c][s].append(src_conf)
                    pred_acc_dict[c][s]["_neg"].append(country_state_source_acc_dict[c][s][src]["_correct"] / country_state_source_acc_dict[c][s][src]["_total"])
                else:
                    pred_acc_dict[c][s]["_pos"].append(country_state_source_acc_dict[c][s][src]["_correct"] / country_state_source_acc_dict[c][s][src]["_total"])
                majorityPrediction += country_state_source_weights[c][s][src] * src_pred[src]
                X_pred_to_print[src] = X_pred_ar_scaled
        if majorityPrediction < 0:
            if c in pred_dict:
                pred_dict[c][s] = (float(negative_count[c][s]) / len(src_pred)) * 100
                pred_weight_dict[c][s] = majorityPrediction
            else:
                pred_dict[c] = {}
                pred_weight_dict[c] = {}
                pred_dict[c][s] = (float(negative_count[c][s]) / len(src_pred)) * 100
                pred_weight_dict[c][s] = majorityPrediction

for c in country_state_source_acc_dict:
    fNameCountry = "accOutCountry/" + c + "_acc.txt"
    fCountry = open(fNameCountry, 'w')
    for s in country_state_source_acc_dict[c]:
        fNameState = "accOutState/" + c + "_" + s + "_acc.txt"
        fState = open(fNameState, 'w')
        for src in country_state_source_acc_dict[c][s]:
            acc = country_state_source_acc_dict[c][s][src]["_correct"] / country_state_source_acc_dict[c][s][src]["_total"]
            newLine = src + "\t" + str(acc) + "\n"
            fState.write(newLine)
            fCountry.write(newLine)
        fState.close()
    fCountry.close()


for c in pred_dict:
    for s in pred_dict[c]:
        conf = 1.0
        acc_neg = 1.0
        for a in pred_acc_dict[c][s]["_neg"]:
            acc_neg *= 1 - a
            conf *= a
        acc_neg = 1 - acc_neg
        acc_pos = 1.0
        if len(pred_acc_dict[c][s]["_pos"]) == 0:
            acc_pos = "N/A"
        else:
            for a in pred_acc_dict[c][s]["_pos"]:
                conf *= 1 - a
                acc_pos *= 1 - a
            acc_pos = 1 - acc_pos
        newline = c + '\t' + s + '\t' + str(pred_dict[c][s]) + '\t' + str(acc_neg) + '\t' + str(acc_pos) + '\t' + str(conf) + '\n'
        out.write(newline)
out.close()

