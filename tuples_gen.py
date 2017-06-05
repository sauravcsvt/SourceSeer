#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"


from embers import geocode, utils
import re
from urllib2 import urlparse, urlopen
from dateutil.parser import parse as dtparse
import webarticle2text
import nltk
from collections import defaultdict

locationcls = geocode.Geo()
wordlemmatizer = nltk.WordNetLemmatizer()


class fourD_tuples(object):
    '''
    This class is responsible for creating the 4D matrix {source, location, word, timepoint; count}
    given a set of articles
    '''

    def __init__(self, json_data, timestamp, stop_words, start_ind):
        self.artl_json = json_data
        self.timestamp = timestamp
        self.stop_words = stop_words
        self.country_list = ["Argentina", "Brazil", "Chile", "Paraguay", "Ecuador", 
                             "Colombia", "El Salvador", "Mexico", "Uruguay", "Venezuela"]
        self.start_ind = start_ind
        self.final_dict = defaultdict(int)

    def contains_digits(self, word):

        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        count_digit = 0
        for element in word:
            if element in digits:
                count_digit += 1
        if count_digit == len(word):
            return True
        else:
            return False

    def return_unique_ids(self):
        ids = []
        for artl in self.artl_json:
            try:
                ids.append(artl['embersId'])
            except Exception:
                ids.append(artl['embers_id'])
        ids = list(set(ids))
        return ids

    def final_tuples(self):

        id_list = self.return_unique_ids()
        for artl in self.artl_json:
            if "embersId" in artl:
                artl_id = artl["embersId"]
            else:
                artl_id = artl["embers_id"]
            if artl_id in id_list:
                id_list.remove(artl_id)
            else:
                continue

            if artl[u'location'][u'country'] not in self.country_list:
                continue
            latitude_num = float(artl[u'location'][u'lat'])
            try:
                longitude_num = float(artl[u'location'][u'lng'])
            except Exception:
                longitude_num = float(artl[u'location'][u'lon'])
            artl_dt = dtparse(artl['date']).date()
            if artl_dt < self.timestamp[0] or artl_dt >= self.timestamp[len(self.timestamp) - 1]:
                continue
            try:
                finalURL = (urlopen(artl['link'])).geturl()
                article_source = urlparse.urlparse(finalURL).netloc
                articleprovince = list(locationcls.lookup_city(latitude_num, longitude_num, 360.)[0])[2]
                articlecountry = list(locationcls.lookup_city(latitude_num, longitude_num, 360.)[0])[1]
            except Exception:
                continue
            if 'BasisEnrichment' not in artl:
                try:
                    content_web = webarticle2text.extractFromURL(finalURL)
                except Exception:
                    content_web = ""
                content_descr = artl['descr']
                tokens = nltk.word_tokenize(content_descr)
                try:
                    tokens_1 = nltk.word_tokenize(content_web)
                    for word in tokens_1:
                        tokens.append(word)
                except Exception:
                    tokens_1 = []
            else:
                POS_list = ["DIG", "PUNCT", "SYM", "SENT", "CM"]
                if not(not(artl['BasisEnrichment']['tokens'])):
                    tokenlist = artl['BasisEnrichment']['tokens']
                for element in tokenlist:
                    if element['POS'] not in POS_list:
                        tokens.append(element['value'])
            token_filtered = []
            token_normalized = []
            for a in xrange(len(self.timestamp) - 1):
                if self.timestamp[a] <= artl_dt < self.timestamp[a + 1]:
                    timestampindex = self.start_ind + a
                    break
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
                try:
                    if not self.contains_digits(word) and word not in self.stop_words:
                        token_normalized.append(utils.normalize_str(word))
                except Exception:
                    continue
            token_unique = list(set(token_normalized))
            for word in token_unique:
                self.final_dict[(word, (articleprovince, articlecountry), article_source, timestampindex)] += token_normalized.count(word)
        return 

