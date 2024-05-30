# -*- encoding:utf-8 -*-
"""
@author:   letian
@homepage: http://www.letiantian.me
@github:   https://github.com/someus/
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import networkx as nx
import numpy as np

from . import util
from .Segmentation import Segmentation

class TextRank4Sentence(object):
    
    def __init__(self, stop_words_file=None, 
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)
        
        self.sentences = None
        self.words_no_filter = None     # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None
        
        self.key_sentences = None
        
    def analyze(self, text, lower=False, 
                source='no_stop_words', 
                sim_func=util.get_similarity,
                pagerank_config={'alpha': 0.85,},
                tokenizer=None):
        self.key_sentences = []
        
        if tokenizer:
            result = tokenizer(text)
            self.sentences = result['sentences']
            self.words_no_filter = result['words_no_filter']
            self.words_no_stop_words = result['words_no_stop_words']
            self.words_all_filters = result['words_all_filters']
        else:
            result = self.seg.segment(text=text, lower=lower)
            self.sentences = result.sentences
            self.words_no_filter = result.words_no_filter
            self.words_no_stop_words = result.words_no_stop_words
            self.words_all_filters = result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']
        if source in options:
            _source = result['words_'+source]
        else:
            _source = result['words_no_stop_words']

        self.key_sentences = util.sort_sentences(sentences=self.sentences,
                                                 words=_source,
                                                 sim_func=sim_func,
                                                 pagerank_config=pagerank_config)

    def get_key_sentences(self, num=6, sentence_min_len=6):
        result = []
        count = 0
        for item in self.key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result

if __name__ == '__main__':
    pass
