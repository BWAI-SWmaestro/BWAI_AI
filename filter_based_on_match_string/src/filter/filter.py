import pymongo
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from filter.secret_info import MONGO_HOST, MONGO_ID, MONGO_PW, MONGO_DB_NAME


class Filter:
    def __init__(self):
        super(Filter, self).__init__()
        # self.db_client = pymongo.MongoClient('mongodb://%s:%s@%s' % (MONGO_ID, MONGO_PW, MONGO_HOST))
        # self.db = self.db_client['BWAI']
        # self.posts = self.db.posts
        # self.posts_labeled = self.db.posts_labeled_v2

        # set은 in operation time complexity average O(1)
        self.bad_words = set(np.loadtxt("bad_word_list/bad_word.csv", delimiter=',', dtype=np.unicode, encoding='utf-8'))
        self.data = pd.read_csv("badword/ratings_labeled_test.csv")

    def _check_bad_word(self, doc):
        doc['is_bad_word'] = []
        doc['label_v1'] = []
        for sentence in doc['post']:
            # 현재 문장을 단어로 나누기
            label = []
            doc['is_bad_word'].append(False)
            for word in sentence.split():
                word_label = ['0']*len(word)
                for bad_word in self.bad_words:
                    bad_word_idx = word.find(bad_word)
                    # 문장의 단어가 욕설 리스트에 포함되어 있으면
                    if bad_word_idx != -1:
                        doc['is_bad_word'][-1] = True
                        word_label[bad_word_idx:bad_word_idx+len(bad_word)] = ['1']*len(bad_word)
                label.append("".join(word_label))
            doc['label_v1'].append(' '.join(label))

        label = []
        doc['title_is_bad_word'] = False
        for word in doc['title'].split():
            word_label = ['0']*len(word)
            for bad_word in self.bad_words:
                bad_word_idx = word.find(bad_word)
                if bad_word_idx != -1:
                    doc['title_is_bad_word'] = True
                    word_label[bad_word_idx:bad_word_idx+len(bad_word)] = ['1']*len(bad_word)
            label.append("".join(word_label))
        doc['title_label_v1'] = " ".join(label)

        # self.posts_labeled.insert_one(doc)

        result = doc['is_bad_word']+[doc['title_is_bad_word']]
        return len(result), sum(result)

    def run_filtering(self):
        sentence_count, bad_word_count = 0, 0
        cursor = self.posts.find({})
        for i, document in enumerate(cursor):
            cnt1, cnt2 = self._check_bad_word(document)
            sentence_count += cnt1
            bad_word_count += cnt2
            if i % 10000 == 0:
                print("{} posts\t{} sentences\t{} bad word".format(i, sentence_count, bad_word_count))
        print("{} posts\t{} sentences\t{} bad word".format(i, sentence_count, bad_word_count))

    def run_filtering_only(self):
        count, pred = 0, []
        for i, row in self.data.iterrows():
            is_bad_word = False
            for bad_word in self.bad_words:
                if bad_word in row['document']:
                    count += 1
                    is_bad_word = True
                    break
            if is_bad_word:
                pred.append(1)
            else:
                pred.append(0)
            if i % 1000 == 0:
                print("{} sentences\t{} bad word".format(i, count))

        metrics = [
            metric(y_true=self.data['label'], y_pred=pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        print("{} sentences\t{} bad word\naccuracy: {}\nprecision: {}\nrecall: {}\nf1_score: {}".format(len(self.data), count, *metrics))
