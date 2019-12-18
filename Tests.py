
import unittest

import pandas as pd


class TestUM(unittest.TestCase):
    bert_data =[]
    w2v_data = []

    def setUp(self):
        self.bert_data = pd.read_csv('data/preprocessedTweetsBERT.csv', sep=',', header=0, encoding='latin-1')
        self.w2v_data = pd.read_csv('data/preprocessedTweetsW2Vec.csv', sep=',', header=0, encoding='latin-1')


    def test_for_Links(self):
        self.assertTrue(not self.bert_data['text'].str.contains(r' www.\S+').any())
        self.assertTrue(not self.w2v_data['text'].str.contains(r' www.\S+').any())
        self.assertTrue(not self.bert_data['text'].str.contains(r' http\S+').any())
        self.assertTrue(not self.w2v_data['text'].str.contains(r' http\S+').any())


    def test_for_twitterHandles(self):
        self.assertTrue(not self.bert_data['text'].str.contains(r'@[^\s]+').any())
        self.assertTrue(not self.w2v_data['text'].str.contains(r'@[^\s]+').any())

    def test_HashTags(self):
        self.assertTrue(not self.bert_data['text'].str.contains('#').any())
        self.assertTrue(not self.w2v_data['text'].str.contains('#').any())

    def test_for_doubleSpaces(self):
        self.assertTrue(not self.bert_data['text'].str.contains('  ').any())
        self.assertTrue(not self.w2v_data['text'].str.contains('  ').any())

    def test_for_smileys(self):
        self.assertTrue(not self.bert_data['text'].str.contains(r':\)|=\)|:\(|:/|=\(|:D|:P').any())
        self.assertTrue(not self.w2v_data['text'].str.contains(r':\)|=\)|:\(|:/|=\(|:D|:P').any())


if __name__ == '__main__':
    unittest.main()