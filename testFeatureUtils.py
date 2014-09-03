from featureUtils import FeatureUtils
import simplejson as json
import csv
import sys

def main():
    utils = FeatureUtils()

    feature = ["hello world", "hello kitty"]
    feature = utils._text2Feature(feature)
    print feature

    feature = [1, 2, 1]
    feature = utils._category2Feature(feature)
    print feature

    feature = ["a", "b", "a"]
    feature = utils._category2Feature(feature)
    print feature

    #first column: ["hello world", "hello kitty"]
    #sec column: ["goodbye", "tsinghua"]
    feature = [["hello world", "goodbye"], ["hello kitty", "tsinghua"]]
    feature = utils.text2Feature(feature)
    print feature

    #first column [1, 2, 1]
    #sec column ["a", "b", "c"]
    feature = [[1, "a"], [2, "b"], [1, "c"]]
    feature = utils.category2Feature(feature)
    print feature

    #dump feature
    testPath = 'testDumpFeature.csv'
    feature = [[1, 2, 3, 4], [5, 6, 7, 8]]
    utils.dumpFeature(testPath, feature)

    #load feature
    keys, labels, feature = utils.loadFeature(testPath, 2)
    print "keys={}  labels={}  feature={}".format(keys, labels, feature)
    assert keys == [['1', '2'], ['5', '6']] and labels == [3, 7] and feature == [[4.0], [8.0]]


    #dump feature
    testPath = 'testDumpFeature.data'
    feature = [[1, 2, 3, 4], [5, 6, 7, 8]]
    utils.dumpFeature(testPath, feature)

    #load feature
    keys, labels, feature = utils.loadFeature(testPath, 2)
    assert keys == [[1, 2], [5, 6]] and labels == [3, 7] and feature == [[4.0], [8.0]]

    #split train test
    keys = ['1', '1', '2', '2', '3', '3', '3']
    trainIdx, testIdx = utils.splitTrainTest(keys, 0.67)
    print "trainIdx=" + str(trainIdx)
    print "train keys=" + str([keys[i] for i in trainIdx])
    print "testIdx=" + str(testIdx)
    print "test keys=" + str([keys[i] for i in testIdx])

    #test k-fold
    print "\ntest k-fold******"
    keys = ['1', '1', '2', '2', '3', '3', '3']
    trainTestIdxPairs = utils.kFold(keys, 3)
    for trainIdx, testIdx in trainTestIdxPairs:
        print "trainIdx=" + str(trainIdx)
        print "train keys=" + str([keys[i] for i in trainIdx])
        print "testIdx=" + str(testIdx)
        print "test keys=" + str([keys[i] for i in testIdx])
        print "\n"

    #calculate precision, recall, fscore
    keys = [1, 1]
    labels = [0, 0]
    preds = [1, 0]
    precision, recall, fscore = utils.calcPRFBinary(keys, labels, preds)
    print "precision={}   recal={}   fscore={}".format(precision, recall, fscore)

    #calculate precision, recall, fscore
    keys = [1, 1, 1]
    labels = [1, 0, 1]
    preds = [1, 1, 0]
    precision, recall, fscore = utils.calcPRFBinary(keys, labels, preds)
    print "precision={}   recal={}   fscore={}".format(precision, recall, fscore)
    assert precision == 0.5 and recall == 0.5

    keys = [2, 2]
    labels = [1, 0]
    preds = [1, 1]
    precision, recall, fscore = utils.calcPRFBinary(keys, labels, preds)
    assert precision == 0.5 and recall == 1.0
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)

    keys = [1, 1, 1, 2, 2]
    labels = [1, 0, 1, 1, 0]
    preds = [1, 1, 0, 1, 1]
    precision, recall, fscore = utils.calcPRFBinary(keys, labels, preds)
    assert precision == 0.5 and recall == 0.75

    keys = [1, 1]
    labels = [1, 1]
    preds = [0.3, 0.7]
    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 1)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 1.0 and recall == 0.5

    keys = [1, 1]
    labels = [1, 0]
    preds = [0.3, 0.7]
    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 2)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 0.5 and recall == 1.0

    keys = [2, 2, 2]
    labels = [1, 0, 1]
    preds = [0.7, 0.3, 0.5]
    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 2)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 1.0 and recall == 1.0

    keys = [2, 2, 2]
    labels = [1, 0, 1]
    preds = [0.7, 0.3, 0.5]
    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 3)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 2.0/3 and recall == 1.0

    keys = [1, 1, 2, 2, 2]
    labels = [1, 0, 1, 0, 1]
    preds = [0.3, 0.7, 0.7, 0.3, 0.5]
    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 2)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 0.75 and recall == 1.0

    precision, recall, fscore = utils.calcPRFRank(keys, labels, preds, 3)
    print "precision={}   recall={}    fscore={}".format(precision, recall, fscore)
    assert precision == 2.0/3 and recall == 1.0





    print "Test Successed!"

if __name__ == '__main__':
    main()
