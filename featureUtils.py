from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import KFold
import csv
import itertools
import cPickle as pickle

class FeatureUtils:
    ACTION_TEXT = "action_text"
    ACTION_CATEGORY = "action_category"

    def __init__(self):
        pass

    def text2SparseFeature(self, column):
        vectorizer = TfidfVectorizer(min_df=1)
        sparseFeature = vectorizer.fit_transform(column)
        return sparseFeature

    def _text2Feature(self, column):
        vectorizer = TfidfVectorizer(min_df=1)
        sparseFeature = vectorizer.fit_transform(column)
        feature = sparseFeature.todense().tolist()
        return feature

    def _category2Feature(self, column):
        #column: [val1, val2, ..., valn], where val_i is the value of a specific feature (column)
        labelEncoder = LabelEncoder()
        column = labelEncoder.fit_transform(column).tolist()

        #the input of one hot encoder is [[val1], [val2], ..., [valn]]
        oneHotEncoder = OneHotEncoder()
        column = [[x] for x in column]
        #the output of oneHotEncoder is [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        column = oneHotEncoder.fit_transform(column).todense().tolist()
        return column

    def rows2Feature(self, rows, action):
        #rows: [[val1, val2, ..., valn], [val1, val2, ..., valn]], where val_i is the value of the i-th feature
        feature = []
        if rows is None or len(rows) == 0:
            return rows
        nColumn = len(rows[0])
        for i in range(nColumn):
            column = [x[i] for x in rows]
            if action == self.ACTION_CATEGORY:
                column = self._category2Feature(column)
            elif action == self.ACTION_TEXT:
                column = self._text2Feature(column)

            if len(feature) == 0:
                feature = column
            else:
                for (x, y) in zip(feature, column):
                    x.extend(y)
        return feature

    def category2Feature(self, rows):
        return self.rows2Feature(rows, self.ACTION_CATEGORY)

    def text2Feature(self, rows):
        return self.rows2Feature(rows, self.ACTION_TEXT)

    def dumpFeature(self, outputPath, rows):
        with open(outputPath, 'wb') as featureFile:
            if outputPath.endswith('.csv'):
                featureWriter = csv.writer(featureFile, lineterminator = '\n')
                for row in rows:
                    featureWriter.writerow(row)
            else:
                pickle.dump(rows, featureFile)

    #def dumpTxt(self, outputPath, rows):
    #    with open(outputPath, 'wb') as f:
    #        for row in rows:
    #            f.write(row + '\n')

    #def loadTxt(self, inputPath):
    #    with open(inputPath 'rb') as f:
    #        return f.readlines()

    def dumpBinary(self, outputPath, obj):
       with open(outputPath, 'wb') as f:
           pickle.dump(obj, f)

    def loadBinary(self, inputPath):
        with open(inputPath, 'rb') as f:
            return pickle.load(f)

    def loadCSV(self, inputPath):
        with open(inputPath, 'rb') as f:
            reader = csv.reader(f)
            return [row for row in reader]

    def dumpCSV(self, outputPath, rows):
        with open(outputPath, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def loadFeature(self, inputPath, labelIdx):
        comments = []
        labels = []
        feature = []
        with open(inputPath, 'rb') as featureFile:
            if inputPath.endswith('.csv'):
                featureReader = csv.reader(featureFile, lineterminator = '\n')
            else:
                featureReader = pickle.load(featureFile)
            for row in featureReader:
                comments.append(row[0 : labelIdx])
                labels.append(int(row[labelIdx]))
                feature.append(map(float, row[labelIdx + 1 : len(row)]))
        return (comments, labels, feature)

    def loadReadableFeature(self, inputPath, labelIdx):
        comments = []
        labels = []
        feature = []
        header = ""
        with open(inputPath, 'rb') as featureFile:
            featureReader = csv.reader(featureFile, lineterminator = '\n')
            header = featureReader.next()
            for row in featureReader:
                comments.append(row[0 : labelIdx])
                labels.append(int(row[labelIdx]))
                feature.append(row[labelIdx + 1 : len(row)])
        return (comments, labels, feature, header)

    #def keyIdx2FeatureIdx(self, trainKeys, testKeys, keys):
    #    trainKeys, testKeys = set(trainKeys), set(testKeys)
    #    trainIdx = [i for i in range(len(keys)) if keys[i] in trainKeys]
    #    testIdx = [i for i in range(len(keys)) if keys[i] in testKeys]
    #    return trainIdx, testIdx

    def getIdxsByGroup(self, keys):
        keyIdxPairs = zip(keys, range(len(keys)))
        key2Idxs = {}
        for key, idx in keyIdxPairs:
            if key not in key2Idxs:
                key2Idxs[key] = [idx]
            else:
                key2Idxs[key].append(idx)
        return key2Idxs

    def mergeIdxByKey(self, key2Idxs, keys):
        idxs = []
        for key in keys:
            idxs.extend(key2Idxs[key])
        return idxs

    def splitTrainTest(self, keys, percent):
        key2Idxs = self.getIdxsByGroup(keys)
        uniqKeys = list(set(keys))
        trainKeys, testKeys = train_test_split(uniqKeys, train_size = percent)
        trainKeys = set(trainKeys)
        testKeys = set(testKeys)

        trainIdx = self.mergeIdxByKey(key2Idxs, trainKeys)
        testIdx = self.mergeIdxByKey(key2Idxs, testKeys)
        return trainIdx, testIdx

    def kFold(self, keys, k):
        key2Idxs = self.getIdxsByGroup(keys)
        uniqKeys = list(set(keys))
        nKey = len(uniqKeys)
        kf = KFold(nKey, k)

        trainTestIdxPairs = []
        for trainKeyIdxs, testKeyIdxs in kf:
            trainKeys = [uniqKeys[i] for i in trainKeyIdxs]
            testKeys = [uniqKeys[i] for i in testKeyIdxs]

            trainIdx = self.mergeIdxByKey(key2Idxs, trainKeys)
            testIdx = self.mergeIdxByKey(key2Idxs, testKeys)

            trainTestIdxPairs.append((trainIdx, testIdx))
        return trainTestIdxPairs

    def calcPRFBinary(self, keys, labels, preds):
        meanPrecision = 0
        meanRecall = 0
        meanFscore = 0
        for key, idxs in itertools.groupby(range(len(keys)), lambda x: keys[x]):
            idxs = list(idxs)
            subLabels = [labels[idx] for idx in idxs]
            subPreds = [preds[idx] for idx in idxs]
            (precision, recall, fscore, support) = precision_recall_fscore_support(subLabels, subPreds, labels = [0, 1], pos_label = 1)
            meanPrecision += precision[1]
            meanRecall += recall[1]
            meanFscore += fscore[1]

        nKey = len(set(keys))
        meanPrecision /= float(nKey)
        meanRecall /= float(nKey)
        meanFscore /= float(nKey)

        return (meanPrecision, meanRecall, meanFscore)

    def calcPRFRank(self, keys, labels, preds, k):
        meanPrecision = 0
        meanRecall = 0
        meanFscore = 0
        count = 0
        for key, idxs in itertools.groupby(range(len(keys)), lambda x: keys[x]):
            idxs = list(idxs)
            nCase = len(idxs)
            if nCase < k:
                continue
            count += 1
            subLabels = [labels[idx] for idx in idxs]
            subPreds = [preds[idx] for idx in idxs]
            subLabelsPreds = zip(subLabels, subPreds)
            subLabelsPreds = sorted(subLabelsPreds, key = lambda x: x[1], reverse=True)
            subLabels = [x[0] for x in subLabelsPreds]
            subPreds = [x[1] for x in subLabelsPreds]

            assert len(subLabels) == len(subPreds) and nCase == len(subPreds)
            subPreds = [1 if i < k else 0 for i in range(nCase)]
            assert len(subLabels) == len(subPreds) and nCase == len(subPreds)

            (precision, recall, fscore, support) = precision_recall_fscore_support(subLabels, subPreds, labels = [0, 1], pos_label = 1)
            meanPrecision += precision[1]
            meanRecall += recall[1]
            meanFscore += fscore[1]

        meanPrecision /= float(count)
        meanRecall /= float(count)
        meanFscore /= float(count)
        return (meanPrecision, meanRecall, meanFscore)

    def outputCases(self, labels, preds, threshold, testIdxs, feature, outputPath, header):
        header = ['label', 'pred'] + header
        with open(outputPath, 'wb') as outputFile:
            writer = csv.writer(outputFile, lineterminator='\n')
            writer.writerow(header)
            for label, pred, testIdx in zip(labels, preds, testIdxs):
                if label - pred > threshold:
                    writer.writerow([label, pred] + feature[testIdx])
