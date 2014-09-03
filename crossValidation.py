import sys
from featureUtils import FeatureUtils
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
from skLearnWrapper import SkLearnWrapper

def subsetList(data, idxs):
    return [data[idx] for idx in idxs]

class CrossValider:
    def __init__(self, clfName, numericFeaturePath, numericFeatureNamePath, cateFeaturePath, cateFeatureNamePath, textFeaturePath, textFeatureNamePath, labelPath, kfold):
        self.utils = FeatureUtils()
        self.numericFeature = self.utils.loadBinary(numericFeaturePath)
        self.numericFeatureName = self.utils.loadBinary(numericFeatureNamePath)
        self.cateFeature = self.utils.loadBinary(cateFeaturePath)
        self.cateFeatureName = self.utils.loadBinary(cateFeatureNamePath)
        self.textFeature = self.utils.loadBinary(textFeaturePath)
        self.textFeatureName = self.utils.loadBinary(textFeatureNamePath)
        self.labels = self.utils.loadBinary(labelPath)
        self.kfold = kfold
        self.clfName = clfName

    def crossValidation(self):
        precisionSum = 0.0
        recallSum = 0.0
        fscoreSum = 0.0

        skf = cross_validation.StratifiedKFold(self.labels, self.kfold)
        nRound = 0
        for trainIdx, testIdx in skf:
            sys.stderr.write('\n{}-fold cross validation: round={}\n'.format(self.kfold, nRound))
            nRound += 1
            trainLabels = subsetList(self.labels, trainIdx)
            trainTextFeature = subsetList(self.textFeature, trainIdx)
            trainCateFeature = subsetList(self.cateFeature, trainIdx)
            trainNumericFeature = subsetList(self.numericFeature, trainIdx)

            testLabels = subsetList(self.labels, testIdx)
            testTextFeature = subsetList(self.textFeature, testIdx)
            testCateFeature = subsetList(self.cateFeature, testIdx)
            testNumericFeature = subsetList(self.numericFeature, testIdx)

            learner = SkLearnWrapper(self.clfName, self.textFeatureName, self.cateFeatureName, self.numericFeatureName)

            sys.stderr.write('learner.fit()\n')
            learner.fit(trainLabels, trainTextFeature, trainCateFeature, trainNumericFeature)

            sys.stderr.write('learner.predict()\n')
            preds = learner.predict(testTextFeature, testCateFeature, testNumericFeature)
            precision, recall, fscore, support = precision_recall_fscore_support(testLabels, preds, labels=[0, 1], pos_label=1)
            precision, recall, fscore, support = precision[1], recall[1], fscore[1], support[1]
            sys.stderr.write('precision={} recall={} fscore={}\n'.format(precision, recall, fscore))

            precisionSum += precision
            recallSum += recall
            fscoreSum += fscore

        precisionMean = precisionSum / self.kfold
        recallMean = recallSum / self.kfold
        fscoreMean = fscoreSum / self.kfold
        sys.stderr.write('MEAN precision={} recall={} fscore={}\n'.format(precisionMean, recallMean, fscoreMean))
