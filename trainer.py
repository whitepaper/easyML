from featureUtils import FeatureUtils
from skLearnWrapper import SkLearnWrapper

class Trainer:
    def __init__(self, clfName, numericFeaturePath, numericFeatureNamePath, cateFeaturePath, cateFeatureNamePath, textFeaturePath, textFeatureNamePath, labelPath, modelPath, featureWeightPath):
        self.clfName = clfName

        self.utils = FeatureUtils()
        self.numericFeature = self.utils.loadBinary(numericFeaturePath)
        self.cateFeature = self.utils.loadBinary(cateFeaturePath)
        self.textFeature = self.utils.loadBinary(textFeaturePath)
        self.numericFeatureName = self.utils.loadBinary(numericFeatureNamePath)
        self.cateFeatureName = self.utils.loadBinary(cateFeatureNamePath)
        self.textFeatureName = self.utils.loadBinary(textFeatureNamePath)
        self.labels = self.utils.loadBinary(labelPath)
        self.modelPath = modelPath
        self.featureWeightPath = featureWeightPath

    def train(self):
        learner = SkLearnWrapper(self.clfName, self.textFeatureName, self.cateFeatureName, self.numericFeatureName)
        learner.fit(self.labels, self.textFeature, self.cateFeature, self.numericFeature)
        featureWeight = learner.getFeatureWeight()
        self.utils.dumpBinary(self.modelPath, learner)
        self.utils.dumpCSV(self.featureWeightPath, featureWeight)
