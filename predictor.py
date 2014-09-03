from featureUtils import FeatureUtils

class Predictor:
    def __init__(self, numericFeaturePath, cateFeaturePath, textFeaturePath, modelPath):
        self.utils = FeatureUtils()
        self.numericFeature = self.utils.loadBinary(numericFeaturePath)
        self.cateFeature = self.utils.loadBinary(cateFeaturePath)
        self.textFeature = self.utils.loadBinary(textFeaturePath)
        self.modelPath = modelPath

    def predict(self):
        learner = self.utils.loadBinary(self.modelPath)
        preds = learner.predict(self.textFeature, self.cateFeature, self.numericFeature)
        return preds
