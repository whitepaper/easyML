import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_selection import VarianceThreshold
from django.utils.encoding import smart_str


class LabelEncoder():
    def __init__(self):
        pass

    def fit_transform(self, column):
        column = column.tolist()
        labelSet = set(column)
        self.labelDict = {label : (i + 1) for i, label in enumerate(labelSet)}
        column = [self.labelDict[label] for label in column]
        return np.array(column)

    def transform(self, column):
        column = column.tolist()
        column = [self.labelDict[label] if label in self.labelDict else 0 for label in column]
        return np.array(column)

    def getLabelSize(self):
        return len(self.labelDict)


class DummyEncoder():
    def __init__(self):
        pass

    def fit_transform(self, featureName, feature):
        if len(feature) == 0 or len(feature[0]) == 0:
            return [], np.array(feature)

        nCol = len(feature[0])
        self.nCol = nCol
        feature = np.array(feature)

        self.labelEncoders = [LabelEncoder() for i in range(nCol)]

        for idx in range(nCol):
            feature[:, idx] = self.labelEncoders[idx].fit_transform(feature[:, idx])

        nLabels = [labelEncoder.getLabelSize() + 1 for labelEncoder in self.labelEncoders]
        self.oneHotEncoder = OneHotEncoder(n_values = nLabels)

        feature = self.oneHotEncoder.fit_transform(feature)

        featureIndices = self.oneHotEncoder.feature_indices_

        newName = ['unknown'] * feature.shape[1]
        for i in range(nCol):
            startIdx = featureIndices[i]
            endIdx = featureIndices[i + 1] - 1
            for idx in range(startIdx, endIdx + 1):
                newName[idx] = featureName[i]
        return newName, feature

    def transform(self, feature):
        if len(feature) == 0 or len(feature[0]) == 0:
            return np.array(feature)

        nCol = len(feature[0])
        assert nCol == self.nCol, 'column size does not match'

        feature = np.array(feature)

        for idx in range(nCol):
            feature[:, idx] = self.labelEncoders[idx].transform(feature[:, idx])

        feature = self.oneHotEncoder.transform(feature)
        return feature


class TextEncoder():
    def __init__(self):
        self.vects = []

    def fit_transform(self, featureName, feature):
        if len(feature) == 0 or len(feature[0]) == 0:
            return [], csr_matrix(feature)

        self.nCol = len(feature[0])
        self.vects = [TfidfVectorizer(min_df=2) for i in range(self.nCol)]
        sparseMat = None

        newName = []
        for idx in range(self.nCol):
            column = [x[idx] for x in feature]
            columnName = featureName[idx]
            vect = self.vects[idx]
            docMat = vect.fit_transform(column)
            words = vect.get_feature_names()
            newName.extend([columnName + ':' + word for word in words])
            if sparseMat is None:
                sparseMat = docMat
            else:
                sparseMat = hstack([sparseMat, docMat])
        return newName, sparseMat

    def transform(self, feature):
        assert self.nCol == len(feature[0]), "column size does not match"
        if len(feature) == 0 or len(feature[0]) == 0:
            return csr_matrix(feature)

        sparseMat = None
        for idx in range(self.nCol):
            column = [x[idx] for x in feature]
            vect = self.vects[idx]
            if sparseMat is None:
                sparseMat = vect.transform(column)
            else:
                sparseMat = hstack([sparseMat, vect.transform(column)])
        return sparseMat


class SkLearnWrapper:
    CLF_SGD = 'SGDClassifier'
    CLF_LSVC = 'LinearSVC'

    def __init__(self, clfName, textFeatureName, cateFeatureName, numericFeatureName):
        self.textEncoder = TextEncoder()
        self.dummyEncoder = DummyEncoder()
        self.scaler = StandardScaler()
        self.varSelector = VarianceThreshold()

        self.textFeatureName = textFeatureName
        self.cateFeatureName = cateFeatureName
        self.numericFeatureName = numericFeatureName

        if clfName == SkLearnWrapper.CLF_SGD:
            self.clf = SGDClassifier(alpha=.0001, n_iter=50)
        elif clfName == SkLearnWrapper.CLF_LSVC:
            self.clf = LinearSVC(loss='l2', dual=False, tol=1e-3)

    def fitTransFeature(self, textFeature, cateFeature, numericFeature):
        sys.stderr.write("fit_transform text feature...")
        self.textFeatureName, textFeature = self.textEncoder.fit_transform(self.textFeatureName, textFeature)
        sys.stderr.write("categorical feature...")
        self.cateFeatureName, cateFeature = self.dummyEncoder.fit_transform(self.cateFeatureName, cateFeature)
        sys.stderr.write("numeric feature...")
        numericFeature = self.scaler.fit_transform(numericFeature)

        sys.stderr.write("hstack all feature\n")
        numericFeature = csr_matrix(numericFeature)
        cateFeature = csr_matrix(cateFeature)
        feature = [x for x in [numericFeature, cateFeature, textFeature] if x.shape[1] > 0]
        feature = hstack(feature)
        self.featureName = self.numericFeatureName + self.cateFeatureName + self.textFeatureName
        return feature

    def transFeature(self, textFeature, cateFeature, numericFeature):
        sys.stderr.write("transform text feature...")
        textFeature = self.textEncoder.transform(textFeature)
        sys.stderr.write("categorical feature...")
        cateFeature = self.dummyEncoder.transform(cateFeature)
        sys.stderr.write("numeric feature...")
        numericFeature = self.scaler.transform(numericFeature)

        sys.stderr.write("hstack all feature\n")
        numericFeature = csr_matrix(numericFeature)
        cateFeature = csr_matrix(cateFeature)
        feature = [x for x in [numericFeature, cateFeature, textFeature] if x.shape[1] > 0]
        feature = hstack(feature)
        return feature

    def selectFitTransFeature(self, feature):
        sys.stderr.write('SELECT fit_transform feature\n')
        feature = self.varSelector.fit_transform(feature)
        featureMasks = self.varSelector.get_support()
        assert len(self.featureName) == len(featureMasks)
        self.featureName = [name for (name, isTrue) in zip(self.featureName, featureMasks) if isTrue]
        return feature

    def selectTransFeature(self, feature):
        sys.stderr.write('SELECT transform feature\n')
        feature = self.varSelector.transform(feature)
        return feature

    def fit(self, labels, textFeature, cateFeature, numericFeature):
        # feature trasformation
        feature = self.fitTransFeature(textFeature, cateFeature, numericFeature)

        # feature selection
        feature = self.selectFitTransFeature(feature)

        # train models
        self.clf.fit(feature, labels)

    def predict(self, textFeature, cateFeature, numericFeature):
        # feature trasformation
        feature = self.transFeature(textFeature, cateFeature, numericFeature)

        # feature selection
        feature = self.selectTransFeature(feature)

        # prediction
        preds = self.clf.predict(feature)
        return preds

    def getFeatureWeight(self):
        if hasattr(self.clf, 'coef_'):
            weights = self.clf.coef_.tolist()[0]
            assert len(self.featureName) == len(weights)
            self.featureName = [smart_str(name) for name in self.featureName]
            nameWeights = zip(self.featureName, weights)
            nameWeights = sorted(nameWeights, key = lambda nameWeight : -nameWeight[1])
            return nameWeights
