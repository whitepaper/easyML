from skLearnWrapper import DummyEncoder
from skLearnWrapper import TextEncoder
from skLearnWrapper import SkLearnWrapper

if __name__ == '__main__':
    # test DummyEncoder
    dummyEncoder = DummyEncoder()

    featureName = ['cate1', 'cate2']
    feature = [[1, 'a'], [2, 'b'], [3, 'a']]
    featureName, feature = dummyEncoder.fit_transform(featureName, feature)
    feature = feature.todense().tolist()
    assert featureName == ['cate1'] * 4 + ['cate2'] * 3
    assert feature == [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]

    feature = [[3, 'b'], [1, 'a']]
    feature = dummyEncoder.transform(feature).todense().tolist()
    assert feature == [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]

    # test TextEncoder
    textEncoder = TextEncoder()
    featureName = ['text1', 'text2']
    feature = [['Hello World', 'Hello Kitty'], ['Hello Kitty', 'Hello World'], ['Kitty World', 'Kitty World']]
    featureName, fitFeature = textEncoder.fit_transform(featureName, feature)
    fitFeature = fitFeature.todense().tolist()

    assert featureName == ['text1:hello', 'text1:kitty', 'text1:world', 'text2:hello', 'text2:kitty', 'text2:world']

    feature = [['Hello World', 'Hello Kitty']]
    transFeature = textEncoder.transform(feature).todense().tolist()
    assert fitFeature[0] == transFeature[0]

    # test SkLearnWrapper.fitTransFeature
    numericFeatureName = ['num1', 'num2']
    cateFeatureName = ['cate1', 'cate2']
    textFeatureName = ['text1', 'text2']
    learner = SkLearnWrapper(SkLearnWrapper.CLF_SGD, textFeatureName, cateFeatureName, numericFeatureName)
    cateFeature = [[1, 'a'], [2, 'b'], [3, 'a']]
    textFeature = [['Hello World', 'Hello Kitty'], ['Hello Kitty', 'Hello World'], ['Kitty World', 'Kitty World']]
    numericFeature = [[1, 2], [1, 4], [1, 3]]
    feature = learner.fitTransFeature(textFeature, cateFeature, numericFeature).todense()
    featureName = learner.featureName
    oldFeature = feature

    assert featureName == ['num1', 'num2', 'cate1', 'cate1', 'cate1', 'cate1', 'cate2', 'cate2', 'cate2', 'text1:hello', 'text1:kitty', 'text1:world', 'text2:hello', 'text2:kitty', 'text2:world']

    #check numeric feature
    assert feature[:, 0].tolist() == [[0.0], [0.0], [0.0]]
    assert (feature[:, 1].tolist())[2] == [0.0]

    #check cate feature
    assert feature[:, [2, 3, 4, 5]].tolist() == [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    assert feature[:, [6, 7, 8]].tolist() == [[0, 1, 0], [0, 0, 1], [0, 1, 0]]

    # test SkLearnWrapper.transFeature
    cateFeature = [[1, 'a'], [2, 'b'], [3, 'a']]
    textFeature = [['Hello World', 'Hello Kitty'], ['Hello Kitty', 'Hello World'], ['Kitty World', 'Kitty World']]
    numericFeature = [[1, 2], [1, 4], [1, 3]]
    feature = learner.transFeature(textFeature, cateFeature, numericFeature).todense()
    assert feature.tolist() == oldFeature.tolist()

    # test SkLearnWrapper.fit

    # test SkLearnWrapper.fitTransFeature
    numericFeatureName = ['num1', 'num2']
    cateFeatureName = ['cate1', 'cate2']
    textFeatureName = ['text1', 'text2']
    learner = SkLearnWrapper(SkLearnWrapper.CLF_SGD, textFeatureName, cateFeatureName, numericFeatureName)
    cateFeature = [[1, 'a'], [2, 'b'], [3, 'a']]
    textFeature = [['Hello World', 'Hello Kitty'], ['Hello Kitty', 'Hello World'], ['Kitty World', 'Kitty World']]
    numericFeature = [[1, 2], [1, 4], [1, 3]]
    labels = ['0', '1', '0']
    learner.fit(labels, textFeature, cateFeature, numericFeature)
    featureName = learner.featureName

    assert featureName == ['num2', 'cate1', 'cate1', 'cate1', 'cate2', 'cate2', 'text1:hello', 'text1:kitty', 'text1:world', 'text2:hello', 'text2:kitty', 'text2:world']

    # test SkLearnWrapper.getFeatureWeight
    featureWeight = learner.getFeatureWeight()
    print featureWeight

    print "Test Success! Congrats!"

