import pandas as pd
import sklearn.tree
import sklearn.ensemble
import sklearn.neighbors
import sklearn.svm
print(pd.__version__)
print(sklearn.__version__)

def save_results(dataset, resultdataset, type_name='dtc'):
    result = pd.DataFrame()
    result['PassengerId'] = dataset['PassengerId']
    result['Survived'] = pd.Series(resultdataset).astype(int)
    result.to_csv('result{}.csv'.format(type_name), index=False)

def bin_age(age_val):
    if age_val <= 20:
        return 0
    elif age_val <= 40:
        return 1
    elif age_val <= 60:
        return 2
    elif age_val <= 80:
        return 3
    else:
        return 1

def preprocess_dataset(dataset, training = False):
    features = ['Pclass', 'Sex', 'Age', 'Embarked']
    X = dataset[features]
    X['Embarked'] = pd.Series((0 if embark_val == 'C' else (1 if embark_val == 'Q' else 2)) for embark_val in dataset['Embarked'])
    X['Sex'] = pd.Series((1 if sex_val == 'male' else 0) for sex_val in dataset['Sex'])
    X['Age'] = pd.Series((bin_age(age_val)) for age_val in dataset['Age'])
    X['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # print(X.describe())
    if training:
        y = dataset['Survived']
        return X, y
    return X

def calc_accuracy(dataset, result, X):
    n = len(dataset)
    correct = 0
    for i in range(n):
        if result[i] == dataset['Survived'][i]:
            correct += 1
    return correct/n

if __name__ == "__main__": 
    titanic_train_data = pd.read_csv('./train.csv')
    titanic_train_data.describe()
    titanic_train_data.columns

    titanic_test_data = pd.read_csv('./test.csv')
    titanic_test_data.describe()
    titanic_test_data.columns

    decision_tree_c = sklearn.tree.DecisionTreeClassifier()
    random_forest = sklearn.ensemble.RandomForestClassifier()
    grad_boost = sklearn.ensemble.GradientBoostingClassifier()
    k_n = sklearn.neighbors.KNeighborsClassifier(2)
    svc = sklearn.svm.SVC()

    X, y = preprocess_dataset(titanic_train_data, True)
        
    decision_tree_c.fit(X, y)
    random_forest.fit(X, y)
    k_n.fit(X, y)
    svc.fit(X, y)
    grad_boost.fit(X, y)

    res_train_c = decision_tree_c.predict(X)
    res_train = random_forest.predict(X)
    res_train_svc = svc.predict(X)
    res_train_kn = k_n.predict(X)
    res_train_gb = grad_boost.predict(X)

    accuracy_c = calc_accuracy(titanic_train_data, res_train_c, X)
    accuracy = calc_accuracy(titanic_train_data, res_train, X)
    accuracy_svc = calc_accuracy(titanic_train_data, res_train_svc, X)
    accuracy_kn = calc_accuracy(titanic_train_data, res_train_kn, X)
    accuracy_gb = calc_accuracy(titanic_train_data, res_train_gb, X)

    print('DTC Accuracy: {}. RF Accuracy: {}. SVC Accuracy: {}. KN Accuracy: {}. GB Accuracy: {}'.format(accuracy_c, accuracy, accuracy_svc, accuracy_kn, accuracy_gb))
    save_results(titanic_train_data, res_train_c)
    save_results(titanic_train_data, res_train, '-rf')
    save_results(titanic_train_data, res_train_kn, '-kn')
    save_results(titanic_train_data, res_train_gb, '-gb')
    save_results(titanic_train_data, res_train_svc, '-svc')

    X_test = preprocess_dataset(titanic_test_data)
    res_rf = random_forest.predict(X_test)
    save_results(titanic_test_data, res_rf, 'rf')  
    res_dtc = decision_tree_c.predict(X_test)
    save_results(titanic_test_data, res_dtc)    
