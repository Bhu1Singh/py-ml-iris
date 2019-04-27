import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load the dataset into a pandas dataframe
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-len', 'sepal-wid', 'petal-len', 'petal-wid', 'class']
df = pd.read_csv(url, names=names)

# print data summary

# # print(df.head())

# # print(f'Shape of iris data: {df.shape}')

# print(df.describe())

# # print(df.groupby('class').size())

# box and whisker plot
# df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)

# df.hist()

# scatter_matrix(df)
# plt.show()


arr = df.values
x = arr[:, 0:4]
y = arr[:, 4]
validation_size = 0.20
seed = 7

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(
    x, y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each of the models
results = []
mnames = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_result = model_selection.cross_val_score(
        model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)

    mnames.append(name)
    msg = f'{name}: {cv_result.mean()} {cv_result.std()}'
    print(msg)

# compare alogrithms

# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# make predictions on validation dataset
print(x_validation)
print(y_validation)
knn = SVC(gamma='auto')
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)
print(accuracy_score(y_validation, predictions, normalize=False))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
