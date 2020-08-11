import pandas as pd
import seaborn as sns

train = pd.read_csv('oversampled_data.csv')

## create x and y variables
y= train['Crop_Damage'].values.reshape(-1,1)
x= train.drop(labels = 'Crop_Damage', axis = 1)


## train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10, )


## modeling using DecisisonTree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

## fit decision tree model
dtree.fit(x_train, y_train)

## predict using fitted model
dtree_predictions = dtree.predict(x_test)


from sklearn.metrics import confusion_matrix, classification_report
dtree_confusion_matrix = confusion_matrix(y_test, dtree_predictions)

dtree_classification_report = classification_report(y_test, dtree_predictions)

