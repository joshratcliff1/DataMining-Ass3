import graphviz
import pandas as pd
from statistics import mean
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

''' https://stackabuse.com/decision-trees-in-python-with-scikit-learn/ '''


# show complete records by changing rules
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset into the dataframe
df = pd.read_csv("8410_processed_sklearn_gender.csv")

# split into attributes and labels
X = df.drop(['srcid','gender'], axis=1)
X = X.drop(['Q6b','Q6c','Q6d','Q7b','Q7c','Q7d','Q10b','Q10c','Q10d'], axis=1)
y = df['gender']

'''srcid,Q1,Q2,Q4,Q6a,Q6b,Q6c,Q6d,Q7a,Q7b,Q7c,Q7d,Q7e,Q10a,Q10b,Q10c,Q10d,sDevType,sOSName,gender'''
# print(X)

# Convert features to Ordinal values
ordinalencoder_X = OrdinalEncoder()
X = ordinalencoder_X.fit_transform(X)
X = X.astype(int)

# Convert target to Ordinal values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# # Split the data into test and train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
# # Initiate the classifier
# dt = DecisionTreeClassifier()
#
# # Train the classifier
# dt.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = dt.predict(X_test)

# # Output confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))



# Plot the tree
# plot_tree(classifier)


dt_list = [DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.4),
           DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.3),
           DecisionTreeClassifier(criterion='entropy',max_depth=4, min_samples_leaf=0.4),
           DecisionTreeClassifier(criterion='entropy', max_depth=4)]

result_list = [[] for x in range(len(dt_list))]

for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    for i in range(len(dt_list)):
        # choose the decision tree parameters
        dt = dt_list[i]

        # Train the classifier
        dt.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = dt.predict(X_test)

        # Output confusion matrix and classification report
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average='weighted')
        # print(f1)
        result_list[i].append(f1)

# print(result_list)

first = mean(result_list[0])
second = mean(result_list[1])
third = mean(result_list[2])
fourth = mean(result_list[3])

print("average of first parameters = ",first)
print("average of second parameters = ",second)
print("average of third parameters = ",third)
print("average of fourth parameters = ",fourth)

# # graphing options
# dot_data = tree.export_graphviz(dt, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("gender")