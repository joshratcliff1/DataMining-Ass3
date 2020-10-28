import pandas as pd
from statistics import mean
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

''' https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn '''
''' For a three layer network with n input and m output neurons, the hidden layer would have sqrt(n∗m) neurons. '''

# Load the dataset into the dataframe
df = pd.read_csv("8410_processed_sklearn_gender.csv")

# split into attributes and labels
X = df.drop(['srcid','gender'], axis=1)
# X = X.drop(['Q6b','Q6c','Q6d','Q7b','Q7c','Q7d','Q10b','Q10c','Q10d'], axis=1)
# X = X.drop(['Q6c','Q6d','Q7b','Q7d','Q10c',], axis=1)
y = df['gender']
# print(X)

# Convert features to Ordinal values
ordinalencoder_X = OrdinalEncoder()
X = ordinalencoder_X.fit_transform(X)
X = X.astype(int)

# # Convert features to OneHotEncoding values
# one_hot_encoder_X = OneHotEncoder()
# X = one_hot_encoder_X.fit_transform(X).toarray()
# X = X.astype(int)
# print(X)


# Convert target to Ordinal values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



nn_list = [MLPClassifier(hidden_layer_sizes=(18), activation='logistic', solver='adam', max_iter=999),
           MLPClassifier(hidden_layer_sizes=(18), activation='relu', solver='adam', max_iter=999)]

result_list = [[] for x in range(len(nn_list))]

for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    for i in range(len(nn_list)):
        # choose the decision tree parameters
        nn = nn_list[i]

        # Train the classifier
        nn.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = nn.predict(X_test)

        # Output confusion matrix and classification report
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average='weighted')
        # print(f1)
        result_list[i].append(f1)

# print(result_list)

first = mean(result_list[0])
second = mean(result_list[1])
# third = mean(result_list[2])
# fourth = mean(result_list[3])

print("average of first parameters = ",first)
print("average of second parameters = ",second)
# print("average of third parameters = ",third)
# print("average of fourth parameters = ",fourth)