import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, GaussianNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

''' https://www.kaggle.com/vbookshelf/sklearn-naive-bayes '''

# Load the dataset into the dataframe
df = pd.read_csv("8410_processed_sklearn_FULL.csv")

# split into attributes and labels
target = 'age_group'
X = df.drop(['srcid', target], axis=1)
X = X.drop(['p_state','sDevType','sOSName','education','gender'], axis=1)
# X = X.drop(['Q6b','Q6c','Q6d','Q7b','Q7c','Q7d','Q10b','Q10c','Q10d'], axis=1)
y = df[target]

''' srcid,Q1,Q2,Q4,Q6a,Q6b,Q6c,Q6d,Q7a,Q7b,Q7c,Q7d,Q7e,Q10a,Q10b,Q10c,Q10d,
    p_state,sDevType,sOSName,gender,age_group,education'''


# # Convert features to Ordinal values
# ordinalencoder_X = OrdinalEncoder()
# X = ordinalencoder_X.fit_transform(X)
# X = X.astype(int)

# Convert features to OneHotEncoding values
one_hot_encoder_X = OneHotEncoder()
X = one_hot_encoder_X.fit_transform(X).toarray()
X = X.astype(int)
print(X)


# Convert target to Ordinal values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

nb_list = [MultinomialNB,ComplementNB,GaussianNB,BernoulliNB]
result_list = [[] for x in range(len(nb_list))]


for j in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    for i in range(len(nb_list)):
        # choose the classifier
        nb = nb_list[i]()

        # Train the classifier
        nb.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = nb.predict(X_test)

        # Output confusion matrix and classification report
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average='weighted')
        # print(f1)
        result_list[i].append(f1)

# print(result_list)

multi = mean(result_list[0])
comp = mean(result_list[1])
gaus = mean(result_list[2])
bern = mean(result_list[3])

print("average multinomal = ",multi)
print("average complement = ",comp)
print("average gaussian = ",gaus)
print("average bernoulli = ",bern)
