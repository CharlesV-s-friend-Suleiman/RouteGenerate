import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# training model
df_raw_train = pd.read_csv('data/trainRealData.csv')

X_train = df_raw_train[['speed','TG','GG','GSD','TS']]
Y_train = df_raw_train['mode']

clf = RandomForestClassifier(max_depth=9, n_estimators=200, max_features=10, random_state=42)
clf.fit(X_train, Y_train)


# testing model, artificial_traj_features.csv realWorldFeatures.csv
df_raw_test = pd.read_csv('data/realWorldMixedFeatures.csv')

X_test = df_raw_test[['speed','TG','GG','GSD','TS']]
Y_test = df_raw_test['mode']
Y_pred = clf.predict(X_test)


def majority_vote(predictions, ids):
    vote_results = {}
    for idx, pred in zip(ids, predictions):
        if idx not in vote_results:
            vote_results[idx] = []
        vote_results[idx].append(pred)

    final_predictions = []
    for idx in ids:
        final_predictions.append(Counter(vote_results[idx]).most_common(1)[0][0])

    return final_predictions

Y_pred_voted = majority_vote(Y_pred, df_raw_test['ID'])

# accuracy,precision,recall,f1
accuracy = accuracy_score(Y_test, Y_pred)
presicion = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')


print('Accuracy: ', accuracy, '\n')
print('Presicion: ', presicion, '\n')
print('Recall: ', recall, '\n')
print('F1: ', f1, '\n')