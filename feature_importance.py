import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from otto_challenge import load_db

X, y = load_db('train.csv')
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

X, y = X_train, y_train

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices, rotation='horizontal')
plt.xlim([-1, X.shape[1]])
plt.show()

indices_best_features = np.arange(len(importances))[importances > 0.01]
print(indices_best_features)

print('----------------------------------')
print('TEST SCORE, ALL FEATURES: ', forest.score(X_valid, y_valid))
print('----------------------------------')
forest_feature = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest_feature.fit(X_train[:, indices_best_features], y_train)
print('TEST SCORE, MAIN FEATURES: ', forest_feature.score(X_valid[:, indices_best_features],
                                                          y_valid))
