import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
DEBUG = False

class DecisionTreesApplication:
    def __init__(self):
        df = pd.read_csv("decision_trees/processed.cleveland.data", header=None)
        df.columns= ['age',
            'sex',
            'cp',
            'restbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal',
            'hd']
        self.data_set = df.loc[(df['ca']!= '?') & (df['thal'] != '?')]
    
    def get_data_set(self):
        return self.data_set
    
    def get_X(self):
        return self.data_set.drop('hd', axis=1).copy()
    
    def get_Y(self):
        return self.data_set['hd'].copy()
    




dts = DecisionTreesApplication()
X = dts.get_X()

# Transform/Augment categorical features to their one-hot encodings for better ML predictions
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])

# Let's restrict the Y values to either 0 or 1 ie; any value > 0 will be set to 1
Y = dts.get_Y()
Y[Y > 0] = 1

# Split the dataset into Training and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=42)
print(f'Traning len: {X_train.shape}, Test len: {X_test.shape}')

# Use scikit-learn library to create the DecisionTree for our classification and fit the training dataset
# What happens internally when the following code is run
#
# - It calculates Gini impurity for each feature and selects the root node 
#   based on the feature that results in the lowest Gini impurity after the split.
# - After selecting the root node, it recursively moves down the tree, creating branches and leaf nodes. 
#   At each level, it repeats the process of calculating Gini impurity for potential splits and selecting the 
#   feature that minimizes impurity.
# - This process continues until all data points are classified into leaf nodes.
#
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
dtc = DecisionTreeClassifier(random_state=42)
dtc = dtc.fit(X_train, Y_train)

# Now the above will try to fit all the training data resulting in overfitting. The below code will help visualize the graph
if DEBUG: # Enabling debugging to see the intermediate stages of understanding the decision trees
    plt.figure(figsize=(15, 7.5))
    plot_tree(dtc, filled=True, rounded=True, class_names=["No HD", "Yes HD"], feature_names=X_encoded.columns)

    # Plot the Confusion Metrix To understand how the model is behaving. You should see it is overfitting at this point and the Train accuracy is lower
    # than the Test accuracy
    ConfusionMatrixDisplay.from_estimator(dtc, X_test, Y_test, display_labels=["Does not have HD", "Have HD"])

# The next step is to Prune the leaves/Tree to generalize the tree and address the overfitting problem
# We can have different trees by pruning the leaves. The question is how to find such a Tree. We will use a regularization technique called 
# CCP (Cost Complexity Pruning) to decide the tree 
# Tree Score = Total GINI Impurity + alpha * (number_of_leaves)
# As tree size is reducing, the alpha value are increasing. Higer the alpha, shorter the tree. We should find the sweet spot to get the best
# alpha

# Cost Complexity Pruning
# - Compute the GINI Impurity for all leaf nodes for that tree + alpha * (number_of_leaves)
# - Ex: Let's say we have 3 different Trees after removing some leaves.
#       - Tree Score for overfitting fully training data will be 0 + alpha * 3 (Assuming there are 3 leaf nodes)
#       - Tree Score for another Tree with a pruning => 0.213 + alpha * 2
#       - Tree Score for another which just predicts the same => 0.342 + alpha * 1 

# Basically we are penalizing the tree if it has too many leaf nodes

ccp_alphas_path = dtc.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas = ccp_alphas_path.ccp_alphas

if DEBUG: # Enabling debugging to see the intermediate stages of understanding the decision trees
    tree_list = []
    for ccp_alpha in ccp_alphas:
        dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        dt.fit(X_train, Y_train)
        tree_list.append(dt)    

    # The scores are nothing but accuracy that we can infer from the Confusion Matrix
    train_scores = [tree.score(X_train, Y_train) for tree in tree_list]
    test_scores  = [tree.score(X_test, Y_test) for tree in tree_list]

    # Now use the test data against each of the above decision tree and get the which one has maximum accuracy
    # We can plot a graph with tree with a specific alpha on X-axis and training and test accuracy on Y-axis

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy V/s Alpha")
    ax.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle="steps-post")
    ax.legend()

# Now the above is for the training/test split we chose. But, there could be some other split which can give us a better approach.
# We have to use k-fold cross validation


all_alpha_mean_accuracy_scores = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    # k-fold validation scores for each alpha
    scores = cross_val_score(dt, X_train, Y_train, cv=5)
    all_alpha_mean_accuracy_scores.append([ccp_alpha, np.mean(scores), np.std(scores)])


results = pd.DataFrame(all_alpha_mean_accuracy_scores, columns=['alpha', 'mean_accuracy', 'std'])
results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')

plt.show()



# print(df.head())
# print(df.dtypes)
# print(df['ca'].unique())
# print(df['thal'].unique())

# print(df.loc[(df['ca']=='?') | (df['thal']== '?')])
# print(f"Total Patients: {len(df)}")