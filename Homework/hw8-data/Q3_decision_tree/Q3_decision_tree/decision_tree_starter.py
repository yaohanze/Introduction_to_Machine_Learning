"""
This is the starter code and some suggested architecture we provide you with. 
But feel free to do any modifications as you wish or just completely ignore 
all of them and have your own implementations.
"""
import numpy as np
import scipy.io
from scipy import stats
import random
import math

def shuffle_data(x, y):
    training = list(zip(x, y))
    shuffle(training)
    return zip(*training)

class Node:

    def __init__(self, samples, left=None, right=None, split=None, label=None, depth=0):
        self.left = left
        self.right = right
        self.split = split
        self.label = label
        self.samples = samples
        self.depth = depth

class DecisionTree:

    def __init__(self, max_depth, smallest_node):
        """
        TODO: initialization of a decision tree
        """
        self.node = None
        self.max_depth = max_depth
        self.smallest_node = smallest_node

    def split(self, X, y, j, B_index):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        n, d = X.shape
        y = np.expand_dims(y, axis=1)
        stacked = np.hstack([X, y])
        A = stacked[stacked[:, j].argsort()]
        yl, yr = A[:B_index, d], A[B_index:, d]
        Sl, Sr = A[:B_index, :d], A[B_index:, :d]
        return (Sl, yl, Sr, yr)

    def max_vote(self, classes, counts):
        return classes[np.argmax(counts)]
    
    def segmenter(self, X, y, feature_set=None):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain

        X: n' x d data matrix
        y: n' x 1 labels
        feature_set: a set of allowable feature indices in the case of random forest
        """
        n, dim = X.shape
        if feature_set is not None:
            allowable_features = feature_set
        else:
            allowable_features = range(dim)
        minH, minj, minB_index, min_thresh = math.inf, dim, n, -1
        y = np.expand_dims(y, axis=1)
        stacked = np.hstack([X, y])
        #Pick a feature to split on
        for j in allowable_features:
            A = stacked[stacked[:, j].argsort()]
            if len(np.unique(A[:, j])) == 1:
                continue
            #Pick a thresold of that feature
            #Note B_index is not an actual threshold value
            for B_index in range(1, n):
                if B_index == 1:
                    C, D, c, d = self.clever_trick(A[:, dim])
                else:
                    if A[B_index-1, dim] == 1:
                        D, d = D+1, d-1
                    else:
                        C, c = C+1, c-1
                    #Skip splitting at B_index if next X_i has the same value for feature j
                    if B_index > 0 and A[B_index][j] == A[B_index-1][j]:
                        continue
                    if C == 0 or D == 0:
                        leftcost = 0
                    else:
                        leftcost = -C*math.log2(C/(C+D)) - D*math.log2(D/(C+D))
                    if c == 0 or d == 0:
                        rightcost = 0
                    else:
                        rightcost = -c*math.log2(c/(c+d)) - d*math.log2(d/(c+d))
                    J = (leftcost + rightcost)/n

                    if J < minH:
                        minH, minj, minB_index, min_thresh = J, j, B_index, A[B_index, j]
        return (minj, minB_index, min_thresh)

    def clever_trick(self, y):
        """
        Given a vetor y sorted by some unknown feature, and assuming the split
        is just on the first item, will return C, D, c, d, where C, D is the 
        occurence of class 0, 1 to the left of the split, and c, d is the 
        occurence of class 0, 1 to the right of the split.
        """
        C, D = 1 - y[0], y[0]
        d = sum(y[1:])
        c = len(y) - d - 1
        return C, D, c, d 

    def grow_tree(self, X, y, depth):
        n, d = X.shape
        classes, counts = np.unique(y, return_counts=True)
        if (len(counts) == 1):
            return Node(n, None, None, None, label=y[0], depth=depth)
        if depth > self.max_depth or n < self.smallest_node:
            return Node(n, None, None, None, label=self.max_vote(classes, counts), depth=depth)
        #Implement more stopping criteria here
        j, B_index, thresh = self.segmenter(X, y)
        if B_index == n:
            return Node(n, None, None, None, label=self.max_vote(classes, counts), depth=depth)
        Sl, yl, Sr, yr = self.split(X, y, j, B_index)
        return Node(n, self.grow_tree(Sl, yl, depth+1), self.grow_tree(Sr, yr, depth+1), [j, thresh], depth=depth)
    
    def train(self, X, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        self.node = self.grow_tree(X, y, 0)

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        y = []
        n, d = X.shape
        curr_tree = self.node
        for point in range(n):
            curr_node = curr_tree
            while curr_node.label is None:
                j, thresh = curr_node.split
                if X[point, j] < thresh:
                    curr_node = curr_node.left 
                else:
                    curr_node = curr_node.right 
            y.append(curr_node.label)
        return np.array(y)


class RandomForest():
    
    def __init__(self, max_depth, min_node, num_trees, xset_size):
        """
        TODO: initialization of a random forest
        """
        self.trees = []
        self.num_trees = num_trees
        self.xset_size = xset_size
        self.max_depth = max_depth
        self.min_node = min_node

    def train(self, X, y, validation=[]):
        """
        TODO: fit the model to a training set.
        """
        n, d = X.shape
        m = d
        classes, counts = np.unique(y, return_counts=True)
        if (len(counts) == 1):
            return Node(n, None, None, None, label=y[0], depth=depth)
        if depth > self.max_depth or n < self.smallest_node:
            return Node(n, None, None, None, label=self.max_vote(classes, counts), depth=depth)
        features = np.random.choice(range(d), m, replace=False)
        j, B_index, thresh = self.segmenter(X, y, feature_set=features)
        if B_index == n:
            return Node(n, None, None, None, label=self.max_vote(classes, counts), depth=depth)
        Sl, yl, Sr, yr = self.segmenter(X, y, feature_set=features)
        return Node(n, self.grow_tree(Sl, yl, depth+1), self.grow_tree(Sr, yr, depth+1), [j, thresh], depth=depth)

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        n, d = X.shape
        ypred_votes = []
        for t in self.trees:
            tvote = t.predict(X)
            ypred_votes.append(tvote)
        ypred_votes = np.array(ypred_votes)
        ypred = []
        for y_index in range(n):
            y = ypred_votes[:, y_index]
            ypred.append(stats.mode(y)[0][0])
        return np.asarray(ypred)


def visualize_tree(curr_node, features, classes, depth=0):
    if curr_node.label is not None:
        print(print(’  ’*curr_node.depth + "\%d.(label=\%s | \%d samples)" \% (depth,classes[int(curr_node.label)],curr_node.samples))
    else:
        print(’  ’*curr_node.depth + "\%d.(feature=\%s, thresh=\%d | \%d samples)" \% (depth,features[curr_node.split[0]], curr_node.split[1], curr_node.samples))
        visualize_tree(curr_node.left,features,classes,depth+1)
        visualize_tree(curr_node.right,features,classes,depth+1)

    
if __name__ == "__main__":

   
    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]
    assert len(features) == 32

    # Load spam data
    path_train = 'datasets/spam-dataset/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    y = np.squeeze(data['training_labels'])
    class_names = ["Ham", "Spam"]
     

    """
    TODO: train decision tree/random forest on different datasets and perform the tasks 
    in the problem
    """
