import os

def csvreader(dataset):
    """ Function to read the csv file 

    returns:
    raw_data = list() containing the data
    features = the header of the dataset
    """

    # assign 2 dummy variables
    temporary = []  # store data temporarily
    raw_data = []   # store the processed data

    for line in dataset:
        # use the \n as the splitting factor
        # then append the the data after the split
        
        # Example: 
        # csv:
        # a,b,c,d,e
        # f,g,h,j,k
        # f,g,h,j,k
        # ...

        # will be:
        # [['a,b,c,d,e',''], ['f,g,h,j,k','']]
        # note: the '' appeared because of the splitting method
        # the last one will not have it because of no \n
        dataline = line.split('\n')
        temporary.append(dataline)

    for line in temporary:
        # the last element of each data is  ''
        # remove the '', if exists

        # Example:
        # [['a,b,c,d,e',''], ['f,g,h,j,k','']] --> [['a,b,c,d,e'], ['f,g,h,j,k'],...]
        for data in line:
            if data == '':
                line.remove(data)

    for line in temporary:
        # split the data (' ,')
        # ['a, b, c, d, e'] --> ['a','b','c', 'd', 'e']
        # append to new list (rawdata) (make a 2d list)
        # Result
        # [[HEADER], ['a','b','c', 'd', 'e'],['a','b','c', 'd', 'e'],
        # ['a','b','c', 'd', 'e'],['a','b','c', 'd', 'e'],...]
        
        for data in line:
            words = data.split(', ')
            raw_data.append(words)

    # the first element of the list is the header (like acidity, ph, etc.)
    # pop the header (we don't want the header to mess up with our data)
    features = raw_data.pop(0)

    return raw_data, features


def convert(dataset):
    """ Function to change the data values (from string to float) 
    and seperates the wine quality"""

    target_data = [] # consists of only 0 and 1

    # iterate 
    for i in range(len(dataset)):

        # the sub list of the 2d list
        current = dataset[i]

        # change them to float
        for k in range(len(current)):
            current[k] = float(current[k])

        # We use the simpler method to classify the data
        # for the last element of the sublist (ratings)
        # classify the data into 2 kind
        # >6 and <=6
        # if smaller <= 6, then change them to 0

        # knowing the real value is not realy important because we are doing classfication
        # in regression, however, real value of the prediction is needed
        # quoted "predict whether the quality score of the wine is above 6 or not."
        # only 'above 6 or not
        if current[-1] > 6:
            current[-1] = 1
            # after changing the value, pop the 0 and 1
            target_data.append(current.pop())
            
        else:
            current[-1] = 0
            # after changing the value, pop the 0 and 1
            target_data.append(current.pop())

    return target_data
    
here = os.path.dirname(os.path.abspath(__file__))
d_train = os.path.join(here, 'train.csv')
d_test = os.path.join(here, 'test.csv')
data_train = open(d_train, 'r')
data_test = open(d_test, 'r')

# train_raw_data is a list of lists (2d lists) that contain the data values of each sample
# excluding the wine quality, in the training data set
# Each sample is stored as its sub list ([[5.6, 3.2, 7.7, etc], [4.5, 8.6, 4.3, etc], etc])

# train_features is a list that contain the features/heading of the data values in the training data set(fixed acidity, volatile acidity, etc)
train_raw_data, train_features = csvreader(data_train)

# Same as above except the data is taken from the (test data) set
test_raw_data, test_features = csvreader(data_test)

# train_target_data is a list that contains the wine quality of the train datasey
# it contains 0 and 1 --> [0,1,0,1,0,0,0,0,1,0,1,0,1,0,...]
train_target_data = convert(train_raw_data)
# Same as above except the data is taken from the test data set
test_target_data = convert(test_raw_data)

""" ^ csv reader """
""" -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" v Building decision tree """

class Node:
    """ Class that defines the node in the decision tree """
    
    def __init__(self, feature=None, threshold=None, true_branch=None, false_branch=None, *, value=None):
        """ Constructor for the Node class
        
        Object: 
        -feature = The column number, the header value (header[0] = fixed acidity or header[1] = volatile acidity)
        -threshold = The threshold value of the feature (wine acidity (feature) value, 6.4(threshold) - acting as comparator)
        -true_branch = The true branch pointer, if threshold is false \
            suppose threshold = 7 and condition is > 4, then it is clear that 7>4 is True
            add to the false branch 
        -false_branch = The false branch pointer, if threshold is false \
            suppose threshold = 7 and condition is < 4, then it is clear that 7<4 is False
            add to the false branch
        -value = Stores the value of the node (will have none if it is a leaf node) """

        self.feature = feature
        self.threshold = threshold
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.value = value

    def leaf_checker(self):
        """ Method to check whether the node is a leaf node or not

        As explained, the leaf node happens if the value == None
        
        returns:
        bool, True or False """
        
        return self.value is not None

class DecisionTree:
    """ Tree Class initializer """

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """ Constructor for the Decision Tree class """

        # The minimum number of samples that the remaining dataset that can split (It can only a minimum of 2 different samples)
        # ^ basically, after splitting the dataset will be reduced
        # Ex:
        # root tree = 100 sample - 75 false and 25 true
        # the 25 true is splitted again (24 and 1), 1 cannot be splitted anymore
        self.min_samples_split = min_samples_split
        
        # Stores the maximum branch that a decision tree can have
        # ^ lets say the tree is: root - branch1 - branch2 - branch3 -branch 4
        # then the depth is 4 (root=0)
        self.max_depth = max_depth 

        # Stores the number of features/columns that the sample has (wine acidity, citric acid, etc)
        self.n_feats = n_feats
        
        # the root node of the tree
        self.root = None

    def count(self, target_data):
        """ Method that counts the occurence of train_target_data samples with the ratings that are <=6
        (denoted as 0) and those above 6 (denoted as 1) 
        
        returns:
        counts = dictionary {0:x,1:y} """

        counts = {}
        # Ex:
        # There are 5 zero and 5 one
        # {0:5, 1:5}
        # There are 2 zero and 7 one
        # {0:2, 1:7}
        
        for item in target_data:
            counts[item] = counts.get(item, 0) + 1

        return counts
    
    def common_label(self, target_data):
        """ Method that looks for the most common label (1 or 0) from the data in a leaf node
        in case of reaching the max depth first

        let's say:
        The max depth is set to 3 only, then the tree building stopped when there are 3 layer of branches
        Ex: root -> branch 1 -> Branch 2 (depth is 3)

        Then among the divided dataset (the dataset is being split everytime), it will get the most common appearing data
        and let the data to be the leaf node.
        In case of the number of 0 and 1 is equal, it will pick 0

        Ex: [0,0,0,0,0,0,1,1,1,0] --> will pick 0

        NOTE: Having small depth can improve the time limit, but the accuracy will be much worse

        returns:
        0 or 1 """

        counts = self.count(target_data) # dictionary

        # sort the dictionary in descending order 
        # check this: https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
        sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}

        # convert the dict (counts) to the list of tuples
        # [(key1,value1),(key2, value2),...]
        counter = [(data, number) for data, number in sorted_counts.items()]
        common_label = counter[0][0] #Takes either 0 or 1 depending on which has more occurences

        return common_label

    def gini(self, target_data):
        """ Method that will calculate the gini index of the node
        
        This will repeteadly count the gini
        
        returns:
        impurity = float """

        counts = self.count(target_data) #The result of the dictionary from the previous method
        # Follows the formula to find the gini index: 
        # 1 - (number of zeros/total data)^2 - (number of ones/total data)^2

        impurity = 1
        for label in counts:
            # for key = 0 and for key = 1
            # new impurity = 1 - (number of zero/total)^2 - (number of one/total)^2
            prob_of_label = counts[label] / float(len(target_data))
            impurity -= prob_of_label**2

        return impurity

    def information_gain(self, true_data, false_data, target_data, current_uncertainty):
        """ Method that will calculate the information gain of numerous split

        Uses the gini impurity improvement to decide the most gain
        * most gain equal to better impurity (the data is less messy)

        This function will be called in the best_split(), as the deciding factor for the best splitting point
        
        Case study for the google play (because the wine dataset is confusing to apply for this logic):
        we assume that using name for the splitting method will not have any effect to the info gain,
            since almost every app has different name

         """
        
        p = float(len(true_data))/(len(true_data) + len(false_data))
        return current_uncertainty - p * self.gini([target_data[i] for i in true_data]) - (1 - p) * self.gini([target_data[i] for i in false_data])

    def train(self, data, target_data):
        """ Method that will train the program according to the training data
        it is a wrapper function for the actual grow tree function
        
        in short, it initialize the grow tree (calls the grow_tree) """

        if not self.n_feats: # Is a safety check used to check the number of features in a tree in case if its zero or has a value
            # will work perfectly fine if this function is deleted (with some modification to the else)
            self.n_feats = len(data[0])
        else:
            self.n_feats = min(self.n_feats, len(data[0])) #Picks the smallest value of the number of features

        self.root = self.grow_tree(data, target_data) #Runs the grow_tree method of the class
    
    def grow_tree(self, data, target_data, depth=0):
        """ Method that will grow the decision tree 
        
        USES RECURSION """

        gain, best_feature, best_threshold = self.best_split(data, target_data) # Stores the information gain, best feature and best threshold to split on from the method of find best split
        n_samples = len(data) # Number of samples in the data set
        n_labels = len(set(target_data)) # The number of classifications of the quality of wine (by default this should be 2 since its only true or false)

        # Base case to stop the recursion to grow the tree
        # If:
        # -the max depth is reached (higher max depth == better accuracy)
        # -it has only 1 sample left (well, basically you can't split a data)
        # -sample is more than 1, but contains the same value (0,0,0,0,0,0,0,0,0,0,0,0)
        # -if the information gain is 0 (pointless to continue then, since the data is in perfect shape)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or gain == 0):
            leaf_value = self.common_label(target_data)

            # if base case reached, assign the node to become a leaf node (contain 0 or 1)
            # 0 for lower than, equal to 6
            # 1 for higher than 6
            return Node(value=leaf_value)
        
        # If base case is not met, then it will split the data set into two
        # the split uses the returns from the:
        # best_split function as the splitting point decider
        true_data, false_data = self.split([row[best_feature] for row in data], best_threshold)

        # true_data stores a list of indices that locate the samples that meet the splitting critera
        #   (if the feature data in that sample is below or equal to the threshold)
        # false_data stores a list of indices that locate the samples that fail to meet the splitting criteria

        # RECURSION - for the splitted dataset (look at split())
        true_branch = self.grow_tree([data[i] for i in true_data], [target_data[i] for i in true_data], depth + 1)
        false_branch = self.grow_tree([data[i] for i in false_data], [target_data[i] for i in false_data], depth + 1)
        
        # assign the Node values
        return Node(best_feature, best_threshold, true_branch, false_branch)

    def best_split(self, data, target_data):
        """ Method that will find the best feature and best threshold to split the data. 
        
        will compare the gain. The largest gain means that it is the best splitting point.
        IT IS POINTLESS TO SPLIT, IF THE SPLIT INCREASES THE IMPURITY

        case study for the google play (because the wine dataset is confusing to apply for this logic):
        the name will not have any effect to the rating, thus the name will not be the threshold
        ^ find best split choose what split benefits the most

        * we want for the depth to be as small as possible, for optimization (because we need to recurse to grow tree
        * and recursion is not efficient, we want as less as recursion as possible that gives us the best accuracy) """

        # will store the value for the best information gain of the split
        # best gain will start at -1, because it's small - so the gain will not be disturbed
        best_gain = -1

        # will store the best feature and the best threshold of the split
        best_feature, best_threshold = None, None 
        current_uncertainty = self.gini(target_data) # calculates the gini index of the data set

        for col in range(self.n_feats):     # Uses a for loop to check every feature in the data set (fixed acidity, volatile acidity)
            values = [row[col] for row in data]     # Stores the values of a column of data from the data set
            thresholds = set(values)    # Stores the unique values of those data
            for threshold in thresholds:    # Uses a for loop to check every one of those unique threshold values 
                true_data, false_data = self.split(values, threshold)   # Stores the list of indices of the true and false data

                if len(true_data) == 0 or len(false_data) == 0: # Checks to skip the current split if it doesnt divide the dataset
                    continue

                gain = self.information_gain(true_data, false_data, target_data, current_uncertainty) # Calculates the information gain of the split

                # compare the current gain
                # if the new gain (by different splitting point) is better
                # assign the new threshold, feature, and gain
                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_threshold = threshold

        return best_gain, best_feature, best_threshold

    def split(self, values, threshold):
        """ Method to split the dataset
        
        it will split the dataset according to the threshold that is setted in best_split() function 
        
         """
        true_data, false_data = [], []
        # Stores the indices of the values that meet the splitting criteria to the assigned list
        # true_data and false_data is a 2d list each
        # they can be called as the new(smaller) datasets, because the tree splits them

        # [
        # [a1,b,1,d,e]
        # [a2,b,2,d,e]
        # [a3,b,3,d,e]
        # [a4,b,4,d,e]
        # [a5,b,5,d,e]
        # ]

        # let's say, the splitting point is the third column and threshold is 3
        # then it will split:
        # 1<=3 - T
        # 2<=3 - T
        # 3<=3 - T
        # 4<=3 - F
        # 5<=3 - F

        # thus the true_data = [[a1,b,1,d,e], [a2,b,2,d,e], [a3,b,3,d,e]]
        # thus the false_data = [[a4,b,4,d,e], [a5,b,5,d,e]]

        # then those data will again be useto grow the tree (using recursion)

        for index, row in enumerate(values):
            if row <= threshold:
                true_data.append(index)
            
            else:
                false_data.append(index)

        return (true_data, false_data)
    
    """ ^ Training """
    """ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
    """ v Testing/predicting """

    def predict(self, data):
        """ Method that will predict whether or not the quality of the wine will be below 6 or above,
        it is a wrapper function for the traverse function (will call the traverse function)
        
        returns:
        list() containing prediction of the rating (uses 0 and 1)
        IE: [0,1,0,1,0,1,1,1,1,1,1,1] """

        # self.traverse() takes lists as its input
        # thus uses a for loop for every data sample to run each sample through self.traverse
        return [self.traverse(sample, self.root) for sample in data] 

    def traverse(self, sample, node):
        """ Method that will traverse the decision tree from the root node to reach a leaf node 
        
        Traverse will use the created tree (from train)
        to predict the quality (0 or 1) of the testing data """

        if node.leaf_checker(): #Base case to stop the recursion
            return node.value

        # If the data of a specific feature in a sample meets the threshold,
        # then it proceeds to the true branch, otherwise it proceeds to the false branch
        if sample[node.feature] <= node.threshold: 

            # recursion if true
            return self.traverse(sample, node.true_branch)

        else:
            # recursion if false
            return self.traverse(sample, node.false_branch)

""" End of class """
""" ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ """

def accuracy(predictions, target_data):
    """ Function to calculate the accuracy of the prediction """

    correct_predictions = 0

    # compare the predictions result (a list) to the real quality (a list from the csv)
    # it will compare if the prediction is correct or not
    # if pred = 0 and real = 0, then it's correct
    for line in range(len(predictions)):

        # print('Real:', target_data[line],'|', 'Prediction:',predictions[line])   # if you want to print the prediction, enable this function
        T_or_F = predictions[line] == target_data[line]
        if T_or_F:
            correct_predictions += 1

    return correct_predictions, (correct_predictions/len(predictions)) * 100

def main():
    """ Main function that will run the program """
    
    my_tree = DecisionTree(max_depth=100)       # you can change the max depth for altering the accuracy
                                                # lower max depth == lower accuracy
    my_tree.train(train_raw_data, train_target_data)    # Trains the decision/builds the tree according to the training data

    predictions = my_tree.predict(test_raw_data)  # Makes predictions of the test data (without wine quality being included as one of the features)
    correct_predictions, acc = accuracy(predictions, test_target_data)  # Finds the accuracy of the system's predictions

    # Prints the results of the decision tree's predictions
    print()
    print("RESULTS")
    print("Number of samples    :", len(test_target_data))
    print("Correct Predictions  :", correct_predictions)
    print("Incorrect Predictions:", len(test_target_data) - correct_predictions)
    print()
    print("Accuracy             : ", '{0:0.3f} %'.format(acc))

if __name__ == "__main__":
    """ Driver code """
    
    main()