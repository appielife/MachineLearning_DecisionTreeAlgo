# This code is orignial work of Arpit Parwal  and Yeonsoo Park of University of Southern California for the subject
# INF 552 Machine Learning

#  The program takes in specific format of input data a sample of which can be found in dt_data.txt file found in this folder
#  The program works on ID3 Algorithm to build a decision tree. It also prints out a neat decision tree.


# This function is used to fetch and fromat training data from a file

prediction_label=''
def getData():
    filePath = 'dt_data.txt'
    attributes = []
    dataset = open(filePath)
    contents = list()
    try:
        line = dataset.readline()
        line = line[1:-2]
        # gets the attributes of the file and save it as a separate structure
        attributes = line.split(",")
        while line != '':  # The EOF char is an empty string
            line = dataset.readline()
            if line.replace("\r", "").replace("\n", ""):
                contents.append(line[4:-2].replace(" ", "").split(","))
    finally:
        dataset.close()
    # contents contains all the training set stored as nested array.
    global prediction_label
    prediction_label=attributes[len(attributes)-1]

    return attributes, contents


# [HELPER FUNCtION] to count number of unique values
# helps in calculate impurity of the set
# in our dataset format, the label is always the last column
def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Question:    
    #  Question is used to divide a dataset.
    # This class just records a 'column number' (e.g., 0 for Occupied, 1 for Price etc) and a
    # 'column value' (e.g., Hight, Low etc.).
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # The 'match' method is used to compare
    # the feature value in an example to the feature value stored in the
    # question. See the demo below.
    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "equal to "
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def divide(rows, question):
    # divides a dataset.

    # For each row in the dataset, check if it matches the question. If
    # so, add it to 'true rows', otherwise, add it to 'false rows'.
    # 
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def entropy(rows):
    # Calculate the entropy/Impurity for a list of rows.  
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    # Information Gain.
    # The uncertainty of the starting node, minus the weighted impurity of
    # two child nodes.
    # 
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows):
    # Find the best question to ask by iterating over every feature / value
    # and calculating the information gain.
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = divide(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain > best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    # A Leaf node classifies data.

    # This holds a dictionary of class (e.g., "High") -> number of times
    # it appears in the rows from the training data that reach this leaf.
    # 

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    # A Decision Node asks a question.

    # This holds a reference to the question, and to the two child nodes.
    # 

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    # Builds the tree.

    # Rules of recursion: 1) Believe that it works. 2) Start by checking
    # for the base case (no further information gain). 3) Prepare for
    # giant stack traces.
    # 

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to divide on.
    true_rows, false_rows = divide(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    # World's most elegant tree printing function.

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        global prediction_label
        print(spacing + ' '+  prediction_label, node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> if true then:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> if false then:')
    print_tree(node.false_branch, spacing + "  ")


def examine(row, node):
    # """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return examine(row, node.true_branch)
    else:
        return examine(row, node.false_branch)


def print_leaf(counts):
    # A nicer way to print the predictions at a leaf. This function tell us how we can split the data 
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == '__main__':
    #  the main function where everything happens

    header, training_data = getData()
    # This gets the data in formatted from 
    # Header contains the attriputes
    # Training data contains all the rows 
    my_tree = build_tree(training_data)
    # This function return the root of the tree as a  decison_node class  object
    print_tree(my_tree)
    #  Just a fucntion to beautifully print a tree
    test_data = [['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']]
    #  A test data as given by professor

    # Why the for loop you ask? JUST for fun !!
    for row in test_data:
        print("Actual: %s. Enjoyed?: %s" %
              (row[-1], print_leaf(examine(row, my_tree))))
