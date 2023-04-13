# Import libraries
import pandas as pd
from collections import Counter
from math import log2
import mltools as ml


def total(cnt):
    """
    Determine the sum of the values of the Counter.
    Takes in cnt (Counter) a counter
    Returns sum of the values of the Counter
    """

    return sum(cnt.values())


def gini(cnt):
    """
    Determine the gini index of the input.
    Takes in cnt (Counter) a counter
    Returns gini index of the input Counter
    """

    tot = total(cnt)
    return 1 - sum([(v / tot) ** 2 for v in cnt.values()])


def entropy(cnt):
    """
    Determine the entropy of the input.
    Takes in cnt (Counter) a counter
    Returns entropy of the input Counter
    """

    tot = total(cnt)
    return sum([(-v / tot) * log2(v / tot) for v in cnt.values()])


def wavg(cnt1, cnt2, measure):
    """
    Determine the weighted average sensitivity.
    Takes in cnt1 (Counter) a counter, cnt2 (Counter) a conter, measure (gini or entropy) which measure to use
    Returns the weighted average sensitivity of the inputted Counters
    """

    tot1 = total(cnt1)
    tot2 = total(cnt2)
    tot = tot1 + tot2
    return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot


def evaluate_split(df, class_col, split_col, feature_val, measure):
    """
    Evaluate the split for the dataframe given the column to split upon, value to split upon, and measure to use.
    Parameters:
        df (dataframe) the dataframe to evaluate the split for
        class_col (array) the target values
        split_col (array) which column to split upon
        feature_val (value) which value to split upon
        measure (gini or entropy) which measure to use

    Returns:
        the weighted average sensitivity of the splits
    """
    df1, df2 = df[df[split_col] == feature_val], df[df[split_col] != feature_val]
    cnt1, cnt2 = Counter(df1[class_col]), Counter(df2[class_col])
    return wavg(cnt1, cnt2, measure)


def best_split_for_column(df, class_col, split_col, method):
    """
    Determine the best split for the column.
    Parameters:
        df (dataframe)
        class_col (array)
        split_col (array)
        method (gini or entropy) which measure to use for determining the best split

    Returns:
        best_v (value) the best value to split upon for given column
        best_meas (gini or entropy) the best measure to use
    """
    best_v = ''
    best_meas = float("inf")

    for v in set(df[split_col]):

        meas = evaluate_split(df, class_col, split_col, v, method)
        if meas < best_meas:
            best_v = v
            best_meas = meas

    return best_v, best_meas


def best_split(df, class_col, method):
    """
    Determine the best split of a dataframe.
    Parameters:
        df (dataframe) the dataframe to split
        class_col (array) the outcomes that one is trying to predict
        method (gini or entropy) which method used to determine the best split

    Returns:
        best_col (array) the best column to split the dataframe upon
        best_v (value) the ideal value to split the dataframe upon
        best_meas (gini or entropy) the best measure for the split
    """
    best_col = 0
    best_v = ''
    best_meas = float("inf")

    for split_col in df.columns:
        if split_col != class_col:
            v, meas = best_split_for_column(df, class_col, split_col, method)
            if meas < best_meas:
                best_v = v
                best_meas = meas
                best_col = split_col

    return best_col, best_v, best_meas


# Some helper functions
def left(t):
    """
    Returns the left branch of the tree if it is not none.
    Takes in T (tree) the tree to return the left branch for
    Returns T (tree) the left branch of the tree if it exists, else None
    """
    return t[1] if t is not None else None


def right(t):
    """
    Returns the right branch of the tree if it is not none.
    Takes in T (tree) the tree to return the right branch for
    Returns T (tree) the right branch of the tree if it exists, else None
    """
    return t[2] if t is not None else None


def depth(t):
    """
    Returns the maximum depth of the tree.
    Takes in T (tree) the tree to calculate maximum depth of
    Returns the value of the greatest depth of the tree
    """
    # If tree is None return -1, else return max depth of left or right branch plus one
    if t is None:
        return -1
    else:
        return 1 + max(depth(left(t)), depth(right(t)))


def dtree(train, criterion, depth=0, max_depth=None, min_instances=2, target_impurity=0.0):
    """
    Construct a decision tree for the given data.
    Parameters:
        train (df) a training dataset 
        criterion (gini or entropy) the attribute selection method used to find the optimal split
        depth (int) the current depth of the tree
        max_depth=None (int) the maximum allowed depth of the tree
        min_instances=2 (int) the minimum number of (heterogenous) instances required to split
        target_impurity=0.0 (double) target impurity at or below which node splitting halts
        
    Returns:
        a model (tuple of tuples) which represents a decision tree
        this model includes feature name, feature value threshold, 
        examples in split, majority class, impurity score, depth, left subtree,
        and right subtree
        
    """
    if train is None or len(train) == 0:
        return (None, None, None, None, None, None, None, None, None)
    
    else:
        # Determine the best column and value to split
        best_col, best_val, best_meas = best_split(train, 'Revenue', criterion)
        
        # Split the data by the determined column/value
        left_vals = train[train[best_col] <= best_val]
        right_vals = train[train[best_col] > best_val]
        
        # Count the instances of each genre in the data
        c = Counter(list(train['Revenue']))
        
        # Return the class with the most counts
        majority_class = c.most_common(1)[0][0]
        
        # If the number of unique genres is less than min instances, stop splitting, and return tree
        if train['Revenue'].nunique() < min_instances:
            return (best_col, best_val, len(train), majority_class, best_meas, depth, None, None, train)
        
        # Elif the score of the criterion is less than target_impurity, stop splitting, and return tree
        elif best_meas < target_impurity:
            return (best_col, best_val, len(train), majority_class, best_meas, depth, None, None, train)
        
        # Elif the max_depth is not none and the depth count is greater than max_depth, stop splitting, and return tree
        elif max_depth is not None and depth >= max_depth:
            return (best_col, best_val, len(train), majority_class,  best_meas, depth, None, None, train)
        
        # Else, add 1 to depth and recursively call the dtree function to split into left and right branches
        else:
            depth += 1
            return (dtree(left_vals, criterion, depth, max_depth, min_instances), dtree(right_vals, criterion, depth, max_depth, min_instances))


def unpack_tuple(nested_tuples):
    """
    Takes in nested tuples and returns a nested list of each individual element in nested tuples.
    """
    # Initiate an empty list
    result = []

    # Iterate over each item in nested_tuples, check if item is of type tuple, if it is,
    # unpack the item and add to tuple, else just add the item to the list.
    for item in nested_tuples:
        if isinstance(item, tuple):
            result.extend(unpack_tuple(item))
        else:
            result.append(item)

    # Return a list of the list
    return list(result)


def predict(model, data):
    """
    Make predictions for the data based on the model
    Parameters:
        model (tuple of tuples) a decision tree
        data (dataframe) the data to make predictions for

    Returns:
        preds, a series of predictions for the data
    """
    # Run the unpack_tuple function on the input model
    res = unpack_tuple(model)

    # Split result of the unpacked tuple into equal size chunks of 9 elements each
    chunks = [res[i * 9:(i + 1) * 9] for i in range((len(res) + 9 - 1) // 9)]

    # Initiate lists
    maj_classes = []
    indices = []

    # Iterate over each list in chunks and check to see if the 8th index element
    # of the list is of object type dataframe. If it is, set the element in the 3rd index of
    # the list as maj_class and append to maj_classes list, set the element in the 8th index of
    # the list as df and append the index of the df as a list into indices list.
    for lst in chunks:
        if isinstance(lst[8], pd.DataFrame):
            maj_class = lst[3]
            maj_classes.append(maj_class)

            df = lst[8]
            indices.append(df.index.tolist())

    # Make a copy of the data
    data_copy = data.copy()

    # Initiate a predictions column in data_copy df with None values
    data_copy["predictions"] = ["None"] * len(data_copy)

    # Populate the predictions with the majority classes
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            for k in range(len(maj_classes)):
                if i == k:
                    data_copy.loc[indices[i][j], "predictions"] = maj_classes[k]

    return data_copy["predictions"]


def split_folds(df, folds=10):
    """
    Takes in a dataframe and number of folds. Split df into n folds.
    Returns train_folds(list), valid_folds(list)
    """
    # Initiate list
    train_folds = []
    valid_folds = []

    # Iterate over each fold values, create train_fold and valid_fold and add to their corresponding lists
    for f in range(folds):
        train_fold = df[df.index % folds != f]
        train_folds.append(train_fold)
    
        valid_fold = df[df.index % folds == f]
        valid_folds.append(valid_fold)
    return train_folds, valid_folds


def find_splitting_criterion(data, class_col, crit_1, crit_2, max_depth):
    """
    Find the best splitting criterion for decision tree
    Parameters:
        df (dataframe)
        crit_1 (gini or entropy)
        crit_2 (gini or entropy)
        
    Returns:
        print statement of optimal criterion   
    """
    result_crit_1 = dtree(data, crit_1, max_depth=max_depth)
    preds_crit_1 = predict(result_crit_1, data)
    metrics_1 = ml.metrics(data[class_col], preds_crit_1)
    
    result_crit_2 = dtree(data, crit_2, max_depth=max_depth)
    preds_crit_2 = predict(result_crit_2, data)
    metrics_2 = ml.metrics(data[class_col], preds_crit_2)
    
    if metrics_1["accuracy"] > metrics_2["accuracy"]:
        print(crit_1, "is the optimal splitting criterion.")
    else:
        print(crit_2, "is the optimal splitting criterion.")
        
        
def find_maxd(data, class_col, criterion, max_depths):
    """
    Determine the best max_depth for the model.
    
    Parameters:
        df (dataframe)
        criterion (gini or entropy)
        max_depths (list of values)
        
    Returns:
        zipped list with depth value corresponding to 
        the accuracy for the data
    """
    # Initiate empty lists
    depth_lst = []
    acc_lst = []
    
    # For each max_depth value
    for max_depth in max_depths:
        
        # Add value to list
        depth_lst.append(max_depth)
        
        # Create a tree and run metrics
        tree = dtree(data, criterion, max_depth=max_depth)
        metrics = ml.metrics(data[class_col], predict(tree, data))
        
        # Add accuracy to list
        acc_lst.append(metrics["accuracy"])
        
    return list(zip(depth_lst, acc_lst))


def find_min_examples(data, class_col, criterion, max_depth, mins_instances):
    """
    Determine the best min_examples for the model.
    
    Parameters:
        df (dataframe)
        criterion (gini or entropy)
        min_instances (list of values)
        
    Returns:
        zipped list with min instance value corresponding to 
        the accuracy for the data

    """
    # Initiate empty lists
    min_instances_lst = []
    acc_lst = []
    
    # For each value
    for min_instances in mins_instances:
        
        # Add value to list
        min_instances_lst.append(min_instances)
        
        # Create a tree and run metrics
        tree = dtree(data, criterion, max_depth=max_depth, min_instances=min_instances)
        metrics = ml.metrics(data[class_col], predict(tree, data))
        
        # Add accuracy to list
        acc_lst.append(metrics["accuracy"])
        
    return list(zip(min_instances_lst, acc_lst))


def find_target_impurity(data, class_col, criterion, max_depth, min_instances, target_impurities):
    """
    Determine the best target_impurity for the model
    
    Parameters:
        df (dataframe)
        criterion (gini or entropy)
        target_impurities (list of values)
        
    Returns:
        zipped list with impurity value corresponding to 
        the accuracy for the data
    """
    # Initiate empty lists
    impurity_lst = []
    acc_lst = []
    
    # For each value in list
    for target_impurity in target_impurities:
        
        # Add value to list
        impurity_lst.append(target_impurity)
        
        # Create a tree and run metrics
        tree = dtree(data, criterion, max_depth=max_depth, min_instances=min_instances, target_impurity=target_impurity)
        metrics = ml.metrics(data[class_col], predict(tree, data))
        
        # Add accuracy to list
        acc_lst.append(metrics["accuracy"])
        
    return list(zip(impurity_lst, acc_lst))