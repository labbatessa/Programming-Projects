"""
    Luke Abbatessa
    DS4400
    Spring 2023
    mltools.py
"""

# Import the necessary libraries/packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from collections import Counter
import math


# HOMEWORK 1 FUNCTIONS


def calculate_mean_squared_error(response_vals, model_preds):
    """
      Parameters: response_vals - actual values of the response variable (a series)
                  model_preds - predictions for the response variable (a series)
      Returns: mean squared error (MSE) (a float)
      Does: Calculates MSE
    """
    actual = np.array(response_vals)
    pred = np.array(model_preds)
    diffs = np.subtract(actual, pred)
    sqrd_diffs = np.square(diffs)
    return sqrd_diffs.mean()


def calculate_mean_absolute_error(response_vals, model_preds):
    """
      Parameters: response_vals - actual values of the response variable (a series)
                  model_preds - predictions for the response variable (a series)
      Returns: mean absolute error (MAE) (a float)
      Does: Calculates MAE
    """
    actual = np.array(response_vals)
    pred = np.array(model_preds)
    diffs = np.subtract(actual, pred)
    abs_diffs = np.absolute(diffs)
    return abs_diffs.mean()


def create_surface_plot(x, y, z):
    """
      Parameters: x - values representative of the x axis of a surface plot (a 1D array)
                  y - values representative of the y axis of a surface plot (a 1D array)
                  z - values representative of the z axis of a surface plot (a 2D array)
      Returns: a surface plot
      Does: Generates a surface plot
    """
    # Create a meshgrid of the explanatory and response variables
    x, y = np.meshgrid(x, y)

    # Build the foundation for a 3d surface plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.viridis)

    # Set axes labels 
    ax.set_xlabel("x", labelpad=20)
    ax.set_ylabel("y", labelpad=20)
    ax.set_zlabel("z", labelpad=0)

    # Implement a colorbar for the surface plot
    fig.colorbar(surf, shrink=.5, aspect=8)

    plt.show()


def implement_random_step_search_algo(df, X, y, k=1000):
    """
      Parameters: df - a dataframe
                  X - a column within the dataframe representative of 
                      the explanatory variable (a string)
                  y - a column within the dataframe representative of 
                      the response variable (a string)
                  k - the number of random updates applied to 
                      the coefficients (an integer)
      Returns: a slope (a float), an intercept (a float),
               a list of the slopes generated over the range of random updates,
               a list of the intercepts generated over the range of random updates
      Does: Implements a random step search algorithm to find the best-fit linear model
            for an explanatory variable and a response variable
    """
    # Instantiate a tracker for the random changes
    n = 0

    # Select a slope and intercept randomly from arrays
    b0 = np.arange(-200, 200, 5)
    b1 = np.arange(-200, 200, 5)

    rand_b0 = np.random.choice(b0, size=1)
    rand_b1 = np.random.choice(b1, size=1)

    # Calculate ypred and MSE using the selected coefficients
    ypred = rand_b0 + (rand_b1 * df[X])
    mse = calculate_mean_squared_error(df[y], ypred)

    b0_lst = []
    b1_lst = []

    while n <= k:
        # Create an array of differently sized adjustments 
        # for the coefficients
        change_amts = np.arange(-1, 1, .025)

        # Initialize new coefficients using a randomly selected adjustment
        # and add the coefficients to lists
        new_b0 = rand_b0 + np.random.choice(change_amts, size=1)
        b0_lst.append(new_b0)

        new_b1 = rand_b1 + np.random.choice(change_amts, size=1)
        b1_lst.append(new_b1)

        # Calculate ypred and MSE using the adjusted coefficients
        new_ypred = new_b0 + (new_b1 * df[X])
        new_mse = calculate_mean_squared_error(df[y], new_ypred)

        # Keep the new values for the coefficients if MSE is reduced
        # otherwise keep the original values
        if new_mse < mse:
            n = 0
            rand_b0, rand_b1 = new_b0, new_b1
            mse = new_mse
        else:
            n += 1

    return rand_b0, rand_b1, b0_lst, b1_lst


def implement_gradient_descent(df, X, y, k=1000):
    """
      Parameters: df - a dataframe
                  X - a column within the dataframe representative of 
                      the explanatory variable (a string)
                  y - a column within the dataframe representative of 
                      the response variable (a string)
                  k - the number of iterations of the algorithm (an integer)
      Returns: a slope (a float), an intercept (a float),
               a list of the slopes generated over the range of iterations,
               a list of the intercepts generated over the range of iterations
      Does: Implements a gradient descent algorithm to find the best-fit linear model
            for an explanatory variable and a response variable
    """
    # Initialize a tracker for the iterations
    n = 0

    # Select a slope and intercept randomly from arrays
    b0 = np.arange(-200, 200, 5)
    b1 = np.arange(-200, 200, 5)

    rand_b0 = np.random.choice(b0, size=1)
    rand_b1 = np.random.choice(b1, size=1)

    # Calculate ypred and MSE using the selected coefficients
    ypred = rand_b0 + (rand_b1 * df[X])
    mse = calculate_mean_squared_error(df[y], ypred)

    b0_lst = []
    b1_lst = []

    while n <= k:
        alpha = 0.01
        # Implement the partial derivatives for the coefficients
        steps_times_diffs = alpha * np.sum((ypred - df[y])) / len(df[y])
        steps_times_diffs_times_x = alpha * np.array(df[X]).T.dot(ypred - df[y]) \
                                    / len(df[y])

        # Adjust the coefficients using the partial derivatives
        # and add the coefficients to lists
        new_b0 = rand_b0 - steps_times_diffs
        b0_lst.append(new_b0)

        new_b1 = rand_b1 - steps_times_diffs_times_x
        b1_lst.append(new_b1)

        # Calculate ypred and MSE using the adjusted coefficients
        new_ypred = new_b0 + (new_b1 * df[X])
        ypred = new_ypred
        new_mse = calculate_mean_squared_error(df[y], new_ypred)

        # Keep the new values for the coefficients
        n += 1
        rand_b0, rand_b1 = new_b0, new_b1
        mse = new_mse

    return rand_b0, rand_b1, b0_lst, b1_lst


def perform_linear_regression(df, X, y):
    """
      Parameters: df - a dataframe
                  X - a column within the dataframe representative of 
                      the explanatory variable (a string)
                  y - a column within the dataframe representative of 
                      the response variable (a string)
      Returns: a slope (a float), an intercept (a float)
      Does: Implements a scikit-learn linear regression model to find 
            the best-fit linear model for an explanatory variable and 
            a response variable
    """
    regr = lm.LinearRegression()
    regr.fit(df[[X]], df[y])
    y_intercept = regr.intercept_
    coef = regr.coef_
    return y_intercept, coef


def generate_mse_contour_plot(x, y, df, explanatory, response):
    """
      Parameters: x - values representative of the x axis of a contour plot (a 1D array)
                  y - values representative of the y axis of a contour plot (a 1D array) 
                  df - a dataframe
                  explanatory - a column within the dataframe representative of 
                      the explanatory variable (a string)
                  response - a column within the dataframe representative of 
                      the response variable (a string)
      Returns: a contour plot
      Does: Generates a contour plot
    """
    # Create a meshgrid of the explanatory and response variables
    X, Y = np.meshgrid(x, y)

    mses = np.array([])
    for i, ele_i in np.ndenumerate(x):
        for j, ele_j in np.ndenumerate(y):
            # Calculate ypred and MSE for all b0-b1 pairs
            ypred = ele_i + (ele_j * df[explanatory])
            mse = calculate_mean_squared_error(df[response], ypred)
            mses = np.append(mses, mse)
    # Reshape the array of MSE values to align with
    # the dimensions of the arrays for b0 and b1
    # and instantiate it as Z
    mses = np.reshape(mses, (len(x), len(x)))
    Z = mses

    # Build the foundation for a contour plot
    plt.figure(figsize=(8, 8))
    cp = plt.contour(X, Y, Z, 50)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())


# HOMEWORK 2 FUNCTIONS


def confusion_matrix(y, ypred):
    """ 
      Parameters: y = actual outcomes (0, 1, 2, ...)
                  ypred = predicted outcomes (0, 1, 2, ...)
      Returns: a confusion matrix as a numpy array
      Does: Generates a confusion matrix
    """

    # Find unique identifiers
    unique_classes = set(y) | set(ypred)
    n_classes = len(unique_classes)

    # Create matrix (all zeros)
    matrix = np.zeros(shape=(n_classes, n_classes), dtype=int)

    # Pair up each actual outcome with the corresponding prediction
    actual_prediction = list(zip(y, ypred))

    # For each pair, increment the correct position in the matrix
    for i, j in actual_prediction:
        matrix[i, j] += 1

    return matrix


def metrics(y, ypred):
    """ 
      Parameters: y = actual outcomes (0, 1, 2, ...)
                  ypred = predicted outcomes (0, 1, 2, ...)
      Returns: model accuracy, sensitivity, specificity, precision, and f1-score 
      Does: Generates accuracy scores for classifier
    """

    scores = {}
    C = confusion_matrix(y, ypred)

    scores["accuracy"] = C.diagonal().sum() / C.sum()

    # Implement scores for binary classification
    if C.shape == (2, 2):
        TN, FP, FN, TP = C.ravel()
        scores["sensitivity"] = TP / (TP + FN)
        scores["specificity"] = TN / (TN + FP)
        scores["precision"] = TP / (TP + FP)
        scores["f1-score"] = 2 * ((scores["precision"] * scores["sensitivity"])
                                  / (scores["precision"] + scores["sensitivity"]))
    else:
        pass

    return scores


def predict(X, w):
    """ 
      Parameters: X = 2-dimensional array of attribute values
                  w = bias and coefficients (weights) of the model
      Returns: scores for separating classes into categories,
               predictions 
      Does: Generates classification scores and predictions
    """

    score = w[0] + X.dot(w[1:])
    ypred = (score >= 0).astype(int)
    return score, ypred


def perceptron(data, alpha=0.0001, epochs=1000):
    """ 
      Parameters: data = a data table
                  alpha = the learning rate
                  epochs - the number of iterations
      Returns: a vector of weights, 
               a list of Mean Perceptron Errors (MPEs),
               a list of the epoch numbers,
               a list of model accuracies
      Does: Generates weights, MPEs, epochs, and accuracies
    """

    # Extract the attributes and labels
    X = data[data.columns[:-1]]
    X = X.to_numpy()
    y = data[data.columns[-1]]

    # Initialize a tracker for the epochs
    n = 0

    # Select weights randomly from arrays
    weights = np.random.uniform(size=len(X[0]) + 1, low=-1, high=1)

    # Calculate ypred and MPE using the selected weights
    score, ypred = predict(X, weights)
    error = y - ypred
    mpe = error.mean()

    mpe_lst = []
    epoch_lst = []
    accuracy_lst = []

    while n <= epochs:
        # Adjust the weights
        weights[0] += alpha * np.sum(y - ypred)

        for i, ele_i in np.ndenumerate(weights[1:]):
            for num, col in enumerate(X.T):
                if i == num + 1:
                    weights[i] += np.sum(alpha * (y - ypred) * col)

        # Calculate ypred and MPE using the adjusted weights
        new_score, new_ypred = predict(X, weights)
        new_error = y - new_ypred
        new_mpe = new_error.mean()

        mpe_lst.append(mpe)
        epoch_lst.append(n)
        five_metrics = metrics(y, ypred)
        accuracy_lst.append(five_metrics["accuracy"])

        # Keep the new values for ypred and MPE
        score, ypred = new_score, new_ypred
        error = new_error

        n += 1
        mpe = new_mpe

    return weights, mpe_lst, epoch_lst, accuracy_lst


def perceptron_scikit(data):
    """ 
      Parameters: data = a data table
      Returns: a vector of weights, 
               predicted outcomes
      Does: Generates weights and predictions
    """

    X = data[data.columns[:-1]]
    y = data.iloc[:, -1]

    clf = lm.Perceptron()
    clf.fit(X, y)

    w = list(clf.intercept_) + list(clf.coef_[0])

    return w, clf.predict(X)


# HOMEWORK 3 FUNCTIONS


def create_word_lst(df, col):
    """
      Parameters: df - a dataframe
                  col - a dataframe column (a string)
      Returns: a list of all the words in the column
      Does: Generates a list of every word in the dataframe column
    """
    word_lsts = []
    total_word_lst = []

    # Generate a 2D list of the column's instances
    for sent in df[col].values:
        word_lst = sent.split()
        word_lsts.append(word_lst)

    # Generate a list of all the words in the column
    for word_lst in word_lsts:
        for word in word_lst:
            total_word_lst.append(word)

    return total_word_lst


def calculate_word_counts(word_lst, df, col):
    """
      Parameters: word_lst - a list of words
                  df - a dataframe
                  col - a dataframe column (a string)
      Returns: a list of the counts of each word in the list of words
      Does: Counts the number of column values that contain each word
    """

    word_counts = []

    for i in range(len(word_lst)):
        word_count = len(df[df[col].str.contains(word_lst[i])])
        word_counts.append(word_count)

    return word_counts


def create_word_prob_dct(speaker_word_lst, total_word_lst, num_words_speaker):
    """
      Parameters: speaker_word_lst - a list of words tied to a specific speaker/class
                  total_word_lst - a list of all the words, regardless of the speaker/class
                  num_words_speaker - the number of words tied to a specific speaker/class
      Returns: a dictionary with words as keys and probabilities as values
      Does: Creates a dictionary with every word from total_word_lst as the keys and 
            the probability of the word occurring in a speaker/class sentence as the values
    """
    speaker_word_prob_dct = {}

    speaker_counter = Counter(speaker_word_lst)

    # Calculate the probability of a word given a speaker
    for word in total_word_lst:
        speaker_word_prob_dct[word] = (speaker_counter[word] + 1) / num_words_speaker

    return speaker_word_prob_dct


def generate_naive_bayes_scores(df, col, speaker_word_prob_dct):
    """
      Parameters: df - a dataframe
                  col - a dataframe column (a string)
                  speaker_word_prob_dct - a dictionary with words as keys 
                                          and probabilities as values
      Returns: a list of Naïve Bayes scores for a specific speaker/class
      Does: Generates a list of NB scores of all sentences, for a speaker/class
    """
    nb_scores_speaker = []

    word_lsts = []

    for sent in df[col].values:
        word_lst = sent.split()
        word_lsts.append(word_lst)

    for word_lst in word_lsts:
        log_probs = [math.log10(speaker_word_prob_dct[word]) for word in word_lst]
        nb_score = sum(log_probs) + math.log10(0.50)
        nb_scores_speaker.append(nb_score)

    return nb_scores_speaker


def create_word_vector(words, word_list, use_frequency=False):
    """ 
      Parameters: words - a list of words which we convert to a vector
                  word_list - the chosen words against which we compare
                  use_frequency - if False, vector components are 1/0, else n = # of occurrences
      Returns: a vector containing numeric values, each of which represent a sentence
      Does: Converts a list of words to a vector by comparing with words in word_list
    """

    word_list = sorted(list(set(word_list)))

    if use_frequency:
        count = Counter(words)
        return [count[w] for w in word_list]
    else:
        return [int(w in words) for w in word_list]


def euclidean(x, y):
    """ 
      Parameters: x - a vector of numeric values representing a sentence
                  y - a vector of numeric values representing a separate sentence
      Returns: the Euclidean Distance between the two sentences
      Does: Takes two vectorized sentences and finds the Euclidean Distance between them
    """
    return np.sqrt(sum((x - y) ** 2))


def manhattan(x, y):
    """ 
      Parameters: x - a vector of numeric values representing a sentence
                  y - a vector of numeric values representing a separate sentence
      Returns: the Manhattan Distance between the two sentences
      Does: Takes two vectorized sentences and finds the Manhattan Distance between them
    """
    return sum(np.abs(x - y))


def cossim(x, y):
    """ 
      Parameters: x - a vector of numeric values representing a sentence
                  y - a vector of numeric values representing a separate sentence
      Returns: the Cosine Similarity between the two sentences
      Does: Takes two vectorized sentences and finds the Cosine Similarity between them
    """
    magx = np.sqrt(np.dot(x, x))
    magy = np.sqrt(np.dot(y, y))
    return np.dot(x, y) / (magx * magy)


def hamming(x, y):
    """ 
      Parameters: x - a vector of numeric values representing a sentence
                  y - a vector of numeric values representing a separate sentence
      Returns: the Hamming Distance between the two sentences
      Does: Takes two vectorized sentences and finds the Hamming Distance between them
    """
    return np.logical_xor(x, y).sum()


def sim_matrix(A, f):
    """
      Parameters: A - array of instance attributes 
                  f - similarity / distance measure
      Returns: a similarity matrix
      Does: Computes similarity matrix 
    """
    m = A.shape[0]
    M = np.zeros(shape=(m, m))
    for i in range(m):
        for j in range(m):
            M[i, j] = f(A[i,], A[j,])

    return M


def implement_n_fold_cross_validation(A, measure, sentences, df, k=1):
    """ 
      Parameters: A - array of instance attributes
                  measure - a distance measure (a function)
                  sentences - a 2D list of classifiers for kNN
                  df - a dataframe
                  k - the number of nearest neighbors (an integer)
      Returns: the classes of each classifier, the average accuracy across all test sets
      Does: Performs n-fold cross validation
    """

    winners = []
    accuracies = []

    # Compute the similarity matrix
    M = sim_matrix(A, measure)

    for i in range(len(sentences)):
        # Get the similarity scores and indexes
        sims = list(zip(M[i], range(M.shape[0])))

        # Sort the similarity score-index tuples depending on the distance measure
        # Focus only on the nearest k
        sims = sorted(sims, reverse=True)[1:k + 1] if measure == cossim \
            else sorted(sims)[1:k + 1]

        # Extract the indexes
        nearest = [idx for sim, idx in sims]

        # Identify the majority class and append it to a list
        vote = Counter(df.iloc[nearest, :].outcome)
        winner = vote.most_common(1)[0][0]
        winners.append(winner)

        # Attain the test set accuracy
        # Append the accuracy to a list
        five_metrics = metrics(df["outcome"], winners)
        accuracy = five_metrics["accuracy"]
        accuracies.append(accuracy)

    # Take the average of the accuracies
    avg_accuracy = sum(accuracies) / len(accuracies)

    return winners, avg_accuracy


def implement_k_nearest_neighbor_learning(A,
                                          measure,
                                          sentences,
                                          df,
                                          k_values=list(range(1, 33, 2))):
    """ 
      Parameters: A - array of instance attributes
                  measure - a distance measure (a function)
                  sentences - a 2D list of classifiers for kNN
                  df - a dataframe
                  k_values - a range of nearest neighbors (a list)
      Returns: the average accuracies across all k values
      Does: Performs n-fold cross validation using a range of k values
    """
    avg_accuracies = []

    for k_value in k_values:
        winners, avg_accuracy = implement_n_fold_cross_validation(A,
                                                                  measure,
                                                                  sentences,
                                                                  df,
                                                                  k_value)
        avg_accuracies.append(avg_accuracy)

    return avg_accuracies


def plot_accuracy_vs_k(avg_accuracies,
                       title,
                       k_values=list(range(1, 33, 2)),
                       marker="x"):
    """ 
      Parameters: avg_accuracies - average accuracies across a range of k values (a list)
                  title - the title of the plot (a string)
                  k_values - a range of nearest neighbors (a list)
                  marker - the symbol for the points on the plot (a string)
      Returns: Nothing, just renders a plot
      Does: Visualizes the average accuracy for n-fold cross validation 
            against different values of k
    """
    plt.scatter(k_values, avg_accuracies, marker=marker)

    plt.plot(k_values, avg_accuracies)

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.show()
