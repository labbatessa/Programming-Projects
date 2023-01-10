"""
    Andy Babb and Luke Abbatessa
    DS2500
    Project #2
    April 16, 2022
    boston_housing.py

    Consulted BTechGeeks for dropping the first column of a dataframe
    https://btechgeeks.com/pandas-delete-first-column-of-dataframe-in-python/#:~:text=del%20keyword%20in%20python%20is%20used%20to%20delete,easy%20to%20delete%20it%20using%20the%20del%20keyword.
    Consulted Statology for how to iterate over the columns in a dataframe
    https://www.statology.org/pandas-iterate-over-dataframe-columns/
    Consulted The Python You Need for editing the values of specific columns in a dataframe
    https://thepythonyouneed.com/how-to-edit-a-dataframe-column-with-pandas/
    Consulted Listen Data for dropping columns from a dataframe
    https://www.listendata.com/2019/06/pandas-drop-columns-from-dataframe.html
    Consulted pandas for the documentation for pandas.DataFrame.dropna
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
    Consulted seaborn for the documentation for seaborn.heatmap
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    Consulted Indian AI Production for more info on the documentation for seaborn.heatmap
    https://indianaiproduction.com/seaborn-heatmap/#13-seaborn-heatmap-annot-parameter----add-a-number-on-each-cell-
"""
# Import the necessary libraries
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA = "Boston.csv"

def read_file(filename, header = True):
    """ Function: read_file
        Parameters: filename, the name of a file (a string)
                    header, indicative of the presence or lack thereof of 
                            a header in the file (a boolean)
        Returns: a dataframe
    """
    if header:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, header = None)
    return df

def create_classes(value):
    """ Function: create_classes
        Parameters: value, a median housing value (a float)
        Returns: an integer representing which class the median housing value
                 belongs to
    """
    # Separate the data into classes based on ranges of median housing values
    if value <= 13.5:
        return 0
    elif value <= 27:
        return 1
    else:
        return 2
    
def clean_data(df):
    """ Function: clean_data
        Parameters: df, a dataframe
        Returns: the same dataframe but with columns dropped and added and 
                 modifications to the values of certain columns
    """
    # Drop the first column of the dataframe
    del df[df.columns[0]]
    
    # Convert the values of certain columns to unit-aligning magnitudes
    for name, values in df.iteritems():
        if name == "nox":
            df["nox_ppm"] = df["nox"] / 10
        elif name == "tax":
            df["tax_rate"] = df["tax"] / 10000
        elif name == "medv":
            df["medv_thous"] = df["medv"] * 1000
    
    # Initialize a column with the three classes for median housing value
    df["range"] = df.apply(lambda x: create_classes(x["medv"]), axis = 1)
    # Drop certain columns, as well as any rows with NA values
    df = df.drop(["nox", "tax", "medv"], axis = 1)
    df = df.dropna(axis = 0, how = "any")
    return df

def create_heatmap(df):
    """ Function: create_heatmap
        Parameters: df, a dataframe
        Returns: nothing, just renders a plot
    """
    # Plot a heatmap visualizing all correlations present in the data
    annot_kws = {"fontsize" : 5.5}
    sns.heatmap(data = df.corr(), vmin = 0, vmax = 1, annot = False, \
                annot_kws = annot_kws, linewidths = 0.5, cbar = True, 
                square = True)
        
def linear_regression(col, df, title):
    """ Function: linear_regression
        Parameters: col, the column being used as the independent variable 
                         (a string)
                    df, the full housing data (a dataframe)
                    title, the title of the plot (a string)
        Returns: a plot of a regression with 95% confidence interval, as well 
                 as the linear_regression object
    """
    # Calculate and visualize a linear regression between median housing value
    # and any other column
    plt.figure()
    lr = stats.linregress(x = df[col], y = df["medv"])
    #print(f"slope: {lr.slope}")
    #print(f"intercept: {lr.intercept}")
    #print(f"r-squared: {lr.rvalue ** 2}")
    housing_val_pred = sns.regplot(x = df[col], y = df["medv"])
    housing_val_pred.set_title(title)
    return lr

def prediction(cols, values, df):
    """ Function: prediction
        Parameters: cols, list of the columns being used to predict (a list)
                    values, in order of columns, the predictive values being 
                            used (a list)
                    df, the dataframe of the full housing data (a dataframe)
        Returns: an estimated median home value based on the predictive model
    """
    value_list = []
    # Iterate through a list of columns and create linear regressions with each
    # column and the median housing value column
    for i in range(len(cols)):
        lr = linear_regression(cols[i], df, None)
        # Generate a predicted median housing value based on a predictive value
        # associated with each column and append it to a list
        value = values[i] * lr.slope + lr.intercept
        value_list.append(value)
    
    # Calculate the average predicted median housing value
    avg_med_value = sum(value_list) / len(value_list)
    return avg_med_value

def knn_classifier(X_train, y_train, X_test, y_test, K):
    """ Function: knn_classifier
        Parameters: X_train and y_train, training data
                    X_test and y_test, testing data
                    K, number of neighbors
        Returns: an accuracy score
    """
    # Implement and fit a KNN classifier
    knn = KNeighborsClassifier(K)
    knn.fit(X_train, y_train.values.ravel())
    
    # Predict the targets of the test set
    predicted = knn.predict(X_test)
    correct = 0
    # Generate an accuracy score based on correctly predicted classes
    for i in range(len(predicted)):
        if predicted[i] == 1:
            correct += 1
    return correct / len(predicted)
    
def main():
    # Read in the CSV file and do some preliminary analysis of the data
    boston_df = read_file("Boston.csv")
    print(boston_df.head(5))
    print(boston_df.dtypes)
    
    # Clean the dataframe
    boston_df_clean = clean_data(boston_df)
    print(boston_df_clean.head(5))
    print(boston_df_clean.shape)
    
    # Create the heatmap visualizing all correlations present in the data
    create_heatmap(boston_df_clean)
    
    # Create linear regressions for the three most significant factors of 
    # median housing value
    linear_regression("rm", boston_df, \
                      "Relationship Between Rooms Per Dwelling and Median "
                      "Housing Value")
    
    linear_regression("zn", boston_df, \
                      "Relationship Between Land Zoned for Lots and Median "
                      "Housing Value")
    
    linear_regression("black", boston_df, \
                      "Relationship Between Black Population Proportion and "
                      "Median Housing Value")
    
    all_list = []
    other_list = []
    curated_list = []
    for i in range(100):
        # Compare all columns in the dataframe besides that of the classes to 
        # the column with the classes, separate the data into training and 
        # testing sets, and generate a list of accuracy scores
        X = boston_df.copy()
        X = X.drop(columns = "range")
        y = boston_df["range"]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        all_list.append(knn_classifier(X_train, y_train, X_test, y_test, 10))
        
        # Repeat the process detailed above, except this time with all columns
        # except the column with classes and the columns with the variables 
        # most significant to median housing value
        X = boston_df.copy()
        X = X.drop(columns = ["range", "black", "rm", "zn"])
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        other_list.append(knn_classifier(X_train, y_train, X_test, y_test, 10))
        
        # Repeat the proces again, except this time just for the columns with 
        # the variables most significant to median housing value
        X = boston_df[["black", "zn", "rm"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        curated_list.append(knn_classifier(X_train, y_train, X_test, y_test, \
                                           10))
    # Calculate the average accuracy score for each list of scores
    all_avg = sum(all_list) / len(all_list)
    other_avg = sum(other_list) / len(other_list)
    curated_avg = sum(curated_list) / len(curated_list)
    
    # Determine which classifer generates the highest average accuracy score
    if all_avg == max(curated_avg, all_avg, other_avg):
        best = "all"
    elif curated_avg == max(curated_avg, all_avg, other_avg):
        best = "curated"
    else:
        best = "other"
    # Tried to limit the code width of this statement, but the output got
    # messed up whenever I tried to do so
    print(f"all average is {all_avg}\ncurated average is {curated_avg}\nother average is {other_avg}")
    print(f"better performance comes from {best}")

if __name__ == "__main__":
    main()
