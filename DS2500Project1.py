""" Luke Abbatessa
    DS2500
    Project #1
    February 27, 2022
    ocean_acid.py
    
    Consulted the National Ocean Service for more information regarding ocean acidification
    https://oceanservice.noaa.gov/facts/acidification.html
    Consulted GeeksforGeeks about filtering a Pandas dataframe by column values
    https://www.geeksforgeeks.org/ways-to-filter-pandas-dataframe-by-column-values/
    Consulted DataScience Made Simple about extracting n characters from a column in a Pandas dataframe
    https://www.datasciencemadesimple.com/return-first-n-character-from-left-of-column-in-pandas-python/
    Consulted w3schools for more information about the subplot() functionality of matplotlib
    https://www.w3schools.com/python/matplotlib_subplots.asp
    Consulted PythonGuides for changing the size of markers in a scatterplot in matplotlib
    https://pythonguides.com/matplotlib-scatter-marker/#:~:text=We%20can%20easily%20increase%20or%20decrease%20the%20size,marker%20is%20given%20below%3A%20matplotlib.pyplot.scatter%20%28x%2C%20y%2C%20s%3DNone%29
    Consulted stackoverflow for hiding or removing axes labels for a plot in seaborn
    https://stackoverflow.com/questions/58476654/how-to-remove-or-hide-x-axis-labels-from-a-seaborn-matplotlib-plot
    Consulted GeeksforGeeks for changing the marker size of points in a regression plot in seaborn
    https://www.geeksforgeeks.org/seaborn-regression-plots/
    Consulted Moonbooks for more information on the parameter in matplotlib.pyplot.scatter() that changes the color of the points of the plot
    https://moonbooks.org/Articles/How-to-create-a-scatter-plot-with-several-colors-in-matplotlib-/#:~:text=To%20change%20the%20color%20of%20a%20scatter%20point,scatter%20plot%20with%20several%20colors%20in%20matplotlib%20%3F
    Consulted matplotlib for the different colors offered in the matplotlib library
    https://matplotlib.org/stable/gallery/color/named_colors.html
    Consulted tutorialspoint for more information on changing the colors of the points and regression line in a seaborn regression plot
    https://www.tutorialspoint.com/how-to-show-different-colors-for-points-and-line-in-a-seaborn-regplot
    Consulted DelftStack for more information on changing the font size of titles and axes labels in matplotlib
    https://www.delftstack.com/howto/matplotlib/how-to-set-the-figure-title-and-axes-labels-font-size-in-matplotlib/#:~:text=set_size%20%28%29%20Method%20to%20Set%20Fontsize%20of%20Title,the%20title%2C%20x-axis%20label%20and%20y-axis%20label%20respectively.
    Consulted stackoverflow for spacing subplots in matplotlib
    https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
    Consulted GeeksforGeeks regarding the sizes of axes labels of seaborn regplots
    https://www.geeksforgeeks.org/change-axis-labels-set-title-and-figure-size-to-plots-with-seaborn/#:~:text=We%20make%20use%20of%20the%20set_title%20%28%29%2C%20set_xlabel,assign%20the%20axes-level%20object%20while%20creating%20the%20plot.
    Consulted stackoverflow for changing the sizes of axes tick labels for plots
    https://stackoverflow.com/questions/38369188/set-size-of-ticks-in-all-subplots#:~:text=If%20you%20want%20to%20change%20the%20tick%20size,current%20code%20as%20there%20is%20only%20one%20plot.
    Consulted DelftStack for adding an overall title to a group of subplots
    https://www.delftstack.com/howto/matplotlib/how-to-set-a-single-main-title-for-all-the-subplots-in-matplotlib/#:~:text=In%20this%20example%2C%20axes.set_title%20%28%29%20method%20is%20used,using%20various%20parameters%20to%20the%20plt.suptitle%20%28%29%20method.
    Consulted How to Data for changing the scales of tick marks for both axes of a plot
    https://nathancarter.github.io/how2data/site/how-to-change-axes-ticks-and-scale-in-a-plot-in-python-using-matplotlib/
    Consulted CodeSpeedy for adding one subplot at a time through iteration
    https://www.codespeedy.com/use-add_subplot-in-matplotlib/
    Consulted medium for a better idea of contraasting colors
    https://medium.com/@mreiner4/color-contrast-and-complementary-colors-make-successful-designs-178c619e8a65
    Consulted stackoverflow for more information on the use of plt.figure()
    https://stackoverflow.com/questions/31686530/matplotlib-generate-a-new-graph-in-a-new-window-for-subsequent-program-runs
    Consulted the U.S. Geological Survey for the confirmation of the legitimacy of negative alkalinity values
    https://or.water.usgs.gov/alk/reporting.html#:~:text=Hydroxide%2C%20carbonate%2C%20and%20bicarbonate%20concentrations%20cannot%20be%20negative.,the%20presence%20of%20some%20amount%20of%20mineral%20acidity.
    Consulted PYnative Python Programming for more information on the use of np.arange()
    https://pynative.com/python-range-for-float-numbers/
"""
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


OCEAN_ACID_DATA = "SpringsCoastCarbonateData.csv"

SITE_ID_LST = ["Weeki Wachee", "Chassahowitzka", "Homosassa", \
                      "Kings Bay", "Rainbow"]

def read_file(filename, header = True):
    """ Function: read_file
        Parameters: filename, a string
                    header, a boolean
        Returns: a dataframe
    """
    if header:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, header = None)
    return df

# Filter a dataframe by a value within one of its columns
def filter_df(df, col, val):
    """ Function: filter_df
        Parameters: df, a dataframe
                    col, a column in the dataframe 
                    val, a value from the column in the dataframe
        Returns: a dataframe with data filtered for one column value
    """
    df_filtered = df[(df[col] == val)]
    return df_filtered

# Group one or more columns of a dataframe by a single different column, 
# whereby the values of the former are grouped according to the values of the
# latter 
def group_df_cols_by_col(df, groupby_col, cols_of_interest, method):
    """ Function: group_df_cols_by_col
        Parameters: df, a dataframe
                    groupby_col, a column in the dataframe
                    cols_of_interest, a list of one or more columns in the 
                                      dataframe
                    method, the means of grouping the columns of the dataframe
        Returns: a dataframe with data from one or more columns grouped by 
                 another column
    """
    df_groupby = df.groupby(by = groupby_col)\
                 [cols_of_interest].agg(method).reset_index()
    return df_groupby

# Calculate the correlation between the values of two columns in a dataframe
def calculate_corr(df, col_x, col_y):
    """ Function: calculate_corr
        Parameters: df, a dataframe
                    col_x, a column in the dataframe representing x
                    col_y, a column in the dataframe representing y
        Returns: a correlation matrix
    """
    df_corr = df[[col_x, col_y]].corr()
    return df_corr

# Produce a linear regression model comparing two columns of a dataframe
def calculate_lin_reg(df, col_x, col_y):
    """ Function: calculate_lin_reg
        Parameters: df, a dataframe
                    col_x, a column in the dataframe representing x
                    col_y, a column in the dataframe representing y
        Returns: a linear regression [model]
    """
    lr = stats.linregress(x = df[col_x], y = df[col_y])
    return lr

# Predict the value of one column in a dataframe based on a linear regression 
# model that is used to predict said value, and simulate this process many 
# times over
def calculate_y_over_time(lr, lower_bound, upper_bound):
    """ Function: calculate_y_over_time
        Parameters: lr, a linear regression [model]
                    lower_bound, integer to signify first x value in prediction
                    upper_bound, integer to signify x value after last x value
                                 in prediction
        Returns: a list of (x, y) tuples
    """
    y_val = lambda x_val: int(x_val * lr.slope + lr.intercept)
    pred = [(x_val, y_val(x_val)) for x_val in range(lower_bound, upper_bound)]
    return pred
    

if __name__ == "__main__":
    # Read in the CSV file as a dataframe, and do some preliminary analysis
    ocean_acid_df = read_file(OCEAN_ACID_DATA)
    print("To get a sense of the dataframe, here are the first few rows.")
    print(ocean_acid_df.head(5))
    
    # Add a new column to the dataframe titled Year based on string splicing
    ocean_acid_df["Year"] = ocean_acid_df["Year_Season"].str[:4]
    print("Here is the same dataframe, but with a new column titled Year.")
    print(ocean_acid_df.head(5))
    
    # Filter the dataframe by the columns used for analysis, and then analyze
    # the dataframe
    ocean_acid_df = ocean_acid_df[["Site Id", "pH (Total) SU", \
                                   "Alkalinity (Total) umol/kg", \
                                   "Atmospheric CO2 (ppm)", "Year"]]
    print("Here is the same dataframe, filtered by the columns used for "
          "analysis.")
    print(ocean_acid_df.head(5))
    print("Here are the summary statistics of the dataframe.")
    print(ocean_acid_df.describe())
    
    
    # Lay the foundation for the subplots of the first figure
    fig_cotwo_ph = plt.figure()
    fig_cotwo_ph.subplots_adjust(hspace = 0.5)
    plt.suptitle("Changes in pH from Atmospheric CO2 in Five Floridian Spring "
                 "Study Sites", fontsize = 15)
        
    # Iterate over every Site Id within the list of Site Ids    
    for n, site_id in enumerate(SITE_ID_LST):
        # Filter the dataframe five-fold, each time by a different Site Id
        df_site_id = filter_df(ocean_acid_df, "Site Id", site_id)
        print("Here is the subdataframe representing the Florida spring " 
              "named", site_id + ".")
        print(df_site_id)
        
        # Group Atmospheric CO2 (ppm) and pH (Total) SU by Year in each of 
        # the five filtered dataframes, doing so by mean
        df_site_id_cotwo_ph = group_df_cols_by_col(df_site_id, "Year", \
                             ["Atmospheric CO2 (ppm)", "pH (Total) SU"], \
                             "mean")
        print("Here are the annual averages for Atmospheric CO2 (ppm) and "
              "pH (Total) SU by year for", site_id + ".")
        print(df_site_id_cotwo_ph)
        
        # Calculate the correlation between the same two variables for each 
        # Site Id
        corr_cotwo_ph = calculate_corr(df_site_id_cotwo_ph, \
                       "Atmospheric CO2 (ppm)", "pH (Total) SU")
        print("Here is the correlation matrix comparing Atmospheric CO2 (ppm) "
              "and pH (Total) SU for", site_id + ".")
        print(corr_cotwo_ph)
        
        # Reserve a subplot for each of the five Site Ids
        fig_cotwo_ph.add_subplot(1, 5, n + 1)
        # Plot a regression plot with a line of best fit comparing 
        # Atmospheric CO2 (ppm) and pH (Total) SU for each dataframe
        cotwo_ph_pred = sns.regplot(x = df_site_id_cotwo_ph\
                      ["Atmospheric CO2 (ppm)"], y = df_site_id_cotwo_ph\
                      ["pH (Total) SU"], \
                      scatter_kws = {"s" : 10, "color" : "green"}, \
                      line_kws = {"color" : "red"})
        plt.xticks(range(350, 425, 25))
        plt.tick_params(axis = "both", which = "major", labelsize = 8)
        
        # Modify the axes labels, tick scales, and titles of specific subplots
        if n == 0:
            cotwo_ph_pred.set_xlabel("Atmospheric CO2 (ppm)", size = 8)
            cotwo_ph_pred.set_ylabel("pH (Total) SU", size = 8)
            cotwo_ph_pred.set_yticks(np.arange(7.1, 7.7, 0.1))
            cotwo_ph_pred.set_title("Weeki Wachee", size = 10)
        
        if n == 1:
            cotwo_ph_pred.set(xlabel = None)
            cotwo_ph_pred.set(ylabel = None)
            cotwo_ph_pred.set_yticks(np.arange(7.05, 7.55, 0.05))
            cotwo_ph_pred.set_title("Chassahowitzka", size = 10)
        
        if n == 2:
            cotwo_ph_pred.set(xlabel = None)
            cotwo_ph_pred.set(ylabel = None)
            cotwo_ph_pred.set_yticks(np.arange(7.3, 7.65, 0.05))
            cotwo_ph_pred.set_title("Homosassa", size = 10)
        
        if n == 3:
            cotwo_ph_pred.set(xlabel = None)
            cotwo_ph_pred.set(ylabel = None)
            cotwo_ph_pred.set_yticks(np.arange(7.2, 8.0, 0.1))
            cotwo_ph_pred.set_title("Kings Bay", size = 10)
        
        if n == 4:
            cotwo_ph_pred.set(xlabel = None)
            cotwo_ph_pred.set(ylabel = None)
            cotwo_ph_pred.set_yticks(np.arange(7.4, 7.85, 0.05))
            cotwo_ph_pred.set_title("Rainbow", size = 10)
        
        # Produce a linear regression model comparing Atmospheric CO@ (ppm) and
        # pH (Total) SU for each dataframe
        cotwo_ph_lr = calculate_lin_reg(df_site_id_cotwo_ph, \
                     "Atmospheric CO2 (ppm)", "pH (Total) SU")
        print("Here are the slope and intercept of the linear regression "
              "model for", site_id, "comparing Atmospheric CO2 (ppm) "
              "to pH (Total) SU.") 
        
        # Print out the slope and intercept for each linear regression model
        print("Slope:", cotwo_ph_lr.slope, "SU/ppm")
        print("Intercept:", cotwo_ph_lr.intercept, "SU")
        
        # Predict the pH values of their respective bodies of water over a 
        # span of 100 years for each Site Id
        site_id_ph = calculate_y_over_time(cotwo_ph_lr, 1, 101)
        print("Here are the pH values for the water representative "
              "of", site_id, "spanning 100 years.")
        print(site_id_ph)
    
    # Modify the spacing of the subplots in the figure
    plt.tight_layout()
    
    
    # Lay the foundation for the subplots of the second figure
    fig_ph_alk = plt.figure()
    fig_ph_alk.subplots_adjust(hspace = 0.5)
    plt.suptitle("Changes in Alkalinity from pH in Five Floridian Spring "
                 "Study Sites", fontsize = 15)
    
    # Iterate over every Site Id within the list of Site Ids    
    for n, site_id in enumerate(SITE_ID_LST):
        # Filter the dataframe five-fold, each time by a different Site Id
        df_site_id = filter_df(ocean_acid_df, "Site Id", site_id)
        
        # Group pH (Total) SU and Alkalinity (Total) umol/kg by Year in each of 
        # the five filtered dataframes, doing so by mean
        df_site_id_ph_alk = group_df_cols_by_col(df_site_id, "Year", \
                             ["pH (Total) SU", "Alkalinity (Total) umol/kg"], \
                             "mean")
        print("Here are the annual averages for pH (Total) SU and "
              "Alkalinity (Total) umol/kg by year for", site_id + ".")
        print(df_site_id_ph_alk)
        
        # Calculate the correlation between the same two variables for each 
        # Site Id
        corr_ph_alk = calculate_corr(df_site_id_ph_alk, \
                       "pH (Total) SU", "Alkalinity (Total) umol/kg")
        print("Here is the correlation matrix comparing pH (Total) SU and "
              "Alkalinity (Total) umol/kg for", site_id + ".")
        print(corr_ph_alk)
        
        # Reserve a subplot for each of the five Site Ids
        fig_ph_alk.add_subplot(1, 5, n + 1)
        # Plot a regression plot with a line of best fit comparing pH 
        # (Total) SU and Alkalinity (Total) umol/kg for each dataframe
        ph_alk_pred = sns.regplot(x = df_site_id_ph_alk\
                      ["pH (Total) SU"], y = df_site_id_ph_alk\
                      ["Alkalinity (Total) umol/kg"], \
                      scatter_kws = {"s" : 10, "color" : "blue"}, \
                      line_kws = {"color" : "orange"})
        plt.tick_params(axis = "both", which = "major", labelsize = 8)
        
        # Modify the axes labels, tick scales, and titles of specific subplots
        if n == 0:
            ph_alk_pred.set_xlabel("pH (Total) SU", size = 8)
            ph_alk_pred.set_xticks(np.arange(7.1, 7.7, 0.3))
            ph_alk_pred.set_ylabel("Alkalinity (Total) umol/kg", size = 8)
            ph_alk_pred.set_yticks(range(2550, 2950, 50))
            ph_alk_pred.set_title("Weeki Wachee", size = 10)
                
        if n == 1:
            ph_alk_pred.set(xlabel = None)
            ph_alk_pred.set_xticks(np.arange(7.0, 7.75, 0.25))
            ph_alk_pred.set(ylabel = None)
            ph_alk_pred.set_yticks(range(2400, 3300, 100))
            ph_alk_pred.set_title("Chassahowitzka", size = 10)
        
        if n == 2:
            ph_alk_pred.set(xlabel = None)
            ph_alk_pred.set_xticks(np.arange(7.3, 7.6, 0.15)) 
            ph_alk_pred.set(ylabel = None)
            ph_alk_pred.set_yticks(range(2100, 2450, 50))
            ph_alk_pred.set_title("Homosassa", size = 10)
        
        if n == 3:
            ph_alk_pred.set(xlabel = None)
            ph_alk_pred.set_xticks(np.arange(7.2, 8.25, 0.35))
            ph_alk_pred.set(ylabel = None)
            ph_alk_pred.set_yticks(range(1800, 2600, 100))
            ph_alk_pred.set_title("Kings Bay", size = 10)
        
        if n == 4:
            ph_alk_pred.set(xlabel = None)
            ph_alk_pred.set_xticks(np.arange(7.4, 8.0, 0.2))
            ph_alk_pred.set(ylabel = None)
            ph_alk_pred.set_yticks(range(1400, 2400, 100))
            ph_alk_pred.set_title("Rainbow", size = 10)
            
        # Produce a linear regression model comparing pH (Total) SU and 
        # Alkalinity (Total) umol/kg for each dataframe
        ph_alk_lr = calculate_lin_reg(df_site_id_ph_alk, \
                     "pH (Total) SU", "Alkalinity (Total) umol/kg")
        print("Here are the slope and intercept of the linear regression "
              "model for", site_id, "comparing pH (Total) SU to "
              "Alkalinity (Total) umol/kg.") 
        
        # Print out the slope and intercept for each linear regression model
        print("Slope:", ph_alk_lr.slope, "umol/kg / SU")
        print("Intercept:", ph_alk_lr.intercept, "umol/kg")
        
        # Predict the alkalinity values of their respective bodies of water 
        # over a span of 100 years for each Site Id
        site_id_alk = calculate_y_over_time(ph_alk_lr, 1, 101)
        print("Here are the alkalinity values for the water representative "
              "of", site_id, "spanning 100 years.")
        print(site_id_alk)
    
    # Modify the spacing of the subplots in the figure
    plt.tight_layout()
        
        
    