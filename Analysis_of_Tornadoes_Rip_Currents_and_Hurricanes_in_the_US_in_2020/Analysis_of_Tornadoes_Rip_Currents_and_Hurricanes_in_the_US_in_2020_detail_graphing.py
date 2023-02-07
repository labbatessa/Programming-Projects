# -*- coding: utf-8 -*-
"""
Ciara Malamug
Data Club: Storms
11.20.2021
Chunking the Data: Details

Referenced:
    https://stackoverflow.com/questions/22258491/read-a-small-random-sample-from-a-big-csv-file-into-a-python-data-frame
    https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
    https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    
Focus on states specified in the areas at the beginning of project

Damage rates across storm types and regions, midwest/south to great lakes
"""
import csv
import random
import matplotlib.pyplot as plt
import pandas as pd

DETAIL_2020 = "StormEvents_details-ftp_v1.0_d2020_c20211019.csv"
DETAIL_2021 = "StormEvents_details-ftp_v1.0_d2021_c20211019.csv"

def read_csv_rand(filename, rand_num=1000):
    """
    Reads in the csv and returns a random ~1000 rows in a list

    Parameters
    ----------
    filename (str): name of the file to read in

    Returns
    -------
    headers (list): list of strings of the header names
    rand (list): list of list of random 1000 rows from the file

    """
    # Opens the file and uses a csv reader to read it in
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        # Saves the headers
        headers = next(csv_reader)
        
        # Creates a list to be returned
        rand = []
        
        # Finds the number of rows in the file
        n = sum(1 for line in open(filename)) - 1 
        
        # Goes through the entire file row by row
        for row in csv_reader:
            if random.randint(0, n) in range(rand_num):
                rand.append(row)
                
        return headers, rand           

def read_csv_rand_pd(filename, rand_num=1000):
    """
    Reads in the csv using pandas (from stack overflow)
    
    Parameters
    ----------
    filename (str): name of the file to read in
    
    Returns
    -------
    rand_df (dataframe): dataframe or random 1000 rows from the file
    headers (list): list of strings of the header names
    
    """
    # number of records in file (excludes header)
    n = sum(1 for line in open(filename)) - 1 
    
    # desired sample size
    s = rand_num 
    
    # the 0-indexed header will not be included in the skip list (aka keeps them)
    skip = sorted(random.sample(range(1,n+1),n-s)) 
    rand_df = pd.read_csv(filename, skiprows=skip)
    
    return rand_df

def graph_event_by_state(storm_df, state):
    """
    Takes in the state wanted to graph, and graphs the num of occurances of 
    each event

    Parameters
    ----------
    storm_df (dataframe): a dataframe containing all the storm data
    state (str): the name of the state we want to plot (full name all caps)
    
    Returns
    -------
    None. Plots a graph
    
    Overwrite with one that takes a list

    """
    state_df = storm_df[storm_df.STATE == state]
    
    events_dict = {}
    
    events = state_df["EVENT_TYPE"].unique()
    
    for event in events:
        
        state_event_df = state_df[state_df.EVENT_TYPE == event]
        
        events_dict[event] = state_event_df.shape[0]
        
    # Sorts the dictionary by value for nicer visuals when graphing
    events_dict = dict(sorted(events_dict.items(), key=lambda item: item[1]))
    
    plt.figure()    
    
    plt.bar(events_dict.keys(), events_dict.values())
    
    plt.title(f"Occurance of Storm Events in {state}")
    
    plt.xlabel("Event Type")
    
    plt.ylabel("Frequency")
    
    plt.xticks(rotation=90)
    
def events_damage(storm_df, damage_type):
    """
    Returns a smaller dataframe containing only the events with property damage

    Parameters
    ----------
    storm_df (dataframe): a dataframe containing storm data
    damage_type (string): the name of the column-- either DAMAGE_PROPERTY or
                            DAMAGE_CROPS

    Returns
    -------
    damage_df (dataframe): a dataframe with only the events that caused damage

    """
    damage_df = storm_df
    
    # Removes the K representing (3 zeros - 2) in the data and M (6 - 2), ect.
    # Already has 2 0s after a decimal point
    letter_num = {"K": "0"*(3-2),"M": "0"*(6-2), "B":"0"*(9-2), "." : ""}
    
    damage_df[damage_type] = damage_df[damage_type].str.translate(
        str.maketrans(letter_num))
    
    damage_df[damage_type] = pd.to_numeric(storm_df[damage_type])
    
    damage_df = damage_df[damage_df.DAMAGE_PROPERTY > 0]
    
    return damage_df

def graph_damage(df, damage_type):
    """
    Graphs the amount of damage per storm type

    Parameters
    ----------
    df (dataframe): a dataframe containing the storms data
    damage_type (string): the name of the column-- either DAMAGE_PROPERTY or
                            DAMAGE_CROPS

    Returns
    -------
    None. Plots a Graph

    """
    # Creates a dataframe contining only the events with property damage
    damage_df = events_damage(df, damage_type)
    
    events_dict = {}
    
    events = damage_df["EVENT_TYPE"].unique()
    
    for event in events:
        
        event_damage_df = damage_df[damage_df.EVENT_TYPE == event]
        
        events_dict[event] = event_damage_df[damage_type].sum()
    
    # Sorts the dictionary by value for nicer visuals when graphing
    events_dict = dict(sorted(events_dict.items(), key=lambda item: item[1]))
    
    plt.figure()    
    
    plt.bar(events_dict.keys(), events_dict.values())
    
    plt.title("{} Damage per Event Type".format(damage_type.split("_")[1]))
    
    plt.xlabel("Event Type")
    
    plt.ylabel("Damage ($)")
    
    plt.xticks(rotation=90)

def event_locations(df, event):
    """
    Plots the top 10 locations for tornados

    Parameters
    ----------
    df (dataframe): a dataframe containing the storm data
    event (string): the name of the event we want to look at

    Returns
    -------
    None. Plots a graph.

    """
    event_df = df[df.EVENT_TYPE == event]
    location_list = event_df["STATE"].unique()
    
    event_dict = {}
    
    for state in location_list:
        event_dict[state] = event_df[event_df.STATE == state].shape[0]
      
    # Sorts the dictionary by value for nicer visuals when graphing
    event_dict = dict(sorted(event_dict.items(), key=lambda item: item[1]))
    
    plt.figure()
    
    plt.bar(event_dict.keys(), event_dict.values())
    
    plt.title(f"Number of {event}s by State")
    
    plt.xlabel("State")
    
    plt.ylabel("Frequency")
    
    plt.xticks(rotation=90)
    
if __name__ == "__main__":
    # So far rand_num has to be less than 100,000
    df = read_csv_rand_pd(DETAIL_2020, rand_num=10000)
    
    # Would be more useful when we know what states we want to target
    # graph_event_by_state(df, "NEW YORK")
    # graph_event_by_state(df, "MASSACHUSETTS")
    
    # Plots the property damage by each event
    # Hurricanes are consistanly causing the most property damage
    # - Is especially sensitive to outliers as we are adding up all of the 
    #   damage
    graph_damage(df, "DAMAGE_PROPERTY")
    
    # Inconsistant top causes of crop damage
    graph_damage(df, "DAMAGE_CROPS")
    
    # Because the number of tornados in the sample is small, we get 
    # inconsistant sets of top tornado occurances by state for each random 
    # sample. Even if we do one consistant seed, it still indicates problems.
    # - Problems being my code not excluding major outliers
    event_locations(df, "Tornado")
    
    # Low frequency of hurricanes, with Louisiana consistantly being the most
    event_locations(df, "Hurricane")

