"""
Luke Abbatessa
DATA Club: Storms Project
Scatterplot Showing Lat/Long Conversions for 2020
02.25.2022
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOC_2020 = "StormEvents_locations-ftp_v1.0_d2020_c20211019.csv"
DET_2020 = "StormEvents_details-ftp_v1.0_d2020_c20211019.csv"

LATMIN = 25
LATMAX = 50
LONGMIN = -125
LONGMAX = -65

WIDTH = 150
HEIGHT = 100

def long_to_col(long, long_max, long_min, width):
    """ Function: long_to_col
        Parameters: longitude point to convert,
                    max/min of long in map
                    width of grid
        Returns: int (col in grid)
    """
    ratio = (long - long_min) / (long_max - long_min)
    col = ratio * width
    return round(col)

def lat_to_row(lat, lat_max, lat_min, height):
    """ Function: lat_to_row
        Parameters: latitude point to convert,
                    max/min of lat in map
                    height of grid
        Returns: int (row in grid)
    """
    ratio = (lat - lat_min) / (lat_max - lat_min)
    row = ratio * height
    row = height - row
    return round(row)

def convert_storms_to_color(string):
    """ Function: convert_all_storms_to_color
        Parameters: string, a specific string in a dataframe
        Returns: a 3-valued RGB color NumPy array
    """
    if string == "Tornado":
        return [255, 0, 0]
    elif string == "Rip Current":
        return [0, 0, 255]
    elif string == "Hurricane":
        return [0, 255, 255]
    else:
        return [224, 224, 224]

if __name__ == "__main__":
    storm_loc = pd.read_csv(LOC_2020)
    print(storm_loc.head(5))
    
    storm_det = pd.read_csv(DET_2020)
    print(storm_det.head(5))
    
    storm_det["AVG_LAT"] = storm_det.apply(lambda row: (row.BEGIN_LAT + \
                           row.END_LAT) / 2, axis = 1)
    storm_det["AVG_LON"] = storm_det.apply(lambda row: (row.BEGIN_LON + \
                           row.END_LON) / 2, axis = 1)
    
    print(storm_det.head(5))
   
    grid = np.full((HEIGHT, WIDTH, 3), 255, dtype = int)
    
    storm_det_clean = storm_det.dropna(subset = ["AVG_LAT", "AVG_LON"])
    print("\nPossible Storm Types:", storm_det_clean["EVENT_TYPE"].unique(), "\n")
    
    for idx, row in storm_det_clean.iterrows():
        y = long_to_col(row["AVG_LON"], LONGMAX, LONGMIN, WIDTH)
        x = lat_to_row(row["AVG_LAT"], LATMAX, LATMIN, HEIGHT)
        if x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH:
            grid[x][y] = convert_storms_to_color(row["EVENT_TYPE"])
    
    plt.imshow(grid)
    
    
