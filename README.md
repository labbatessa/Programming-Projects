# Programming-Projects
This repository includes the folders/files representative of the programming projects I have completed thus far

Analysis_of_Tornadoes_Rip_Currents_and_Hurricanes_in_the_US_in_2020 - folder that contains four things: 
1) Analysis_of_Tornadoes_Rip_Currents_and_Hurricanes_in_the_US_in_2020_Lat_Long_Viz.py - a .py file (Python file) that holds a visualization illustrating the latitudes and longitudes of tornadoes in the United States in the year 2020 (libraries used: pandas, numpy, matplotlib)
2) Analysis_of_Tornadoes_Rip_Currents_and_Hurricanes_in_the_US_in_2020_Statistical_Tests.Rmd - a .Rmd file (R Markdown file) that includes data cleaning steps (i.e. creating subsets, omitting null values, performing log transformations) and a plethora of statistical tests (i.e. Shapiro-Wilk tests, Kruskal-Wallis tests, Mann-Whitney U tests, Spearman rank correlation tests, G-tests) conducted based on the questions posed for the project
3) Analysis_of_Tornadoes_Rip_Currents_and_Hurricanes_in_the_US_in_2020_detail_graphing.py - a .py file that holds visualizations illustrating the number of storm events by state in the US, the frequency of storm events by month, and the property damage per event type (libraries used: csv, random, matplotlib, pandas)
4) DATA Club Snowball Project - Luke Abbatessa & Ciara Malamug.pptx - a .pptx file that represents a slideshow presentation detailing the specifics of the project

***This project involved the collection, analysis, visualization, & interpretation of data on previous locations, meteorological patterns, & economic damage rates of tornadoes, rip currents, & hurricanes in the US in 2020 to aid preparation for these events in the future***

Analyzing_the_Relationships_Between_Median_Housing_Value_and_its_Possible_Influences_in_Boston_in_the_Late_1900s - folder that contains two things:
1) Analyzing_the_Relationships_Between_Median_Housing_Value_and_its_Possible_Influences_in_Boston_in_the_Late_1900s.py - a .py file that includes data cleaning steps (i.e. dropping columns of a dataframe, changing the magnitudes of column values, creating a new column), the creation of a heatmap, and multiple machine learning models (i.e. linear regression, prediction of median home value based on influencing factors, knn classification) (libraries used: scipy, seaborn, matplotlib, sklearn)
2) Project Slides - Project #2 - DS2500 - Luke Abbatessa & Andy Babb - Northeastern University.pptx - a .pptx file that represents a slideshow presentation detailing the specifics of the project

***This project analyzed the relationships between median housing value and its possible influences in Boston in the late 1900’s by looking at factors such as crime, whether an area was primarily residential or not, the average size of a home, and others***

Investigating_the_Relationship_Between_Spreads_of_NFL_Games_and_NFL_Game_Type - folder that contains two things:
1) Final Project Presentation - DS2001 - Luke Abbatessa & John McCarthy - Northeastern University.pptx - a .pptx file that represents a slideshow presentation detailing the specifics of the project
2) Investigating_the_Relationship_Between_Spreads_of_NFL_Games_and_NFL_Game_Type.ipynb - a .ipynb file (Jupyter Notebook) that includes data cleaning steps (i.e. removing null values, converting values to floats), implementation of a t-test, and visualizations (e.g. bar chart, histograms) (libraries used: csv, statistics, math, matplotlib, google)

***This project investigated the relationship between spreads of NFL games and NFL game type, and it rejected the null hypothesis that the average spread for regular season games is equal to the average spread for playoff games***

Ocean_acidification_in_west_central_Florida - folder that contains two things:
1) Ocean_acidification_in_west_central_Florida.py - a .py file that includes data cleaning steps (i.e. adding a column to a dataframe, filtering values from a dataframe, grouping multiple columns by a single separate column), calculating correlation coefficients, implementing linear regression, visualizing regression plots as subplots, and predicting a response variable over time (libraries used: pandas, seaborn, scipy, matplotlib, numpy)
2) Project Slides - Project #1 - DS2500 - Luke Abbatessa - Northeastern University.pptx - a .pptx file that represents a slideshow presentation detailing the specifics of the project

***This project observed predicted relationships between atmospheric CO2 and pH and between pH and alkalinity for five coastal springs in west-central Florida in an effort to analyze ocean acidification***

Public_Statement_Analysis - folder that contains 10 things:
1) data_files - a subfolder containing 11 corporate apologies for data breaches, both as .txt files and as .json files
2) stock_data_files - a subfolder containing stock data for the companies involved in the apologies
3) Final Report - DS3500 Final Project.pdf - a .pdf file representative of the final report written for the project
4) data_prep.py - a .py file that provides a foundation for gathering VADER sentiment scores for a group of files (libraries used: textquisite (a .py file representative of a manually created reusable extensible NLP framework), textquisite_parsers (a .py file representative of a manually created .json parser))
5) project_dashboard.py - a .py file that establishes a dashboard illustrating data breach corporate apology comparisons (libraries used: dash_bootstrap_components, dash, sentiment_stock_plot (a .py file that graphs an average sentiment score vs. stock price percentage plot), textquisite, matplotlib, stock_price_plot (a .py file that graphs a stock price plot), dash_bootstrap_templates)
6) sentiment_nltk.py - a .py file that tokenizes and performs nltk vader analysis on texts (libraries used: nltk, pandas)
7) sentiment_stock_plot.py - a .py file that provides a foundation for creating an average sentiment score vs. stock price percentage plot (libraries used: pandas, plotly, data_prep, collections)
8) stock_price_plot.py - a .py file that graphs a stock price plot (libraries used: plotly, pandas)
9) textquisite.py - a .py file that establishes a reusable NLP library (libraries used: collections, sentiment_nltk, plotly, nltk)
10) textquisite_parsers.py - a .py file that establishes a custom .json parser for the user to implement (libraries used: json, textquisite, collections, sentiment_nltk)

***This project allowed users to explore sentiment scores of 11 corporate apologies for data breaches and corresponding stock fluctuations as a result***

Weather_Disaster_Prediction - folder that contains 3 things:
1) DS3000FinalProjectCodeWalkthrough.mp4 - a .mp4 file representative of a walkthrough of the code for the project
2) DS3000_final_poster.pdf - a .pdf representative of the final poster for the project
3) weather_disasters.ipynb - a .ipynb file that includes data cleaning steps (i.e. merging and concatenating dataframes, removing duplicate rows from a dataframe, filtering a dataframe based on columns of interest, changing column data types, modifying column values, deleting rows with missing values) and the implementation of three machine learning algorithms (random forest regression, knn regression, and multiple linear regression) (libraries used: seaborn, pandas, sklearn, numpy, matplotlib)

***This project predicted storm property damage based on storm event properties using data from the NOAA National Centers for Environmental Information Storm Events Database during 2012-2022***
