"""
DS4420: NewMovMie_App.py 
Luke Abbatessa and Jocelyn Ju

This Python script serves as the StreamLit web app designed 
for the project New MovMie. It contains three pages: a landing
page, an interactive display of the collaborative filtering
ratings, and an interactive plot of the hybrid model.

Last Updated: 14 April 2025
"""

import streamlit as st
import pandas as pd
import altair as alt

# import data
df_results = pd.read_csv("data/all_results.csv")
df_metrics = pd.read_csv("data/comparison_results.csv")
df_mlp = pd.read_csv("data/mlp_results.csv")
df_cf = pd.read_csv("data/10_CF_results.csv")
df_hybrid = pd.read_csv("data/hybrid_results.csv")

# ensure standardization of naming columns
df_mlp.columns = ["UserID", "MovieID", "MLP_rating"]
df_cf.columns = ["UserID", "MovieID", "CF_rating"]
df_hybrid.columns = ["UserID", "MovieID", "Highest_Rating"]
df_cf["MovieID"] = df_cf["MovieID"].astype(str).str.replace("Rating.", "",regex=False).astype(int)
df_mlp["MovieID"] = df_cf["MovieID"].astype(int)

TARGET_USERS = [196, 186, 22, 244, 166, 298, 115, 253, 305, 6]

def landing():
    # sidebar
    st.sidebar.success("Select a page to navigate to")

    # content for landing page
    reasons = ["I'm tired of trying to figure out what to watch until 2 am.",
               "My friends always make me pick, and I'm out of ideas.",
               "I want to try something new, but I don't know where to start.",
               "I'm bored because I've seen the same thing over and over again, but nothing else is good, so I might as well play The Office again.",
               "All of the above.",
               "Something else."]
    
    responses = ["2 am is pretty late. Not great for decision making, unlike...",
                 "Some people would tell you to get new friends. I'm telling you to get a...",
                 "They say that weddings should something old, something new, something borrowed, something blue. Not sure what you're going to do about all that other stuff, but here's something...",
                 "Yeah, that sucks. Sounds like you might need some kind of...",
                 "Oh yikes. Wouldn't it be great if I had the solution to all of your problems? Like maybe...",
                 "I thought these were pretty comprehensive, but I guess you want to be different. Whatever. Here's..."]

    # ask the user what their issue is
    reason_why = st.selectbox(
        "Why do you want a movie recommendation?",
        [reason for reason in reasons]
    )        

    if reason_why:
        # select the specific response to the selected reason
        reason_index = reasons.index(reason_why)
        st.markdown(responses[reason_index])

        # title
        st.title("✨🎬 New MovMie 🎬✨")
        st.subheader("Leveraging Collaborative Filtering and Multi-Layer Perceptron for Movie Recommendations")

        # intro
        st.markdown(
        """
        ##### About New MovMie:

        New MovMie */noo moov-mee/* is a recommendation system designed to provide you with a movie recommendation (i.e. a "New Movie for Me").
        It is so easy, especially with the current resigned acceptance around doomscrolling and phone addiction,
        to sit back and say 'whatever, it doesn't matter'. But being trapped in an echo chamber and never hearing
        new perspectives is scientifically proven to be depressing and lonely and isolating.
        While this project is specifically geared for long-form media (movies), we hope that it will spark an interest
        in a New Mov(M)ie. Even if it doesn't, we're proud of you for trying.

        ##### Technical Overview:

        We combined collaborative filtering (CF) and a multi-layer perceptron model (MLP)
        to create these unique, 'you-will-probably-like-this-but-we're-not-totally-sure' ratings.
        
        Our *CF method* used a k-value of 3 and the cosine similarity metric to predict ratings.
        
        Our *MLP method* had one hidden layer of 10 nodes, a step size of 0.1, loss function MSE,
        the ReLU activation function (hidden), and the sigmoid activation function (output).
        
        The combined, *Hybrid Model* used the generated datsets of user | movie | rating from CF 
        and MLP, extracted the movies a user hasn't rated and the corresponding CF & MLP scores,
        then evaluated the scores, returning the movie with the highest CF for the lowest MLP.
        """)

def cf_plot():
    # menu
    st.sidebar.success("Select a page to navigate to")

    # explanation
    st.markdown(
    """
    # Collaborative Filtering Plot
    Welcome to the CF plot! Please select which k-values and similarity metrics
    you would like to display data for below.
    """
    )

    st.write("The ***k-values [3/5/10/20]*** are how many nearest neighbors the algorithm takes into account in performing the rating calculations.")
    ## selectbox to choose which data to show
    selected_k = st.multiselect(
        "What k value(s) would you like displayed?",
        [3, 5, 10, 20],
        [3, 5, 10, 20]
    )

    st.write("The ***similarity metric [Cosine/L2]*** refers to the metric used when performing rating calculations.")
    selected_sim = st.multiselect(
        "What similarity metric(s) would you like displayed?",
        ["Cosine", "L2"],
        ["Cosine", "L2"]
    )

    # filter data by selected_k and selected_sim to show a scatterplot 
    # actual vs. predicted values
    new_dat = df_results[
        (df_results["k"].isin(selected_k)) &
        (df_results["Similarity"].str.lower().isin([sim.lower() for sim in selected_sim]))
        ]
    
    st.markdown(
        """
        The below plot will show the CF's actual vs. predicted ratings for the 
        test dataset with the selected parameters. Beneath the plot, a chart with
        the specified parameters and resulting metrics is displayed.
        """
    )

    # scatter plot
    st.scatter_chart(new_dat, x="actual", y="pred", x_label="Actual Rating", y_label="Predicted Rating")

    # filter metrics to display
    new_metrics = df_metrics[
        (df_metrics["k"].isin(selected_k)) &
        (df_metrics["Similarity"].str.lower().isin([sim.lower() for sim in selected_sim]))
    ]
    st.write(new_metrics)

def hybrid_plot():
    # menu
    st.sidebar.success("Select a page to navigate to")

    # title
    st.markdown(
        """
        # Hybrid Plot
        """
    )

    # pick a user from dropdown
    user = st.selectbox(
        "Select a user:",
        sorted(TARGET_USERS)
    )

    # combine dfs
    merged_df = pd.merge(df_cf, df_mlp, on=["UserID", "MovieID"])

    # extract only the user
    merged_df = merged_df[merged_df["UserID"] == user]

    # add a difference column
    merged_df["RatingDiff"] = merged_df["CF_rating"] - merged_df["MLP_rating"]

    # show the top [x] movies
    top_movies = st.select_slider(
        "How many top movies would you like to see?",
        [i for i in range(1, len(merged_df["UserID"] + 1))],
        value = len(merged_df["CF_rating"]) - 1
    )

    # filter by top movies
    top_cf = merged_df.sort_values(by="CF_rating", ascending=False).head(top_movies)

    # Legend Title
    st.markdown(f"""## Comparing the top {top_movies} movies for User: {user}""")

    # plot the data with altair 
    col1, col2 = st.columns([4,1])   
    with col1:
        base = alt.Chart(top_cf).encode(x="MovieID")
        line_cf = base.mark_line(color="blue").encode(y="CF_rating",
                                                    tooltip=["MovieID", "CF_rating"])
        line_mlp = base.mark_line(color="orange").encode(y="MLP_rating",
                                                    tooltip=["MovieID", "MLP_rating"])
        bar = base.mark_bar(color="green").encode(y="RatingDiff:Q",
                                                color=alt.condition(
                                                    alt.datum.RatingDiff > 0,
                                                    alt.value("green"), # green if CF > MLP
                                                    alt.value("crimson") # red else
                                                ),
                                                tooltip=["MovieID", "RatingDiff"])
        # put them together
        (line_cf + line_mlp + bar)
    with col2:
        st.markdown(
            """
            ### Legend
            🟦**CF Rating**

            🟧**MLP Rating**

            🟩**CF > MLP**

            🟥**MLP > CF**
            """
        )
    
    # explanation
    st.markdown(
        """
        ### How to (Generally) Interpret:
        * What we want: tallest green bar for the point that has a high blue line value
        and a relatively low orange line value.
        * What it means:
            * **Tall green bar**: the CF rating is higher than the MLP rating
                * *Why is this good?* It means others similar to the User liked the movie more than was anticipated by the MLP.
            * **High blue line**: the CF rating is high
                * *Why is this good?* High collaborative filtering scores indicate the user will likely enjoy the movie.
            * **Relatively low orange line**: the MLP rating is relatively low
                * *Why is this good?* Low MLP rating scores indicate that the user would likely not pick that movie on their own.

        Want an easier layout? Table below:
        """
    )

    # display data for that user
    st.write(top_cf)


page_names_to_funcs = {
    "Home": landing,
    "Collaborative Filtering Plot": cf_plot,
    "Hybrid Plot": hybrid_plot,
}

nav = st.sidebar.selectbox("Navigation Bar", page_names_to_funcs.keys())
if nav:
    page_names_to_funcs[nav]()
else:
    page_names_to_funcs["Home"]()
