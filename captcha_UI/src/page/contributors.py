import streamlit as st
from utils.utils import get_contributors
from config import TEST_MODE
import pandas as pd


def get_top_contributors():
    if TEST_MODE:
        df=pd.read_csv('./assets/annotations.csv')
    else:
        df= get_contributors()
    df['pred_correct'] = df['label'] == df['true_label']
    contributor_stats = df.groupby('user').agg(
        total_contributions=('id', 'count'),
        correct_contributions=('pred_correct', 'sum')
    ).reset_index() 

    # Calculate accuracy as a percentage
    contributor_stats['accuracy'] = (
        contributor_stats['correct_contributions'] 
        / contributor_stats['total_contributions']
        * 100)
    return contributor_stats


def highlight_user(row, user):
    """Highlight the row if the Username matches the given user."""
    if row["user"] == user:
        return ["background-color: yellow"] * len(row)
    return [""] * len(row)

def show():
    st.title("Scoreboard")
    
    contributors = get_top_contributors()

    
    
    medals = ["🥇", "🥈", "🥉"]
    contributors=contributors.sort_values('total_contributions',ascending=False)
    contributors=contributors.reset_index()
    contributors=contributors.reset_index()
    contributors["Placing"] = contributors['level_0'].apply(lambda x: x+1)
    contributors["Placing"]=contributors["Placing"].astype('str')
    
    for i, medal in enumerate(medals):
        if i < len(contributors):
            contributors.at[i, "Placing"] = medal

    contributors = contributors[['Placing', 'user', 'total_contributions','accuracy']]

    search_term = st.text_input("Search for a contributor:")
    if search_term:
        contributors = contributors[contributors["user"].str.contains(search_term, case=False)]

    # Highlighting the current user
    if "username" in st.session_state:
        styled_df = (contributors.style
                     .apply(highlight_user, user=st.session_state.username, axis=1)
                     .set_properties(**{'text-align': 'center'}))
        st.dataframe(styled_df, width=600, height=400, hide_index=True)
    else:
        styled_df = contributors.style.set_properties(**{'text-align': 'center'})
        st.dataframe(styled_df, width=600, height=400, hide_index=True)