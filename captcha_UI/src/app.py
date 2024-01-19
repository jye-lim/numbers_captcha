#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs
import streamlit as st

# Custom
from page import inference,annotation,contributors

##################
# Configurations #
##################

logo = "./assets/images/logo.png"

PAGES = {
    "Annotation": annotation,
    "Inference Results": inference,
    "Scoreboard": contributors
}

#############
# Functions #
#############

def main():
    st.set_page_config(page_title="Number Captcha", layout='wide', initial_sidebar_state='auto', page_icon=logo)
    with st.sidebar:
        st.image(logo, use_column_width=True)
        st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Dynamically load the selected page
    page = PAGES[choice]
    page.show()

##########
# Script #
##########

if __name__ == "__main__":
    main()
