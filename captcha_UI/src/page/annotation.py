#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import requests
import base64
import json


# Libs
import streamlit as st

# Custom


##################
# Configurations #
##################
ss = st.session_state

from config import ANNOTATION_ENDPOINT, TEST_MODE

#############
# Functions #
#############
def username_handler(username):
    """ Sets session username and shows an error if it is not set 
    
    Args:
        username (str): String to identify oracle user
    """

    if username:
        ss.username = username
        st.success(f"Session started for {username}.  ")
    else:
        st.warning("Please enter a name before starting the session!")

def load_image_as_base64(image_path):

    """ Load an image from the specified path and convert it to a base64
        encoded string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded image as a string.
    """
    with open(image_path, 'rb') as img:
        file_bytes = img.read()
        return base64.b64encode(file_bytes).decode('utf-8')

def fetch_annotation_data(endpoint=ANNOTATION_ENDPOINT):
    """Fetch an image for annotation
    Args:
        endpoint (str): API GET endpoint for fetching of annotation data
    Returns:
        dict: {'image': imageData
                'paths': A list of paths associated with the image sent}
    """
    if TEST_MODE:
        dummy_data_path = "./assets/dummy_values/dummy_annotation.json"
        with open(dummy_data_path, "r") as file:
            data = json.load(file)
            data['image'] = load_image_as_base64(data['image_path'])
    else:
        response = requests.get(endpoint)
        if response.status_code != 200:
            st.error("Failed to fetch data from the server!")
            return None
        data = response.json()
    ss['annotation'] = ""
    ss.submitted = False

    return data

def submit_annotation_data(
    paths, 
    labels, 
    username, 
    true_labels, 
    endpoint: str = ANNOTATION_ENDPOINT):
    """Submit annotation data to the server.

    Args:
        paths (list): A list of paths associated with the annotations.
        labels (list): A list of annotation labels.
        username (str): The username of the annotator.
        true_labels (list): A list of true labels (if available).
        endpoint (str): API POST endpoint for submission of annotation data

    Returns:
        bool: True if the data was successfully submitted, False otherwise.
    """
    if TEST_MODE:
        payload = {"paths": paths, "labels": labels}
        st.success("Dummy submit successful!")
        return True
    else:
        payload = {"paths": paths, 
                   "labels": labels, 
                   "username": username, 
                   "true_labels": true_labels}
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            st.success("Data submitted successfully!")
            return True
        else:
            st.error("Failed to submit data to the server!")
            return False
        
def display_image_and_input():
    """Display an image and input field for annotation.

    Args:
        image_data (bytes): The image data to be displayed.
        paths (list): A list of paths associated with the annotation.

    Returns:
        None
    """
    image_data = base64.b64decode(ss.data.get("image", None))
    paths = ss.data.get("paths", [])
    st.image(image_data, caption="Image", width=480, output_format="JPEG")
    labels = st.text_input(
        "Enter a 10-digit number:", 
        max_chars=10, 
        key="annotation")

    # Check if the "Submit" button has been pressed before
    if ss.get("submitted", False):
        st.button("Submit", disabled=True)  # Render a disabled "Submit" button
    else:
        if st.button("Submit"):
            if len(labels) == 10 and labels.isdecimal():
                with st.spinner("Submitting data..."):
                    success = submit_annotation_data(
                        paths, 
                        labels, 
                        ss.username, 
                        ss.data.get("true_labels", []))
                if success:
                    ss.submitted_count += 1
                    ss.successful_submit = True 
                    ss.submitted = True  
                    st.rerun()
            else: 
                st.warning("Please input a valid 10-digit number.")

def check_username():
        if "username" not in ss:
            username = st.text_input("Please enter your name:")
            st.button("Start Session", on_click=username_handler, args=(username,))
            return False
        else:
            return True

def load_data():
    """Check's if annotation data if needed.

    Returns:
        bool: True if data has been loaded, False otherwise.
    """

    if "data" not in ss or ss.get("submitted", False):
        if ss.get("successful_submit", False):
            ss.successful_submit = False  # Reset the successful submit flag
            st.success(("Thank you for your input, check out ",
                        "the leaderboard for updated scores!"))
            with st.spinner("Loading next sample..."):
                ss.data = fetch_annotation_data()
        else:
            with st.spinner("Loading sample..."):
                ss.data = fetch_annotation_data()
        ss['annotation'] = ""
        ss.submitted = False
    return True

def render_session_info():
    """ Renders session info, including username and number of samples 
    submitted
    """
    ss.submitted_count = ss.submitted_count if "submitted_count" in ss else 0

    st.subheader("Session Info")
    st.write(f"User: {ss.username}")
    st.write(f"Samples submitted: {ss.submitted_count}")

##########
# Script #
##########

# The show function renders 
def show():
    st.title("Active Learning")
    if check_username():
        load_data()
        col1, col2 = st.columns((3, 1))
        
        with col1:
            display_image_and_input()

        with col2:
            render_session_info()


