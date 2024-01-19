import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import requests
import pandas as pd
from decouple import config
import json


TEST_MODE = config("TEST_MODE", default="True") == "True"  # Convert string to boolean
INFERENCE_ENDPOINT = config("API_URL")+config("INFERENCE_ENDPOINT")
CHECKPOINT_METRICS_ENDPOINT = config("API_URL")+config("CHECKPOINT_METRICS_ENDPOINT")


def show():
    st.title("Inferencing and Model Performance")

    # Load the data
    data, _ = load_data()
    images = data["images"]
    base_model_predictions = data["base_model_predictions"]
    new_model_predictions = data["new_model_predictions"]

    # Create columns for image and graph layout
    col1, _, col2, _ = st.columns([3, 1, 3, 1])

    # Display images and predictions on the left
    with col1:
        st.subheader("Inference Results")
        for i, image in enumerate(images):
            st.image(image, caption=f"Input Image {i+1}", width=480,use_column_width=True)

            # Create a table for model predictions
            model_predictions = pd.DataFrame(
                {
                    "Model": ["Base Model", "New Model"],
                    "Prediction": [base_model_predictions[i], new_model_predictions[i]],
                }
            )
            st.table(model_predictions)
    # Display bar chart for model checkpoint comparison on the right
    with col2:
        st.subheader("Model Checkpoint Comparison")

        # Load checkpoint metrics 
        checkpoint_metrics = load_checkpoint_metrics()

        # Dropdowns for model and metric selection
        model_options = [
            model
            for model in checkpoint_metrics.keys()
            if model != "base_model_0500.pt"
        ]
        selected_checkpoint = st.selectbox("Select checkpoint:", model_options)

        # All metrics for the selected checkpoint
        metric_options = list(checkpoint_metrics[selected_checkpoint].keys())
        selected_metric = st.multiselect(
            "Select metrics:", metric_options, default=metric_options
        )

        for metric_name in selected_metric:
            st.write(f"**{metric_name} Comparison**")
            comparison_data = {
                "Base_model_0500": checkpoint_metrics["base_model_0500.pt"][metric_name],
                selected_checkpoint: checkpoint_metrics[selected_checkpoint][
                    metric_name
                ],
            }
            df = pd.DataFrame(
                [comparison_data]
            ).transpose()  # Transpose so that models are columns
            df.columns = ["Value"]  # Naming the column for clarity in the chart
            st.bar_chart(df)


def load_data():
    if TEST_MODE:
        return load_dummy_inference_data(), load_checkpoint_metrics()
    else:
        return (
            fetch_inference_data_from_endpoint(),
            fetch_checkpoint_metrics_from_endpoint(),
        )


def load_dummy_inference_data():
    """Load dummy inference data."""
    with open(
        "./assets/dummy_values/dummy_inference_data.json", "r"
    ) as file:
        dummy_data = json.load(file)

    # Loading all the images from the paths specified in the JSON
    images = []
    for image_path in dummy_data["images"]:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            images.append(img)

    # Replacing the image paths with actual images in the dummy_data dictionary
    dummy_data["images"] = images

    return dummy_data


def fetch_inference_data_from_endpoint(data_type="bytestring"):
    """Fetch inference data from an endpoint, expecting either bytestrings or image URLs based on data_type."""
    response = requests.get(INFERENCE_ENDPOINT)
    data = response.json()

    images = []
    if data_type == "bytestring":
        for image_bytestring in data["images"]:
            image_data = base64.b64decode(image_bytestring)
            image = Image.open(BytesIO(image_data))
            images.append(image)
    elif data_type == "url":
        for image_url in data["images"]:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            images.append(image)
    else:
        raise ValueError("Invalid data_type. Expected 'bytestring' or 'url'.")

    # Replace either bytestrings or URLs with actual images in the data dictionary
    data["images"] = images

    return data


def load_checkpoint_metrics():
    """Load checkpoint metrics."""
    if TEST_MODE:
        with open(
            "./assets/dummy_values/checkpoint_metrics.json", "r"
        ) as file:
            checkpoint_metrics = json.load(file)
        return checkpoint_metrics
    else:
        return fetch_checkpoint_metrics_from_endpoint()


def fetch_checkpoint_metrics_from_endpoint():
    """Fetch checkpoint metrics from an endpoint."""
    response = requests.get(CHECKPOINT_METRICS_ENDPOINT)
    data = response.json()
    return data
