# Active Learning on the SVHN Dataset

## Project Overview

In this project, we create a simple annotation deployment for the SVHN dataset (from which we remove labels), demonstrating a scalable approach towards Active Learning.

We also explore the cost, benefits, and tradeoffs of different methods in both the streaming and non-streaming case, for which we implement the VeSSAL algorithm as outlined in [Streaming Active Learning with Deep Neural Networks](https://doi.org/10.48550/arXiv.2303.02535).


## Problem Description
Some common problems in **Active Learning** that we identified and tried to solve.

- **Labeled Data Is Expensive**: The fundamental problem Active Learning is trying to solve is that labeled data is more expensive than unlabeled data. We try to solve this by picking what data is to be labeled, attempting to find the most **informative** and **representative** data.

- **Streaming vs Non-Streaming Cases**: The problem also becomes quite different whether we have access to the full unlabeled dataset or only able to view limited data at a time, known in literature as the **non-streaming** and **streaming** cases.

- **Unreliable Oracles or Adversarial Attacks**: What do we do in the case of unreliable Oracles(annotators)? How do we identify and mitigate the damage?

## Approach and Deployment
### Dataset
To simulate a largely unlabeled dataset, we split our SVHN dataset into three sections:

- Labeled Training Dataset
- Labeled Evaluation Dataset
- Annotation Dataset (With labels removed)

### Deployment
The service is split into two parts as follows.

- **Server**:
  - Serving up of sampled unlabeled data
  - Storing of annotated data, as well as sources
  - Continuous Training and Evaluation of Models

- **User Interface**:
  - Displaying unlabeled data for annotation to users
  - Collection of annotation data, sent back to server


## Visuals(GIFS TO BE ADDED)


Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Quick Start

### Streamlit app

#### Installation
- `cd /streamlit_app`
- `pip install -r streamlit_requirements.txt`
- `cd ..`

#### Run
- `streamlit streamlit_app/src/app.py`

#### Pages
- Annotation
- Inference
- Scoreboard

### FastAPI

#### Installation
- `pip install -r requirements.txt`
- `uvicorn run src.router:app` (may have to use `- gunicorn` for deployment)

#### Endpoints available

- **`/annotation` (GET)**  
  **Description:** Fetches a concatenated image of 10 random images and the list of paths used to generate the image.  
  **Response:**  
  - `image`: A concatenated image of size 32x320 pixels.
  - `ids`: List of 10 image paths used to generate the concatenated image.

- **`/annotation` (POST)**  
  **Description:** Accepts a prediction and updates the demo CSV with the predicted labels. Schedules training if not already scheduled.  
  **Request Body:**  
  - `labels`: String representation of the labels.
  - `ids`: List of image paths.  
  **Response:**  
  - `pred`: List of predicted integers.

- **`/checkpoint_metrics` (GET)**  
  **Description:** Fetches metrics from the checkpoint.  
  **Response:** JSON object containing the checkpoint metrics.

- **`/inference` (GET)**  
  **Description:** Performs inference on a sample of test data using both the base model and the current model.  
  **Response:**  
  - `images`: List of concatenated images.
  - `base_model_predictions`: Predictions made by the base model.
  - `new_model_predictions`: Predictions made by the current model.

- **`/train` (GET)**  
  **Description:** Starts the training process for the model. Can initialize a new model or continue training the previous model.  
  **Parameters:**  
  - `new_cycle` (optional): Boolean indicating whether to start a new training cycle. Default is False.
  - `export` (optional): Boolean indicating whether to export the model after training. Default is True.
  - `epochs` (optional): Number of training epochs. Default is 5.  
  **Response:** Returns the result of the `trigger_train_loop` function.

- **`/delaytrain` (GET)**  
  **Description:** Schedules a delayed training session to start after a predefined trigger cycle.  
  **Response:** Boolean indicating the success of scheduling the training.



#### Docker Containers 
Execute the command `docker-compose up --build` to build Docker containers using the `docker-compose.yml` configuration. This will use `Dockerfile_fastapi` and `Dockerfile_streamlit` to run the `fastapi` and `streamlit_app` services respectively.

After the build process, the `FastAPI` image has been pushed to `registry.aisingapore.net/aiap-14-dsp/numbers_captcha_fastapi`, and the `Streamlit` image has been pushed to `registry.aisingapore.net/aiap-14-dsp/numbers_captcha_streamlit_app`. 

Deploy the FastAPI using `kubectl apply -f fastapi-deployment.yaml`. 

Determine the node for FastAPI pods using `kubectl get pods -l app=fastapi -o wide`. Currently, it is `runai-worker-1` for FastAPI pod.  

Get IP address of all the nodes using `kubectl get nodes -o wide`. `runai-worker-1` has an Internal-IP (NodeIP) of `172.19.152.111`. 

To find the NodePort FastAPI, execute the following kubectl commands: `kubectl get services | grep fastapi`. For FastAPI it's 30594. Ensure that within the `streamlit-deployment.yaml` file, we've set the value of `API_URL` to the NodeIP:NodePort combination from FastAPI. Currently, it is `172.19.152.111:30594`

Deploy the Streamlit application using `kubectl apply -f streamlit-deployment.yaml`. Determine the NodeIP and NodePort for Streamlit. Afterwards, go to URL of NodeIP:NodePort combination from Streamlit. Currently, it is `http://172.19.152.111:30203/`. 


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.