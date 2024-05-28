# Data-Structure-and-Algorithm-Chatbot
This project is a chatbot designed to provide information, explanations, and lookup functions for basic data structures and algorithms. The chatbot aims to assist users in understanding key concepts and theories, as well as provide quick references to various underlying algorithms and data structures.

## How to Use This Project
To get started with this project, follow these steps:

### Clone repo from github

### Download the Model:

Refer to the modelinstruction.txt file located in the model folder of the project. This file contains instructions on how to download the necessary model for the chatbot.

### Set Up Anaconda Environment:

Download and install Anaconda from [here](https://www.anaconda.com/) if you haven't already.
Open your terminal or command prompt.
#### Notice : You need to add Conda to your PATH 

Create a new Anaconda environment using the following command:

```bash
conda create -n alchatbot python=3.11 -y
```
Activate the Anaconda Environment:
Activate the newly created environment by running:

```bash
conda activate alchatbot
```

### Install Required Packages:

Install all the necessary packages listed in the requirements.txt file by running:

```bash
pip install -r requirements.txt
```

### Run the Application:

Use the terminal to run the application with the following command:

```bash
python app.py
```

### Access the Chatbot:

Open your web browser and go to http://localhost:5000 to start using the data structures and algorithms chatbot.

## Technical Details

### Libraries and Tools

LangChain:
- Document Loaders: PyMuPDFLoader, DirectoryLoader
- Text Splitter: RecursiveCharacterTextSplitter
- Embeddings: HuggingFaceEmbeddings
- Vector Store: FAISS

CTransformers: Loads and uses the Llama-2 language model.

Flask: Web framework for the chatbot interface.