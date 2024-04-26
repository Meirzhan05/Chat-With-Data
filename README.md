# Chat-With-Website

## Features

- **Website Interaction**: The chatbot uses the latest version of LangChain to interact with and extract information from various websites.
- **Large Language Model Integration**: Compatibility with models like GPT-4, Mstrala, Llama2, and ollama. In this code I am using GPT-4, but you can change it to any other model.
- **Streamlit GUI**: A clean and intuitive user interface built with Streamlit, making it accessible for users with varying levels of technical expertise.
- **Python-based**: Entirely coded in Python.

## Installation

### Ensure you have Python installed on your system. Then clone this repository:
```console
git clone [repository-link]
cd [repository-directory]
```
### Install the required packages:
```console
pip install -r requirements.txt
```
### Create your own .env file with the following variables:
```console
OPENAI_API_KEY=[your-openai-api-key]
```
### To run the Streamlit app:
```console
streamlit run app.py
```