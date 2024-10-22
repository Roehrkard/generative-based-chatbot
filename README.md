# AAI-520-NLP-Generative Chatbot
This project focuses on designing and implementing a generative chatbot capable of conducting multi-turn conversations, adapting to context, and handling a variety of topics. Using state-of-the-art architectures like Transformers, BERT, or GPT, the goal is to build a functional chatbot that demonstrates both technical knowledge and practical implementation.

### Requirements

- Python 3.x
- [pip](https://pip.pypa.io/en/stable/)

## Installation

1. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv final_project_venv
    ```

2. Activate the virtual environment:

    - **On Windows**:

    ```bash
    final_project_venv\Scripts\activate
    ```

    - **On macOS/Linux**:

    ```bash
    source final_project_venv/bin/activate
    ```

3. Install dependencies using `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Notebooks

### Option 1: Running Locally in a Jupyter Notebook

1. Install Jupyter Notebook (if not already installed):

   ```bash
   pip install jupyterlab
   ```

3. Start the Jupyter Notebook server:

- **On Windows**:

  ```bash
  final_project_venv\Scripts\activate
  jupyter notebook
  ```

- **On macOS/Linux**:

  ```bash
  source final_project_venv/bin/activate
  jupyter notebook
  ```

3. Ensure that your virtual environment is being used in the Jupyter notebook by selecting the appropriate kernel:

- Click on the Kernel menu.
    - Choose Change kernel.
    - Select your environment (final_project_venv).

4. Navigate to your project directory and open the notebook where you want to run the code.

5. Enter questions related to the Stanford Question Answering Dataset
```https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset```
```Who was the first president of the United States?
What is the capital of France?
How do you make a chocolate cake?
Can you explain the theory of relativity?
What are the symptoms of the flu?
```

### Option 2: Running in Google Colab

- More robust evaluation using the official SQuAD Evaluation Script:
    - https://colab.research.google.com/drive/1i2cQ_YmLI80OryMwIk081lhRJzGyeMNM?usp=sharing

- Evaluation Metrics with HF's Evaluation Library, Working version:
    - https://colab.research.google.com/drive/1JjhvuVpaTdUC0QGB6dkDwg_TW354E9_j?usp=sharing

### Option 3: Flask Generative Chatbot Web Interface

1. Download repo folder

2. Start the Jupyter Notebook server

3. Launch a terminal and navigate to the project folder

4. Enter the command
```python app.py```

5. visit http://127.0.0.1:5000/

6. Enter questions related to the Stanford Question Answering Dataset
```https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset```
```Who was the first president of the United States?
What is the capital of France?
How do you make a chocolate cake?
Can you explain the theory of relativity?
What are the symptoms of the flu?
```

### Option 4: Gradio Generative Chatbot Web Interface

1. Run the following Colab script
- Gardio interface, Version that includes context for each answer:
    - https://colab.research.google.com/drive/1h5Z5ZAttYBXGlRjyjJKXylK9TqDP7YuM?usp=sharing

2. Visit running Gradio server
    - https://80d9500beec42ba879.gradio.live/

2. Enter questions related to the Stanford Question Answering Dataset
```https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset```
```Who was the first president of the United States?
What is the capital of France?
How do you make a chocolate cake?
Can you explain the theory of relativity?
What are the symptoms of the flu?
```

