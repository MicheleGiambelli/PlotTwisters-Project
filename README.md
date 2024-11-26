# PlotTwisters-Project

Welcome to the PlotTwisters Project! This repository contains the code, notebooks, and resources developed by our team to analyse and extract insights from textual datasets. Below, you'll find a detailed explanation of our dataset, objectives, and instructions on how to navigate and use the repository.

## Objective
The primary objective of this project is to develop a system that, given an input sentence, decomposes it into tokens and assigns a Named Entity Recognition (NER) tag to each token. To achieve this, we have implemented two approaches:

1. A neural network-based model, designed to provide a robust and efficient solution for NER tasks.
2. A fine-tuned transformer model, leveraging state-of-the-art architectures like DistilBERT to achieve high accuracy, even on complex or informal text.
These complementary approaches aim to extract structured information, such as entities (e.g., people, locations, organizations, dates), from unstructured text data with precision and scalability.

## Dataset Description
The dataset consists of English-language tweets, each broken down into tokens. For every token, a corresponding Named Entity Recognition (NER) tag label is assigned. These tags identify various entities such as people, locations, organizations, dates, and other structured information present in the text.

This dataset provides a rich source of natural language data with diverse linguistic features, including abbreviations, slang, and informal expressions, making it particularly suitable for training and evaluating models designed for NER tasks in informal text contexts.


## File Structure and Execution Order
To help you navigate the repository, here is the recommended order in which to execute the files, along with a description of each file:

### 1. DEA.ipynb
- *Description*: This notebook performs Data Exploratory Analysis (DEA) on the dataset. It includes:
   - Text preprocessing steps such as tokenization and stopword removal.
   - Visualizations of word frequencies, word clouds, and sentiment distributions.
   - Insights into the structure and characteristics of the dataset.
- *Purpose*: Provides an initial understanding of the dataset and highlights key areas of interest for further analysis.
- *Why it's important*: DEA helps in understanding the data's quality and structure, enabling informed decisions for model development.


### 2. Algorithm2.ipynb (currently in the MICHELE branch but will be moved)
- *Description*: This notebook implements a neural network-based approach for Named Entity Recognition (NER). Specifically, it develops and evaluates four different models:
    1. LSTM (Long Short-Term Memory): A sequential model capable of capturing long-term dependencies in text data.
    2. BiLSTM (Bidirectional LSTM): An extension of LSTM that processes input sequences in both forward and backward directions for improved context understanding.
    3. GRU (Gated Recurrent Unit): A simpler alternative to LSTM with comparable performance, optimized for faster computation.
    4. Multinomial Naive Bayes: A baseline probabilistic model included for comparison with the neural network approaches.
- *Purpose*: The notebook benchmarks these models to assess their performance in classifying tokens with the appropriate NER tags.
- *Why it's important*: This step provides a diverse range of approaches for solving the NER task, enabling comparisons between traditional probabilistic methods and advanced neural architectures.


### 3. Transformer_funzionante.ipynb
- *Description*: This notebook is dedicated to the fine-tuning and evaluation of a transformer model for the Named Entity Recognition (NER) task. It involves:
    - Loading a pre-trained DistilBERT model.
    - Fine-tuning the model on the dataset to recognize named entities such as people, places, and organizations.
    - Evaluating the model's performance using metrics like F1-score and accuracy.
- *Purpose*: Develops the core machine learning component of the project.
- *Why it's important*: The transformer model is the heart of the pipeline, providing the primary analytical output for the dashboard and reporting.

  
### 4. Dashboard (To be added)
- *Description*: The dashboard is an interactive web application designed to visualize and explore the results of our analysis. It includes:
    - Graphs and charts summarizing NER performance and data insights.
    - A searchable interface to explore the dataset and model outputs.
    - Real-time visualizations of token-level NER predictions and other key metrics.
- *Purpose*: To present the findings of the project in an accessible and visually appealing format for stakeholders and end-users.
- *Why it's important*: The dashboard bridges the gap between technical analysis and stakeholder comprehension, enabling actionable insights from the NER task.
- *Usage Instructions*: To use the dashboard, it is necessary to download the following files:
    1. Transformer Models: Download the transformer models fine-tuned in step 3 from this Hugging Face link: https://huggingface.co/Emma-Cap/Transformer . These files are critical for running the NER predictions within the dashboard.
    2. Dashboard Files: Download the files located in the dashboard folder (from step 2), as they contain preprocessed data and other resources needed for visualization.
Ensure that all required files are placed in the appropriate directories before running the dashboard application.


# Need Help?
If you have any questions about the structure or usage of the files, or if you encounter any issues while downloading or running the models and dashboard, please don't hesitate to reach out to us.

We are more than happy to assist you and ensure everything runs smoothly. Feel free to contact us at any time for clarifications or support.

# Acknowledgements
We would like to express our sincere gratitude to *Professor Andrea Belli* for his invaluable guidance throughout the project and for providing us with the dataset, which served as the foundation of our work. His support and insights were crucial in shaping the direction and success of this project.

Additionally, we acknowledge the efforts and collaboration of our team members:
- Michele
- Emma
- Alberto

This project would not have been possible without the dedication and teamwork of everyone involved.
