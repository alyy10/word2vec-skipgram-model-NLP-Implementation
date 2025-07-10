# Skip-Gram Model for Word Embeddings (Word2Vec)

![ProjectArchitecture](https://github.com/user-attachments/assets/67afc518-138d-4788-87a9-6c8de8bc2496)


## Overview

This project implements the **Continuous Skip-Gram model** for generating **word embeddings** using the **Word2Vec** algorithm, applied to a dataset of customer complaints about consumer financial products. The Skip-Gram model helps convert words into vector representations in a continuous vector space, where semantically similar words are closer together.

The project aims to:

- Understand the continuous Skip-Gram algorithm.
- Implement the Skip-Gram model in Python using PyTorch.
- Create word embeddings using the Skip-Gram model.

## Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`: For data manipulation and analysis.
  - `torch`: For building and training the Skip-Gram model.
  - `nltk`: For natural language processing tasks like tokenization.
  - `numpy`: For numerical operations.
  - `pickle`: For saving and loading models and data.
  - `re`: For regular expressions (used in text preprocessing).
  - `os`: For interacting with the operating system (handling file paths).
  - `tqdm`: For showing progress bars during long-running processes.

## Prerequisites

- Basic knowledge of Python and machine learning concepts.
- Familiarity with PyTorch and how neural networks are built and trained.
- Understanding of natural language processing techniques like tokenization and word embeddings.

## Project Structure

```plaintext
├── config.py           # Configuration file with hyperparameters
├── data.py             # Data loading and preprocessing (tokenization, cleaning)
├── model.py            # Skip-Gram model definition using PyTorch
├── utils.py            # Utility functions for saving/loading files
├── Dataset/              # Directory containing input data files (e.g., complaints.csv)
├── Output/             # Directory for saving output files (e.g., tokenized data, embeddings)
├── README.md           # Project overview and instructions
```

## Dataset

The dataset used in this project contains over two million customer complaints regarding financial products. It has the following columns:

- `Consumer complaint narrative`: Contains the actual complaint text.
- `Product`: The product related to the complaint.

### Data Preprocessing

The preprocessing steps include:

1. **Loading Data**: Reading the dataset using `pandas`.
2. **Missing Values**: Dropping rows with missing values in the text column.
3. **Text Cleaning**:
   - Convert text to lowercase.
   - Remove punctuation (except apostrophes).
   - Remove digits.
   - Remove repeated occurrences of 'x' (e.g., "xxxx...").
   - Normalize spaces by replacing multiple spaces with a single space.
4. **Tokenization**: Tokenizing the cleaned text using `nltk.word_tokenize()` to split text into words.
5. **Saving Tokens**: Saving the tokenized text to a file using `pickle`.

## Approach

### 1. **Understanding Word Embeddings and Word2Vec**

- **Word2Vec**: A group of models for computing vector representations of words. Word2Vec can be implemented using two approaches:

  - **CBOW (Continuous Bag of Words)**: Predicts a target word based on context words.
  - **Skip-Gram**: Predicts context words from a target word.

  This project implements the **Skip-Gram model**, which is particularly useful for rare words in the corpus.
- **Semantic vs Syntactic Similarity**: Word embeddings capture both semantic and syntactic similarities between words. Words with similar meanings will be located close together in the vector space.

### 2. **Continuous Skip-Gram Algorithm**

- The **Skip-Gram model** works by taking a central word (the "target") and using it to predict the context words around it within a specified context window.
- **Negative Sampling**: To speed up training, negative samples are used, where randomly chosen words (not in the context) are fed to the model as well.

### 3. **Training the Model**

- **SkipGramDataset**: A custom dataset class in `data.py` that processes the tokenized text, prepares positive and negative samples, and creates the data loader for training.
- **SkipGram Model**: Defined in `model.py` using PyTorch's `nn.Module`. The model consists of:
  - **Embedding Layer**: Converts words into continuous vector representations.
  - **Context Word Weights**: A weight matrix for the context words.
  - **LogSigmoid**: Applied to output probabilities of context words.

### 4. **Training Procedure**

- **Loss Function**: The `Negative Log Likelihood Loss (NLLLoss)` is used as the loss function to train the model.
- **Optimization**: The Adam optimizer is used to adjust the weights and minimize the loss function.

### 5. **Saving and Loading Model**

- The trained word embeddings and model weights are saved using `pickle` for future use and loading during inference.

### 6. **Using the Embeddings**

- After training, the learned word embeddings are used to get the vector representation for any word in the vocabulary.

### 7. **Model Evaluation**

- The quality of the word embeddings can be evaluated using various tasks like analogy questions (e.g., "king - man + woman = queen").

## Procedure

### Step 1: **Data Preprocessing**

- Download the dataset and save it to the `Input/` directory.
- Run the data preprocessing script (`data.py`) to clean the text, tokenize it, and save the tokens to the `Output/` directory.

### Step 2: **Training the Skip-Gram Model**

- Set the configuration in `config.py` (e.g., context window, embedding size, number of epochs).
- Run `model.py` to define the Skip-Gram model and train it on the processed dataset.

### Step 3: **Evaluate the Word Embeddings**

- After training, use the embeddings to check the similarity between words using cosine similarity or any other method to see how well the model has captured semantic meanings.
