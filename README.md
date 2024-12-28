# metaphor-detection-cnn-lstm
Metaphor detection using cnn lstm

<a href="https://ibb.co/tpdbW5M"><img src="https://i.ibb.co/1GS9wxf/Screenshot-12-27-2024-9-11-52-PM.png" alt="Screenshot-12-27-2024-9-11-52-PM" border="0"></a><br /><a target='_blank' href='https://freeonlinedice.com/'>dice roll simulator</a><br />

Preprocessing_Plus_Visualization.ipynb implements a comprehensive data preprocessing pipeline for a text classification dataset. Here's a detailed report of the steps and their purposes:

Data Cleaning and Preprocessing:

1. Duplicate Removal
   
Reads the original CSV file 'train-1.csv'.
Removes duplicate rows using drop_duplicates(), keeping the first occurrence.
Saves the deduplicated dataset as 'train-1_cleaned_dup.csv'.

3. Punctuation Removal
   
Removes all punctuation from the 'text' column using string.punctuation.
Saves the result as 'train-1_cleaned_pun.csv'.

5. Lowercase Conversion
   
Converts all text in the 'text' column to lowercase.
Saves the result as 'train-1_cleaned_lc.csv'.

7. Stop Word Removal
   
Defines a set of common English stop words (limited to 'a', 'an', 'the').
Removes these stop words from the 'text' column.
Saves the result as 'train-1_cleaned_nostop.csv'.

9. URL and HTML Removal
    
Removes URLs and HTML codes from the 'text' column using regular expressions.
Cleans extra whitespace.
Saves the result as 'train-1_cleaned_nourl.csv'.
Data Transformation

11. Mapping MetaphorID to Words
    
Maps numerical MetaphorIDs to corresponding words (e.g., 0 to 'road', 1 to 'candle', etc.).
Saves the result as 'train-1_cleaned_mapped.csv'.

13. Handling Imbalanced Dataset
    
Identifies rows where the 'label' is False.
Duplicates these rows twice to balance the dataset.
Combines the original and duplicated data.
Saves the balanced dataset as 'train-1_cleaned_balanced.csv'.

15. Remapping Words to MetaphorID
    
Converts the word representations back to numerical MetaphorIDs.
Saves the final preprocessed dataset as 'train-1_cleaned_balanced_num.csv'.

Key Features:

Modular Approach: Each preprocessing step is implemented separately, allowing for easy modification or omission of specific steps.
Data Integrity: The code maintains the structure of the original dataset while cleaning and transforming the text data.
Imbalance Handling: Addresses class imbalance by duplicating minority class samples.
Reversible Transformations: The MetaphorID mapping is reversible, allowing for both human-readable and numerical representations.



CNN_LSTM.ipynb implements a complex deep learning model for text classification using a combination of convolutional neural networks (CNNs) and long short-term memory (LSTM) networks for the metaphor detection task. Here's a comprehensive report on the code:

Data Preprocessing:

The code starts by importing necessary libraries and downloading required NLTK data.

Text preprocessing functions are defined:

get_wordnet_pos: Maps POS tags to WordNet POS tags.
preprocess_text: Applies text cleaning, tokenization, lemmatization, and stemming.

The input CSV file is read and preprocessed:

Duplicate rows are removed.
Text is processed using the preprocess_text function.
Empty rows are filtered out.
Boolean labels are converted to strings.

Model Architecture:

The create_complex_model function defines a sophisticated neural network:

Input layer with embedding.
Three parallel convolutional layers with different kernel sizes.
Concatenation of convolutional outputs.
Global max pooling.
Five dense layers with residual connections, layer normalization, and dropout.
Reshape layer.
Two bidirectional LSTM layers.
Final dense layers for classification.

Training Process

Data is split into training and testing sets.
Text is tokenized and padded.
Labels are one-hot encoded.
The model is compiled with Adam optimizer and custom F1 score metric.

Training includes:

Learning rate reduction on plateau.
Early stopping based on accuracy threshold.
Batch size of 32 and up to 100 epochs.

Evaluation:

The model's performance is evaluated using:

Accuracy
F1 score (macro)
Precision (macro)
Recall (macro)

Key Features

Advanced text preprocessing with lemmatization and stemming.
Complex model architecture combining CNNs and LSTMs.
Residual connections and layer normalization for better gradient flow.
Custom F1 score metric for imbalanced datasets.
Early stopping based on accuracy threshold.
Learning rate reduction to fine-tune training.
