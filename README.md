# metaphor-detection-cnn-lstm
Metaphor detection using cnn lstm


CNN_LSTM.ipynb implements a complex deep learning model for text classification using a combination of convolutional neural networks (CNNs) and long short-term memory (LSTM) networks for the metaphor detection task. Here's a comprehensive report on the code:

#Data Preprocessing#:

The code starts by importing necessary libraries and downloading required NLTK data.

#Text preprocessing functions are defined:

get_wordnet_pos: Maps POS tags to WordNet POS tags.
preprocess_text: Applies text cleaning, tokenization, lemmatization, and stemming.

#The input CSV file is read and preprocessed:

Duplicate rows are removed.
Text is processed using the preprocess_text function.
Empty rows are filtered out.
Boolean labels are converted to strings.

#Model Architecture:

#The create_complex_model function defines a sophisticated neural network:

Input layer with embedding.
Three parallel convolutional layers with different kernel sizes.
Concatenation of convolutional outputs.
Global max pooling.
Five dense layers with residual connections, layer normalization, and dropout.
Reshape layer.
Two bidirectional LSTM layers.
Final dense layers for classification.

#Training Process

Data is split into training and testing sets.
Text is tokenized and padded.
Labels are one-hot encoded.
The model is compiled with Adam optimizer and custom F1 score metric.

#Training includes:

Learning rate reduction on plateau.
Early stopping based on accuracy threshold.
Batch size of 32 and up to 100 epochs.

#Evaluation:

#The model's performance is evaluated using:

Accuracy
F1 score (macro)
Precision (macro)
Recall (macro)

#Key Features

Advanced text preprocessing with lemmatization and stemming.
Complex model architecture combining CNNs and LSTMs.
Residual connections and layer normalization for better gradient flow.
Custom F1 score metric for imbalanced datasets.
Early stopping based on accuracy threshold.
Learning rate reduction to fine-tune training.
