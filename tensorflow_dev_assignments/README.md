## Labs and Assignments
C3 - Course 3 of the Specialization

W1 - Week 1 of the Course

# `Natural Language Processing in Tensorflow` of `DeepLearning.AI`
This is assignments collection of `Natural Language Processing in Tensorflow` Course from `DeepLearning.AI` on `Cousera`.


## Proprecessing ( Tokenizing, Padding )
[tokenizer_basic.ipynb](tokenizer_basic.ipynb)
- using basic `tokenizer` in `TensorFlow`

[sequences_basic.ipynb](sequences_basic.ipynb)
- using  `texts_to_sequences` and `pad_sequences` in `TensorFlow`

[sarcasm_data_preprocessing.ipynb](sarcasm_data_preprocessing.ipynb)
- tokenizing and preprocessing [Sarcasm Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home) 

[W1_Assignment.ipynb](C3W1_Assignment.ipynb)
- preprocessing [BBC News Classification Dataset](https://www.kaggle.com/c/learn-ai-bbc/overview)
- removing stopwords, tokening and preprocessing data in `TensorFlow`
 
[W1_Assignment_solution.ipynb](C3W1_Assignment_solution.ipynb)
- The assignment file of Week 1 in `NLP in TF` with the `Solution`

## Embedding
[sarcasm_classifier.ipynb](C3W2_Lab2_sarcasm_classifier.ipynb) 
- includes how GlobalAveragePooling1D() works, 
  how hyperparameters like vocab size, padded sequences size, embedding dimensions can effect the confident of the predictions (based on loss),
  visualization of the results based on model's history 
  and how to check the embedding results from the embedding layers by using [Tensorflow Embedding Projector](https://projector.tensorflow.org/).
  
[imdb_subwords.ipynb](C3W2_Lab3_imdb_subwords.ipynb)
- Comparing `Tokenizer` and `SubwordTextEncoder`
- IMDB review classification and evaluation with pretrained Subword Tokenizer

[W2_Assignment.ipynb](C3W2_Assignment.ipynb)
- The assignment file of Week 2 in `NLP in TF`
- [BBC News Classification Dataset](https://www.kaggle.com/c/learn-ai-bbc/overview)

[W2_Assignment_withSolution.ipynb](C3W2_Assignment_withSolution.ipynb)
- need ot use [bbc-text.csv](bbc-text.csv)
- includes `preprocessing`, 'Tokenization`, 'Model building`, `Evaluation` with `visualization` and creating the file to test with [Tensorflow's Embedding Projector](https://projector.tensorflow.org/).
- uses the `Embedding`, `GlobalAveragePooling1D` and `Dense` layers only.
- compiles with `sparse_categorical_crossentropy` loss function and `adam` optimizer.

## Recurrent Neural Network, LSTM
[single_layer_LSTM.ipynb](C3W3_Lab1_single_layer_LSTM.ipynb)
- training single layer LSTM with the dataset `imdb_reviews/subwords8k` from built-in `tensorflow_datasets`
- prepares the data, builds and complies the model and evaluates it

[multiple_layer_LSTM.ipynb](C3W3_Lab2_multiple_layer_LSTM.ipynb)
- training multiple layer LSTM with dataset `imdb_reviews/subwords8k` from built-in `tensorflow_datasets`
- shows how to parse between LSTM and the difference of outputs
- prepares the data, builds and complies the model and evaluates it

[sarcasm_dataset_bi_lstm.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W3_Lab5_sarcasm_with_bi_LSTM.ipynb)
- Dataset : [News headline dataset for scarcasm detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- embedding, Bidirectional LSTM

## Convolutional Neural Network
[Conv1D_subword8k.ipynb](C3W3_Lab3_Conv1D_subword8k.ipynb)
- training Subword8k Text Data with Convolution 1D Layer

[dnn_lstm_gru_conv1d.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W3_Lab4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb)
- imbd reviews dataset
- comparing the results among DNN, LSTM, GRU and Conv1D

[sarcasm_dataset_conv1d.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W3_Lab6_sarcasm_with_1D_convolutional.ipynb)
- Dataset : [News headline dataset for scarcasm detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- embedding, conv1d

[twitter_sentiment_classification.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W3_Assignment_sentiment_analysis.ipynb)
- Dataset : [sentiment140 dataset](http://help.sentiment140.com/home)
- Sentiment Analysis, Pos vs Neg
- Uses pretrained GLOVE embedding
- Conv1D, Bidirectional LSTM

## Text Generation
[text_generation.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W4_Lab1_textgeneration.ipynb)
- training on [Lanigan's Ball](https://en.wikipedia.org/wiki/Lanigan%27s_Ball), a traditional Irish song

[lyrics_generation.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W4_Lab2_irish_lyrics_generation.ipynb)
- training on more Irish songs
- Bidirectional LSTM

[shakespeare_sonnets_generation.ipynb](https://github.com/yiyichanmyae/nlp/blob/master/tensorflow_dev_assignments/C3W4_Assignment_Shakespeare_sonnets_generation.ipynb)
- Data : [ShakeSpeare Sonnets](https://www.opensourceshakespeare.org/views/sonnets/sonnet_view.php?range=viewrange&sonnetrange1=1&sonnetrange2=154)
- Bidriection LSTM 

# `Natural Language Processing` Specialization of `DeepLearning.AI`

[LogisticRegression_Assignment.ipynb](LogisticRegression_W1_Assignment.ipynb)
- assignment notebook without solution
- if you wanna check the solution, please see [LogisticRegression_fromScratch.ipynb](../LogisticRegression_fromScratch.ipynb)
