# nlp

This repository includes the notebooks which handles the basic things, likes algorithms and basic tools for doing NLP (Natural Language Processing) within each specific folder.
Mainly this is intended for the beginners who want to start NLP stuffs. It shows what kinds of things he/she should be familiar with with details explanation.

1. Visualization with Matplotlib
  - with the notebooks to see different basic visualization with matplotlib in each cell.

2. Topic Modeling
  - with the notebooks to do topic modeling with Latent Dirichlet Allocation.
  
  
Notebooks

[tokenize_basic_tensorflow_keras.ipynb](tokenize_basic_tensorflow_keras.ipynb) - Notebook with basic tokenization code to tokenize the sentences with spaces using tensorflow and keras

[WordNet.ipynb](WordNet.ipynb) - checking synonyms and hypernyms of WordNet from NLTK

[preprocessing.ipynb](preprocessing.ipynb) - normalizing and tokenizing the tweets including processing with stopwords, punctuations, stemming, lowercase and hyperlinks, needs to import [utils.py](utils.py)

[utils.py](utils.py) - the utility file to be imported in [preprocessing.ipynb](preprocessing.ipynb), [building_and_visualizing_word_frequencies.ipynb](building_and_visualizing_word_frequencies.ipynb) 

[linear_algebra.ipynb](linear_algebra.ipynb) - the notebook how to do linear algebra with vectors and matrices with numpy

[manipulating_word_embeddings.ipynb](manipulating_word_embeddings.ipynb) - to see how word vectors works and find the relations betweens words

[building_and_visualizing_word_frequencies.ipynb](building_and_visualizing_word_frequencies.ipynb) -  to create word frequencies for feature extraction, needs to import [utils.py](utils.py)

[Explanation_PCA.ipynb](Explanation_PCA.ipynb) - Explaining PCA, based on the Singular Value Decomposition (SVD) of the Covariance Matrix of the original dataset, related to Eigenvalues and Eigenvectors which are used as [The Rotation Matrices](The Rotation Matrix.pdf)
