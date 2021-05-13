# Topic Modeling

This repository contains a collection of several topic modeling techniques.

An overview of well-known topic modeling techniques is presented in the jupyter notebook [topic_modeling_techniques_overview.ipynb](topic_modeling_techniques_overview.ipynb).

## Install

Create a new environment with the necessary packages:
```
conda env create -f environment.yml
```

Download required nltk data for stopwords removal and sentence tokenizer:
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```