# Text Summarization

This repository contains a collection of several automatic text summarization techniques.

## Install

Create a new environment.

Install the necessary packages in the environment:
```
pip install requirements.txt
```

Download required nltk data for stopwords removal and sentence tokenizer:
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Run

You can try out the summarizer by activating the environment and then running the following command:
```
python main.py "My long text ... " --language "english" --summary-length 2
```