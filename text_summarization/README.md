# Text Summarization

This repository contains a collection of several automatic text summarization techniques.

An overview of well-known text summarization techniques is presented in the jupyter notebook [text_summarization_techniques_overview.ipynb](text_summarization_techniques_overview.ipynb).

Some techniques have been implemented from scratch. Their implementation can be found inside the summarizers/ folder and their code can be run by using the [main.py](main.py) script.

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