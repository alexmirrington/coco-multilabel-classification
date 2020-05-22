# COCO Multilabel Classification

## Running Instructions

Download the dataset, unzip it and place it in the `input` folder as follows:

```Text
|_ code
    |_ algorithm
    |   |_ main.py
    |   |_ ...
    |_ input
    |   |_ data
    |   |   |_ 0.jpg
    |   |   |_ ...
    |   |   |_ 39999.jpg
    |   |_ test.csv
    |   |_ train.csv
    |_ output
    |   |_ predicted_labels.txt
    |_ results
        |_ ...
```

Install all requirements from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

## Development Environment Setup

Before contributing to the code base, ensure you have the `pre-commit` hook set up properly by running the following command:

```Bash
pre-commit install
```

This ensures that all committed code adheres to `flake8`, `pylint` and `pydocstyle` linter rules. Additionally, I would recommend installing `flake8`, `pylint` and `pydocstyle` so you can minimise warnings and errors in your text editor or IDE while programming so there is less to fix up before committing.
