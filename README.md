# Naive Bayes Classification

A simple Naive Bayes classifier built from scratch with Python and pandas. The project uses categorical customer attributes to predict whether a customer will buy a computer.

## Dataset

The included `allelectronics.csv` dataset contains 14 records with these fields:

| Column | Description |
| --- | --- |
| `RID` | Record identifier |
| `age` | Customer age group |
| `income` | Customer income level |
| `student` | Whether the customer is a student |
| `credit_rating` | Customer credit rating |
| `buys_computer` | Target class (`yes` or `no`) |

## How it works

The classifier:

1. Calculates the prior probability of each target class.
2. Calculates conditional probabilities for every feature value given a class.
3. Multiplies the prior and conditional probabilities using the Naive Bayes independence assumption.
4. Returns the class with the highest probability.

The implementation is intentionally small and educational. It does not use scikit-learn or apply Laplace smoothing, so an unseen feature/class combination can reduce a class probability to zero.

## Requirements

- Python 3.8+
- pandas

Install the dependency with:

```bash
python -m pip install pandas
```

## Run the project

Clone the repository, move into its directory, and run:

```bash
python main.py
```

The sample currently defined in `main.py` is:

```python
sample = {
    "age": "youth",
    "income": "medium",
    "student": "yes",
    "credit_rating": "fair"
}
```

Expected output:

```text
Loading dataset...

Predicted Class: yes
```

## Try another prediction

Edit the `sample` dictionary in `run_classifier()` using categorical values found in the dataset, then run the script again.

## Project structure

```text
.
|-- allelectronics.csv  # Training data
|-- main.py             # Classifier and example prediction
`-- README.md           # Project documentation
```
