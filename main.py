import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, filepath):
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)


    def compute_prior(self):
        total = len(self.df)
        priors = {}

        for label in self.df["buys_computer"].unique():
            count = len(self.df[self.df["buys_computer"] == label])
            priors[label] = count / total

        return priors


    def conditional_probability(self, column, value, class_label):
        subset = self.df[self.df["buys_computer"] == class_label]
        count_value = len(subset[subset[column] == value])
        return count_value / len(subset)


    def predict(self, sample):
        priors = self.compute_prior()
        probabilities = {}

        for label in priors:
            prob = priors[label]
            for column, value in sample.items():
                prob *= self.conditional_probability(column, value, label)
            probabilities[label] = prob

        return max(probabilities, key=probabilities.get)


def run_classifier():
    model = NaiveBayesClassifier("allelectronics.csv")
    sample = {
        "age": "youth",
        "income": "medium",
        "student": "yes",
        "credit_rating": "fair"
    }
    prediction = model.predict(sample)
    return prediction


if __name__ == "__main__":
    result = run_classifier()
    print("\nPredicted Class:", result)