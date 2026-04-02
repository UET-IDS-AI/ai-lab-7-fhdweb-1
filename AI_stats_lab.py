"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Tokenize
    tokenized = [text.split() for text in texts]

    # Vocabulary
    vocab = set()
    for words in tokenized:
        vocab.update(words)

    vocab = list(vocab)

    # Priors
    priors = {
        1: np.mean(labels == 1),
        0: np.mean(labels == 0)
    }

    # Word counts per class
    word_counts = {
        1: {word: 0 for word in vocab},
        0: {word: 0 for word in vocab}
    }

    total_words = {1: 0, 0: 0}

    for words, label in zip(tokenized, labels):
        for word in words:
            word_counts[label][word] += 1
            total_words[label] += 1

    # Word probabilities (MLE, no smoothing)
    word_probs = {
        1: {},
        0: {}
    }

    for c in [0, 1]:
        for word in vocab:
            if total_words[c] == 0:
                word_probs[c][word] = 0
            else:
                word_probs[c][word] = word_counts[c][word] / total_words[c]

    # Prediction
    test_words = test_email.split()

    log_prob = {0: np.log(priors[0]), 1: np.log(priors[1])}

    for c in [0, 1]:
        for word in test_words:
            prob = word_probs[c].get(word, 0)

            if prob == 0:
                log_prob[c] += -1e9  # simulate log(0)
            else:
                log_prob[c] += np.log(prob)

    prediction = 1 if log_prob[1] > log_prob[0] else 0

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Distance function
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Predict function
    def predict(X1, X2, y2):
        preds = []

        for x in X1:
            distances = [euclidean(x, x_train) for x_train in X2]

            # Get indices of k nearest
            k_idx = np.argsort(distances)[:k]

            k_labels = y2[k_idx]

            # Majority vote
            values, counts = np.unique(k_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.array(preds)

    # Predictions
    train_pred = predict(X_train, X_train, y_train)
    test_pred = predict(X_test, X_train, y_train)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    return train_accuracy, test_accuracy, test_pred
