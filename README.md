# Naive Bayes: Spam Classification from Scratch to scikit-learn

A Jupyter notebook that builds up **Naive Bayes** from first principles (Bayes' theorem, the naive independence assumption, Laplace smoothing, and the bias-variance trade-off) and applies **Multinomial Naive Bayes** to an **email/SMS spam classification** problem.

---

## Notebook

| Notebook | Description |
| --- | --- |
| [`Naive_Bayes.ipynb`](Naive_Bayes.ipynb) | Math intuition, worked examples, and a scikit-learn implementation of Naive Bayes for spam detection |

---

## Use case: Spam Classification

You play a data scientist building a model that decides whether a message is **spam** (class 1) or **ham**, meaning not spam (class 0), based on its text. Certain words ("lottery", "prize", "Nigerian prince") are strong signals, and Naive Bayes turns that intuition into probabilities.

---

## What you'll learn

### 1. Problem setup & EDA
- Representing a text as a sequence of words
- Loading and exploring the data, including its **class imbalance**
- Using preprocessed (cleaned) text for modeling
- Train/test split (75/25)

### 2. Mathematical intuition
- Goal: compare `P(y=1 | text)` and `P(y=0 | text)` and pick the larger one
- A refresher on **conditional probability** and **Bayes' theorem**, with a cricket example
- Applying Bayes' theorem to text, and dropping the shared denominator since we only compare the two values
- **Class priors** and **likelihoods**

### 3. The naive assumption
- Why computing `P(text | y)` directly is impractical
- **Conditional independence**: words are assumed independent *given the class*
- Why the assumption is "naive", and the common interview pitfall of forgetting that the independence is conditioned on the class
- Estimating `P(word | class)` from the training data

### 4. Laplace (additive) smoothing
- What goes wrong when a test message contains a word never seen in training (a zero probability wipes out the whole product)
- Fixing it by adding a constant **α** to the numerator and **Cα** to the denominator, and a worked example

### 5. Bias-variance trade-off
- **α is a hyperparameter**
  - Large α: likelihoods are pushed toward a uniform value, so predictions rely on the class priors, which means **underfitting**
  - Small α: likelihoods approach the raw counts, which means **overfitting**

### 6. Multinomial Naive Bayes
- Bernoulli-style features (word present or absent) vs. **word counts**
- Likelihoods conditioned on how many times a word occurs

### 7. Code walkthrough (scikit-learn)
- `CountVectorizer` to build a sparse bag-of-words matrix
- `MultinomialNB` with `GridSearchCV` over `alpha` values `[0.01, 0.1, 1, 10]`
- Using **F1 score** as the metric because the data is imbalanced
- Evaluating train and test F1 for the best model

---

## Dataset

| File | Description |
| --- | --- |
| `spam_clean.csv` | Raw messages with a `type` column (`spam` / `ham`); read with `encoding='latin-1'` |
| `processed_spam_data.pkl` | Preprocessed data with a `cleaned_message` column, used for modeling |

Both files are downloaded via `gdown` inside the notebook. The text-cleaning steps are summarized in the notebook and covered in a separate companion notebook.

---

## Getting started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab (or Google Colab)
- Helpful background: basic probability, classification metrics (especially F1), and cross-validation

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install numpy pandas matplotlib scikit-learn gdown jupyter
```

### Run

```bash
jupyter notebook Naive_Bayes.ipynb
```

> If `gdown` throws an error, try `pip install gdown==4.6.0`.

---

## Libraries used

- **NumPy**, **pandas**: data handling
- **Matplotlib**: plots
- **scikit-learn**: `CountVectorizer`, `MultinomialNB`, `GridSearchCV`, `train_test_split`, metrics
- **gdown**: fetching data files

---

## Repository structure

```
.
├── Naive_Bayes.ipynb
└── README.md
```

---

## Key takeaways

- Naive Bayes classifies by comparing `P(class) × Π P(word | class)` across classes, and the shared denominator can be ignored.
- The **naive assumption** is that words are independent **given the class**. It's rarely exactly true, but it works well in practice.
- **Laplace smoothing** prevents zero probabilities for unseen words.
- The smoothing constant **α** controls the bias-variance trade-off: larger means underfit, smaller means overfit.
- On imbalanced data, evaluate with **F1** (or precision/recall) rather than accuracy.
- Naive Bayes is fast to train, works with sparse high-dimensional text features, and makes a strong baseline for text classification.

---

## Additional resources

- [scikit-learn Naive Bayes guide](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [`CountVectorizer` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

---

## Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## License

Add your preferred license here (for example, MIT).
