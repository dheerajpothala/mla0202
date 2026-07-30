import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("/home/claude/project/data/sms.tsv", sep="\t", header=None, names=["label", "message"])
X_train_text, X_test_text, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

vec = CountVectorizer(stop_words="english", lowercase=True)
X_train = vec.fit_transform(X_train_text).toarray()
X_test = vec.transform(X_test_text)

prior_spam = (y_train == "spam").mean()
prior_ham = (y_train == "ham").mean()

vocab = np.array(vec.get_feature_names_out())
wc_spam = X_train[y_train.values == "spam"].sum(axis=0)
wc_ham = X_train[y_train.values == "ham"].sum(axis=0)
tot_spam, tot_ham, V = wc_spam.sum(), wc_ham.sum(), len(vocab)

def word_p(w, cls):
    idx = np.where(vocab == w)[0]
    c = (wc_spam[idx[0]] if cls == "spam" else wc_ham[idx[0]]) if len(idx) else 0
    tot = tot_spam if cls == "spam" else tot_ham
    return (c + 1) / (tot + V)

def classify(msg):
    words = [w.lower() for w in msg.split() if w.isalpha()]
    ls, lh = np.log(prior_spam), np.log(prior_ham)
    for w in words:
        ls += np.log(word_p(w, "spam"))
        lh += np.log(word_p(w, "ham"))
    m = max(ls, lh)
    ps, ph = np.exp(ls - m), np.exp(lh - m)
    ps, ph = ps / (ps + ph), ph / (ps + ph)
    return ("spam" if ps > ph else "ham"), ps, ph

msg = "WINNER!! You have been selected to receive a FREE cash prize call now"
label, ps, ph = classify(msg)
print(f"P(Spam)={prior_spam:.4f}  P(Ham)={prior_ham:.4f}")
print(f'Message: "{msg}"')
print(f"P(Spam|Msg)={ps:.6f}  P(Ham|Msg)={ph:.6f}  ->  {label.upper()}")

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy={accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred, labels=["spam", "ham"]))
print(classification_report(y_test, y_pred))
