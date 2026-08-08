"""Q3. Naive Bayes - Zero Probability Problem (simplified)"""
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Name: Dheeraj k | ID: 192525230\n")

# ---- Create small dataset ----
data = [
    ("great product love it", "positive"), ("amazing quality very happy", "positive"),
    ("excellent service fast", "positive"), ("love this item works great", "positive"),
    ("happy with purchase", "positive"),
    ("terrible product broke fast", "negative"), ("bad quality very disappointed", "negative"),
    ("worst service ever", "negative"), ("hate this item stopped working", "negative"),
    ("disappointed with purchase", "negative"),
]
df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("naive_bayes_reviews_dataset.csv", index=False)
test_sentence = "excellent product but slow service"

def train_nb(df, smoothing=0.0):
    vocab = set(w for t in df.text for w in t.split())
    priors, counts, totals = {}, {}, {}
    for c in df.label.unique():
        texts = df[df.label == c].text
        priors[c] = len(texts) / len(df)
        cnt = defaultdict(int)
        for t in texts:
            for w in t.split(): cnt[w] += 1
        counts[c], totals[c] = cnt, sum(cnt.values())
    prob = lambda w, c: (counts[c].get(w, 0) + smoothing) / (totals[c] + smoothing * len(vocab))
    return priors, prob

def score(priors, prob):
    out = {}
    for c in priors:
        s = priors[c]
        for w in test_sentence.split(): s *= prob(w, c)
        out[c] = s
    return out

# ---- i)/ii) WITHOUT smoothing -> zero probability issue ----
priors, prob_raw = train_nb(df, smoothing=0.0)
print("WITHOUT smoothing:", score(priors, prob_raw))
print("Issue: 'excellent' never appears in 'negative' -> P(excellent|negative)=0",
      "-> multiplies whole class score to 0 (zero-frequency problem)\n")

# ---- iii) Fix: Laplace smoothing ----
priors_s, prob_smooth = train_nb(df, smoothing=1.0)
scores_s = score(priors_s, prob_smooth)
total = sum(scores_s.values())
print("WITH Laplace smoothing:", {k: round(v/total, 3) for k, v in scores_s.items()})

# ---- Verify with sklearn ----
vec = CountVectorizer()
X = vec.fit_transform(df.text)
nb = MultinomialNB(alpha=1.0).fit(X, df.label)
print("sklearn prediction:", nb.predict(vec.transform([test_sentence]))[0])
