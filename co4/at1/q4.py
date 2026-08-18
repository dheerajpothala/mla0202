import sklearn_crfsuite

print("Dheeraj Krushna/192525230")

train_sentences = [
    [("Hello", "O"), ("this", "O"), ("is", "O"), ("John", "CUST_NAME"), ("Smith", "CUST_NAME"),
     ("my", "O"), ("order", "O"), ("ORD1023", "ORDER_ID"), ("for", "O"), ("Laptop", "PRODUCT")],
    [("Hi", "O"), ("I", "O"), ("am", "O"), ("Sarah", "CUST_NAME"), ("Lee", "CUST_NAME"),
     ("order", "O"), ("ORD5566", "ORDER_ID"), ("about", "O"), ("Headphones", "PRODUCT")],
    [("This", "O"), ("is", "O"), ("Mike", "CUST_NAME"), ("Ross", "CUST_NAME"),
     ("order", "O"), ("ORD7788", "ORDER_ID"), ("regarding", "O"), ("Monitor", "PRODUCT")]
]

def word_features(sentence, i):
    word = sentence[i][0]
    features = {
        "word.lower()": word.lower(),
        "word.istitle()": word.istitle(),
        "word.isupper()": word.isupper(),
        "word.isdigit()": word.isdigit(),
        "word[:3]": word[:3],
        "BOS": i == 0,
        "EOS": i == len(sentence) - 1
    }
    return features

def sentence_features(sentence):
    return [word_features(sentence, i) for i in range(len(sentence))]

def sentence_labels(sentence):
    return [label for (_, label) in sentence]

X_train = [sentence_features(s) for s in train_sentences]
y_train = [sentence_labels(s) for s in train_sentences]

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", max_iterations=100)
crf.fit(X_train, y_train)

test_sentence = [("Hi", ""), ("this", ""), ("is", ""), ("Emma", ""), ("Watson", ""),
                  ("order", ""), ("ORD9911", ""), ("for", ""), ("Keyboard", "")]
X_test = [sentence_features(test_sentence)]
predicted_labels = crf.predict(X_test)[0]

for (word, _), label in zip(test_sentence, predicted_labels):
    print(word, "->", label)

'''
OUTPUT:
Dheeraj Krushna/192525230
Hi -> O
this -> O
is -> O
Emma -> CUST_NAME
Watson -> CUST_NAME
order -> O
ORD9911 -> ORDER_ID
for -> O
Keyboard -> PRODUCT
'''
