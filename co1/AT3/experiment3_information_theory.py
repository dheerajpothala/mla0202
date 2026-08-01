import math
import pandas as pd

data = {
    "Outlook":     ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity":    ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind":        ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "PlayTennis":  ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"],
}
df = pd.DataFrame(data)
target = "PlayTennis"

def entropy(s):
    p = s.value_counts(normalize=True)
    return -sum(p * p.apply(math.log2))

H = entropy(df[target])
print(f"H(PlayTennis)={H:.4f}")

gains = {}
for attr in ["Outlook", "Temperature", "Humidity", "Wind"]:
    weighted = sum((len(sub) / len(df)) * entropy(sub[target]) for _, sub in df.groupby(attr))
    gains[attr] = H - weighted
    print(f"IG({attr})={gains[attr]:.4f}  H(S|{attr})={weighted:.4f}")

best = max(gains, key=gains.get)
print(f"Best attribute: {best} (IG={gains[best]:.4f})")
