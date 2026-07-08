import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Current Working Directory:")
print(os.getcwd())


dataset_path = "spam.csv"

if not os.path.exists(dataset_path):
    print("Dataset not found!")
    raise SystemExit


data = pd.read_csv(dataset_path, encoding="latin-1")

data = data.iloc[:, :2]
data.columns = ["label", "message"]


data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


X = data["message"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

print("\nTraining Model...\n")

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("="*50)
print("Accuracy :", round(accuracy*100,2), "%")
print("="*50)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))


pickle.dump(model, open("model.pkl","wb"))

print("\nModel Saved Successfully.")


while True:

    sms = input("\nEnter SMS (type exit to quit): ")

    if sms.lower() == "exit":
        break

    prediction = model.predict([sms])

    if prediction[0] == 1:
        print("Prediction : SPAM")
    else:
        print("Prediction : NOT SPAM")