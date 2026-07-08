import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Current Working Directory:")
print(os.getcwd())


dataset = "Churn_Modelling.csv"

if not os.path.exists(dataset):
    print("Dataset not found!")
    raise SystemExit


data = pd.read_csv(dataset)

print("\nDataset Shape :", data.shape)


data.drop(
    columns=["RowNumber","CustomerId","Surname"],
    inplace=True
)


encoder = LabelEncoder()

data["Gender"] = encoder.fit_transform(data["Gender"])
data["Geography"] = encoder.fit_transform(data["Geography"])


X = data.drop("Exited", axis=1)
y = data["Exited"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

print("\nTraining Model...\n")

model.fit(X_train, y_train)


prediction = model.predict(X_test)


accuracy = accuracy_score(y_test, prediction)

print("="*50)
print("Accuracy :", round(accuracy*100,2),"%")
print("="*50)

print("\nClassification Report\n")
print(classification_report(y_test,prediction))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,prediction))


pickle.dump(model, open("model.pkl","wb"))

print("\nModel Saved Successfully")



while True:

    ch = input("\nDo you want to test a customer? (yes/no): ")

    if ch.lower()=="no":
        break

    row = int(input("Enter row number (0-{}): ".format(len(X_test)-1)))

    sample = X_test.iloc[[row]]

    ans = model.predict(sample)

    if ans[0]==1:
        print("\nPrediction : Customer Will Leave")
    else:
        print("\nPrediction : Customer Will Stay")