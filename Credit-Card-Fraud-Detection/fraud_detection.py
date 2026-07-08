import sys
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Current Working Directory:")
print(os.getcwd())


train_path = r"fraudTrain.csv"
test_path = r"fraudTest.csv"


if not os.path.exists(train_path):
    print("Training file not found!")
    sys.exit()

if not os.path.exists(test_path):
    print("Testing file not found!")
    sys.exit()


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\nTraining Data Shape:", train.shape)
print("Testing Data Shape :", test.shape)


drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "dob",
    "trans_num"
]

train.drop(columns=drop_columns, inplace=True, errors="ignore")
test.drop(columns=drop_columns, inplace=True, errors="ignore")


categorical = train.select_dtypes(include="object").columns

for col in categorical:
    train[col] = train[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes


X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

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

    choice = input("\nDo you want to test a transaction? (yes/no): ")

    if choice.lower() == "no":
        break

    sample = X_test.iloc[[0]]

    prediction = model.predict(sample)

    if prediction[0] == 1:
        print("Prediction : Fraudulent Transaction")
    else:
        print("Prediction : Legitimate Transaction")