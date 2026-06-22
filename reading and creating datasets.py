# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:53:35 2026

@author: Eshaan
"""

# %% Imports

import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from statistics import mode

# %% Lets begin with data resampling

data = pd.read_csv('C:/Users/eshph/OneDrive/Desktop/ongoing python projects/Disease predictor/improved_disease_dataset.csv')

encoder = LabelEncoder()    ## From scikitlearn, converts string data to 0,1,2,3 .. n
data["disease"] = encoder.fit_transform(data["disease"])  ## changing the disease to 1,2,3, etc. need inverse_transform to convert back

X = data.iloc[:, :-1]    ### selecting all data from 0 to 100 x axis, 0 to except last column?  this is extracting
y = data.iloc[:, -1]        ## This is for selecting last column

plt.figure()
sns.countplot(x=y)          ## Histogram, x is x-axis. y is data to count
plt.show()

ros = RandomOverSampler(random_state=42)            ## 42 is seed, this function is creating a randomiser kind off. 42 can be any variable, but this is seed no.
X_resampled, y_resampled = ros.fit_resample(X, y)       ## now using randomiser developed earlier, we make it so all datasets are equal in results. Like in this
## no 2 disease is lowest. After randomiser, the resampled ones all be equal , It adds new random copies. Check this new histogram

sns.countplot(x=y_resampled)        ## compare with countplot(x = y), now all equal


# %%  cleaning up data


if 'fever' in X_resampled.columns:       print("yes")           ## checking column names

    
    
X_resampled = X_resampled.fillna(0)         ## to convert na to 0


y_resampled.shape           ## This tells shape or number of rows/columns of data 



# %% starting models

models = {    "Decision Tree": DecisionTreeClassifier(),    "Random Forest": RandomForestClassifier(),    "SVM": SVC()}


## Ok so this seems a dictionary, calling up function when named

# %%


cv_scoring = 'accuracy'     ## try f1_weighted or roc_auc_ovr    ## Some kind of scoring method? it also mentioned one can use f1 weighted, roc_auc_over, but dont know how
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)           ## they say this is for splitting data 
                                                                                        ## But I am not sure
# %%   Running model test?

for model_name, model in models.items():                ## calling previous model which we kind of created
    try:                                                    ## error handling? try catch
        scores = cross_val_score(                   ## This is some function, google says here training happens under the hood. Then it is discarded?
            model,                                  ## here we input the model
            X_resampled,                                ## The data
            y_resampled,                                ## The output
            cv=stratified_kfold,                        ## previously created, we split data in 5 sets, in random
            scoring=cv_scoring,                         ## We testing for accuracy in this case, let me try f1-weighted
            n_jobs=-1,                                  ## no. of cores
            error_score='raise'                             ## types of error scores
        )
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}")
    except Exception as e:                          ## try catch , catch
        print("=" * 50)
        print(f"Model: {model_name} failed with error:")
        print(e)



# %% DONT RUN THIS PART. This is just test


#scores = cross_val_score(DecisionTreeClassifier(),X_resampled, y_resampled, cv=stratified_kfold, scoring=cv_scoring,  n_jobs=-1,    error_score='raise' )
#print("=" * 50)
#print(f"Model: {model_name}")
#print(f"Scores: {scores}")
#print(f"Mean Accuracy: {scores.mean():.4f}")





# %%   Testing confusion Matrix
## SVC

svm_model = SVC()   ## assigning a variable the model, here the training weights will be stored
svm_model.fit(X_resampled, y_resampled)    ## trying to fit? Google says another training here
svm_preds = svm_model.predict(X_resampled)    ## creating prediction fit

cf_matrix_svm = confusion_matrix(y_resampled, svm_preds)    ## creating matrix

plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_svm, annot=True, fmt="d")
plt.title("Confusion Matrix for SVM Classifier")
plt.show()

print(f"SVM Accuracy: {accuracy_score(y_resampled, svm_preds) * 100:.2f}%")


# %% Testing confusion matrix of Naive Baysian


nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_preds = nb_model.predict(X_resampled)

cf_matrix_nb = confusion_matrix(y_resampled, nb_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_nb, annot=True, fmt="d")
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.show()

print(f"Naive Bayes Accuracy: {accuracy_score(y_resampled, nb_preds) * 100:.2f}%")



# %%   Random Forest Confusion matrix



rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
rf_preds = rf_model.predict(X_resampled)

cf_matrix_rf = confusion_matrix(y_resampled, rf_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_rf, annot=True, fmt="d")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()

print(f"Random Forest Accuracy: {accuracy_score(y_resampled, rf_preds) * 100:.2f}%")


# %%
## Note to self, The confusion matrix done above seem to predict, not train. And they are predicting on same data as training data
## Question: Where are we training them? 

## They seem to be training in .fit function and cross_val_score function    


# %%  Combining all models for robustness? more accuracy?

final_preds = [mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]   ## So this is taking up most frequent value from all the models, and putting it in the confusion matrix

cf_matrix_combined = confusion_matrix(y_resampled, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix_combined, annot=True, fmt="d")
plt.title("Confusion Matrix for Combined Model")
plt.show()


print(f"Combined Model Accuracy: {accuracy_score(y_resampled, final_preds) * 100:.2f}%")


# %%  Creating predictor function???

symptoms = X.columns.values   ## extracting column values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}  ## a dictionary, for what? is it same as label encoder?

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")  ## enter symptoms
    input_data = [0] * len(symptom_index)       ## Symptoms seems to convert to a 1 row matrix, with yes or no as output
    
    for symptom in input_symptoms:    ## Symptoms seems to convert to a 1 row matrix, with yes or no as output
        if symptom in symptom_index:   ## Symptoms seems to convert to a 1 row matrix, with yes or no as output
            input_data[symptom_index[symptom]] = 1   ## Symptoms seems to convert to a 1 row matrix, with yes or no as output

    input_df = pd.DataFrame([input_data], columns=symptoms)    # this will be actual input

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])
    
    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }
# %%  Testing outputs


print(predict_disease("skin_rash,fever,headache"))


















































