from mainfile import final_df
from model import create_pyramidal_model
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Preparing the data
X = final_df.drop(['PHC_Diagnosis', 'PHC_Race'], axis=1).values
y = final_df['PHC_Diagnosis'].values
races = final_df['PHC_Race'].values

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Parameters
iterations = 5
race_categories = [3, 5]

# Initialize lists to store AUROC results
auroc_results = {race: [] for race in race_categories}

for iteration in range(iterations):
    # Train-test split
    X_train, X_test, y_train, y_test, races_train, races_test = train_test_split(
        X, y_categorical, races, test_size=0.2, random_state=42 + iteration)  # Different random_state for each iteration

    for race in race_categories:
        # Select training data where race is the current race category
        X_train_race = X_train[races_train == race]
        y_train_race = y_train[races_train == race]

        # Select testing data where race is the current race category
        X_test_race = X_test[races_test == race]
        y_test_race = y_test[races_test == race]

        # Cross-validation setup
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_scores = []

        for train_index, val_index in skf.split(X_train_race, np.argmax(y_train_race, axis=1)):
            X_train_fold, X_val_fold = X_train_race[train_index], X_train_race[val_index]
            y_train_fold, y_val_fold = y_train_race[train_index], y_train_race[val_index]

            model = create_pyramidal_model(input_dim=X_train_race.shape[1], output_dim=y_categorical.shape[1])

            model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=1)
            val_scores.append(model.evaluate(X_val_fold, y_val_fold, verbose=0))

        print(f'Cross-validation scores for race {race}, iteration {iteration + 1}: {val_scores}')
        print(f'Average Validation Accuracy for race {race}, iteration {iteration + 1}: {np.mean([score[1] for score in val_scores]):.4f}')

        # Predict and calculate AUROC on the race-specific test data
        y_pred = model.predict(X_test_race)
        fpr, tpr, _ = roc_curve(y_test_race.ravel(), y_pred.ravel())
        roc_auc = auc(fpr, tpr)

        # Store AUROC result
        auroc_results[race].append(roc_auc)

# Calculate and display median and average AUROC for each race across all iterations
for race in race_categories:
    avg_auroc = np.mean(auroc_results[race])
    median_auroc = np.median(auroc_results[race])
    print(f'Race {race} - Average AUROC over {iterations} iterations: {avg_auroc:.4f}')
    print(f'Race {race} - Median AUROC over {iterations} iterations: {median_auroc:.4f}')