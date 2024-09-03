from data_arrange import final_df
from model import create_pyramidal_model
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Number of iterations
n_iterations = 40

# Preparing the data
X = final_df.drop(['PHC_Diagnosis', 'PHC_Race'], axis=1).values
y = final_df['PHC_Diagnosis'].values
races = final_df['PHC_Race'].values

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Store AUROC values for each race and the entire test dataset across all iterations
auroc_values_race3 = []
auroc_values_race5 = []
auroc_values_entire_test = []

for iteration in range(n_iterations):
    # Train-test split
    X_train, X_test, y_train, y_test, races_train, races_test = train_test_split(
        X, y_categorical, races, test_size=0.2, random_state=42 + iteration)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []

    # Train the model using cross-validation
    for train_index, val_index in skf.split(X_train, np.argmax(y_train, axis=1)):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = create_pyramidal_model(input_dim=X_train.shape[1], output_dim=y_categorical.shape[1])

        model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=32, validation_data=(X_val_fold, y_val_fold), verbose=1)
        val_scores.append(model.evaluate(X_val_fold, y_val_fold, verbose=0))

    print(f'Iteration {iteration + 1} - Cross-validation scores: {val_scores}')
    print(f'Iteration {iteration + 1} - Average Validation Accuracy: {np.mean([score[1] for score in val_scores]):.4f}')

    # Test the model on separate race categories and store AUROC
    for race, auroc_values in [(3, auroc_values_race3), (5, auroc_values_race5)]:
        X_test_race = X_test[races_test == race]
        y_test_race = y_test[races_test == race]

        # Predict and calculate AUROC
        y_pred = model.predict(X_test_race)
        fpr, tpr, _ = roc_curve(y_test_race.ravel(), y_pred.ravel())
        roc_auc = auc(fpr, tpr)
        auroc_values.append(roc_auc)
    
    # Calculate AUROC for the entire test dataset
    y_pred_entire_test = model.predict(X_test)
    fpr_entire_test, tpr_entire_test, _ = roc_curve(y_test.ravel(), y_pred_entire_test.ravel())
    roc_auc_entire_test = auc(fpr_entire_test, tpr_entire_test)
    auroc_values_entire_test.append(roc_auc_entire_test)

# Calculate median and average AUROC for each race and the entire test dataset
median_auroc_mixture1 = np.median(auroc_values_race3)
average_auroc_mixture1 = np.mean(auroc_values_race3)
median_auroc_mixture2 = np.median(auroc_values_race5)
average_auroc_mixture2 = np.mean(auroc_values_race5)
median_auroc_mixture0 = np.median(auroc_values_entire_test)
average_auroc_mixture0 = np.mean(auroc_values_entire_test)

# Write the median and average AUROC values to the file
with open("results_mixture.txt", "w") as file:
    file.write(f'Median AUROC for Mixture 0 (Entire Test Dataset): {median_auroc_mixture0:.4f}\n')
    file.write(f'Average AUROC for Mixture 0 (Entire Test Dataset): {average_auroc_mixture0:.4f}\n')
    file.write(f'Median AUROC for Mixture 1 (Race 3): {median_auroc_mixture1:.4f}\n')
    file.write(f'Average AUROC for Mixture 1 (Race 3): {average_auroc_mixture1:.4f}\n')
    file.write(f'Median AUROC for Mixture 2 (Race 5): {median_auroc_mixture2:.4f}\n')
    file.write(f'Average AUROC for Mixture 2 (Race 5): {average_auroc_mixture2:.4f}\n')

# Plot AUROC for Race 3
plt.figure()
plt.plot(range(n_iterations), auroc_values_race3, color='blue', lw=2, label='AUROC for Race 3')
plt.axhline(y=median_auroc_mixture1, color='blue', linestyle='--', label=f'Median AUROC for Mixture 1 (Race 3) = {median_auroc_mixture1:.2f}')
plt.axhline(y=average_auroc_mixture1, color='blue', linestyle=':', label=f'Average AUROC for Mixture 1 (Race 3) = {average_auroc_mixture1:.2f}')
plt.xlabel('Iteration')
plt.ylabel('AUROC')
plt.title('AUROC across Iterations for Race 3')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("mixture_1.png")
plt.show()

# Plot AUROC for Race 5
plt.figure()
plt.plot(range(n_iterations), auroc_values_race5, color='green', lw=2, label='AUROC for Race 5')
plt.axhline(y=median_auroc_mixture2, color='green', linestyle='--', label=f'Median AUROC for Mixture 2(Race 5) = {median_auroc_mixture2:.2f}')
plt.axhline(y=average_auroc_mixture2, color='green', linestyle=':', label=f'Average AUROC for Mixture 2(Race 5) = {average_auroc_mixture2:.2f}')
plt.xlabel('Iteration')
plt.ylabel('AUROC')
plt.title('AUROC across Iterations for Race 5')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("mixture_2.png")
plt.show()

# Plot AUROC for Entire Test Dataset
plt.figure()
plt.plot(range(n_iterations), auroc_values_entire_test, color='red', lw=2, label='AUROC for Entire Test Dataset')
plt.axhline(y=median_auroc_mixture0, color='red', linestyle='--', label=f'Median AUROC for Mixture 0 (Entire Test Dataset) = {median_auroc_mixture0:.2f}')
plt.axhline(y=average_auroc_mixture0, color='red', linestyle=':', label=f'Average AUROC for Mixture 0 (Entire Test Dataset) = {average_auroc_mixture0:.2f}')
plt.xlabel('Iteration')
plt.ylabel('AUROC')
plt.title('AUROC across Iterations for Entire Test Dataset')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("mixture_0.png")
plt.show()
