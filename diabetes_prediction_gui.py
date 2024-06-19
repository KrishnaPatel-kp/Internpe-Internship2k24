import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separate features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Compute the average parameters for non-diabetic individuals
non_diabetic_data = diabetes_dataset[diabetes_dataset['Outcome'] == 0]
average_parameters = non_diabetic_data.mean()

# Function to display normal parameters
def display_normal_parameters():
    param_text = "Average parameters for non-diabetic individuals:\n"
    for feature, value in average_parameters.items():
        param_text += f"{feature}: {value:.2f}\n"
    return param_text

# Function to get user input from the GUI
def get_user_input():
    input_data = []
    try:
        input_data.append(float(pregnancies_entry.get()))
        input_data.append(float(glucose_entry.get()))
        input_data.append(float(bloodpressure_entry.get()))
        input_data.append(float(skinthickness_entry.get()))
        input_data.append(float(insulin_entry.get()))
        input_data.append(float(bmi_entry.get()))
        input_data.append(float(dpf_entry.get()))
        input_data.append(float(age_entry.get()))
        return input_data
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
        return None

# Function to compare user input with normal parameters
def compare_with_normal(input_data):
    comparison_df = pd.DataFrame({
        'Feature': average_parameters.index,
        'Average Non-Diabetic': average_parameters.values,
        'User Input': input_data
    })
    
    comparison_df.set_index('Feature', inplace=True)
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Comparison of User Input with Average Non-Diabetic Parameters')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.show()

# Function to predict diabetes
def predict_diabetes():
    input_data = get_user_input()
    if input_data is None:
        return

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    
    prediction = classifier.predict(std_data)
    if prediction[0] == 0:
        result_text = 'The person is not diabetic'
    else:
        result_text = 'The person is diabetic'
    
    result_label.config(text=result_text)
    compare_with_normal(input_data)

# Creating the Tkinter GUI
root = tk.Tk()
root.title("Diabetes Prediction Tracker")

# Adding a text box to display normal parameters
normal_params_text = tk.Text(root, height=10, width=50)
normal_params_text.insert(tk.END, display_normal_parameters())
normal_params_text.config(state=tk.DISABLED)
normal_params_text.grid(row=0, column=0, columnspan=2, pady=10)

# Adding input fields for user parameters
tk.Label(root, text="Pregnancies").grid(row=1, column=0, padx=10, pady=5)
pregnancies_entry = tk.Entry(root)
pregnancies_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Glucose").grid(row=2, column=0, padx=10, pady=5)
glucose_entry = tk.Entry(root)
glucose_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="BloodPressure").grid(row=3, column=0, padx=10, pady=5)
bloodpressure_entry = tk.Entry(root)
bloodpressure_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="SkinThickness").grid(row=4, column=0, padx=10, pady=5)
skinthickness_entry = tk.Entry(root)
skinthickness_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Insulin").grid(row=5, column=0, padx=10, pady=5)
insulin_entry = tk.Entry(root)
insulin_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="BMI").grid(row=6, column=0, padx=10, pady=5)
bmi_entry = tk.Entry(root)
bmi_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="DiabetesPedigreeFunction").grid(row=7, column=0, padx=10, pady=5)
dpf_entry = tk.Entry(root)
dpf_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Age").grid(row=8, column=0, padx=10, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=8, column=1, padx=10, pady=5)

# Adding a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Diabetes", command=predict_diabetes)
predict_button.grid(row=9, column=0, columnspan=2, pady=20)

# Label to display prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.grid(row=10, column=0, columnspan=2, pady=20)

# Running the Tkinter event loop
root.mainloop()
