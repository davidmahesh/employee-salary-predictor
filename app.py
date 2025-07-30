import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Sample dataset
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'EducationLevel': ['Bachelors', 'Masters', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'PhD', 'Bachelors', 'Masters'],
    'Location': ['Delhi', 'Bangalore', 'Hyderabad', 'Delhi', 'Hyderabad', 'Bangalore', 'Delhi', 'Hyderabad', 'Bangalore', 'Delhi'],
    'JobRole': ['Data Scientist', 'ML Engineer', 'Software Developer', 'Data Scientist', 'ML Engineer', 'Software Developer', 'ML Engineer', 'Data Scientist', 'Software Developer', 'ML Engineer'],
    'Skills': ['Python,SQL', 'Python,ML', 'SQL,Cloud', 'Python,Deep Learning', 'SQL,ML', 'Cloud,ML', 'Python,Cloud', 'ML,Deep Learning', 'Python,SQL', 'Python,Cloud'],
    'Salary': [35000, 42000, 39000, 53000, 41000, 60000, 58000, 65000, 47000, 62000]
}

df = pd.DataFrame(data)

# Skills processing
skill_set = ['Python', 'SQL', 'ML', 'Deep Learning', 'Cloud']
for skill in skill_set:
    df[skill] = df['Skills'].apply(lambda x: int(skill in x))
df.drop(columns=['Skills'], inplace=True)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['EducationLevel', 'Location', 'JobRole'], drop_first=False)

# Define features and target
X = df_encoded.drop('Salary', axis=1)
y = df_encoded['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Store expected columns for prediction input
expected_columns = X.columns.tolist()

# Prediction function with error debugging
def predict_salary(experience, education, location, jobrole, skills):
    try:
        skill_values = [int(skill in skills) for skill in skill_set]

        # Construct input data
        row = {
            'YearsExperience': experience,
            'Python': skill_values[0],
            'SQL': skill_values[1],
            'ML': skill_values[2],
            'Deep Learning': skill_values[3],
            'Cloud': skill_values[4],
        }

        # One-hot encode inputs manually
        for level in ['Bachelors', 'Masters', 'PhD']:
            row[f'EducationLevel_{level}'] = 1 if education == level else 0

        for city in ['Delhi', 'Bangalore', 'Hyderabad']:
            row[f'Location_{city}'] = 1 if location == city else 0

        for role in ['Data Scientist', 'ML Engineer', 'Software Developer']:
            row[f'JobRole_{role}'] = 1 if jobrole == role else 0

        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in row:
                row[col] = 0

        input_df = pd.DataFrame([row])[expected_columns]

        # Debug prints
        print("✅ Input shape:", input_df.shape)
        print("✅ Input columns:", input_df.columns.tolist())

        prediction = model.predict(input_df)[0]
        return f"💼 Predicted Salary: ₹{round(prediction, 2)}"

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return "❌ Error occurred. Check Colab output for details."

# Gradio UI
interface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Years of Experience"),
        gr.Radio(["Bachelors", "Masters", "PhD"], label="Education Level"),
        gr.Dropdown(["Delhi", "Bangalore", "Hyderabad"], label="Location"),
        gr.Dropdown(["Data Scientist", "ML Engineer", "Software Developer"], label="Job Role"),
        gr.CheckboxGroup(skill_set, label="Skills")
    ],
    outputs="text",
    title="💰 Employee Salary Predictor",
    description="Enter your profile to estimate salary"
)

interface.launch(share=True)
