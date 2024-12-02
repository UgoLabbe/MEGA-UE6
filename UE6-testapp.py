#Import dependencies
import pandas as pd
import streamlit as sl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import product, cycle

#Import file
df = pd.read_csv("Food_and_Nutrition__.csv")
print(df.head(5))

#Pre Processing

#Scale numerical and Change activity level column 
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

#Remove "useless" columns
df = df.drop(columns=["Breakfast Suggestion",
                      "Lunch Suggestion","Dinner Suggestion","Snack Suggestion"])

# Mapping the activity levels to numeric values
activity_mapping = {
    'Sedentary': 1,
    'Lightly Active': 2,
    'Moderately Active': 3,
    'Very Active': 4,
    'Extremely Active': 5
}

# Replace the values in the column
df['Activity Level'] = df['Activity Level'].replace(activity_mapping)

# Separate the disease (don't keep acne, diabetes and weight loss) to create a new one
# Splitting diseases into a list and getting unique diseases
df["Disease List"] = df["Disease"].str.split(", ")
unique_diseases = set(disease for diseases in df["Disease List"] for disease in diseases)

# Creating columns for each disease
for disease in unique_diseases:
    df[disease] = df["Disease List"].apply(lambda x: 1 if disease in x else 0)

# Creating a new column with diseases excluding Acne, Weight Loss, and Diabetes
df["Disease"] = df["Disease List"].apply(lambda x: ", ".join([disease for disease in x if disease not in ["Acne", "Weight Loss", "Diabetes", "Weight Gain"]]))

df_encoded = df.copy()
#Remove the rows that doesn't interest us
df_encoded = df_encoded[df_encoded["Disease"].str.strip().ne('')]

print(df_encoded["Disease"])

df_diseases = df_encoded["Disease"]
label_encoder = LabelEncoder()
df_diseases_encoded = label_encoder.fit_transform(df_diseases)

#Only keep what we want
df_features = df_encoded[["Activity Level","Daily Calorie Target","Protein","Sugar","Sodium","Calories","Carbohydrates","Fiber","Fat"]]

# LDA for dimensionality reduction (Supervised)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r = lda.fit_transform(df_features, df_diseases_encoded)  # Apply LDA with target disease labels

plt.figure(figsize=(10, 6))
# Get unique diseases
unique_diseases = df_diseases.unique()

# Create a color map
colors = ["red", "blue", "m", "green"]

# Plot each disease with a different color
for disease, color in zip(unique_diseases, colors):
    mask = df_diseases == disease
    plt.scatter(X_r[mask, 0], X_r[mask, 1], c=[color], label=disease, alpha=0.9, s=35)

plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA Visualization of Diseases')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()