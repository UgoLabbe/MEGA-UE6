import streamlit as sl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    sl.title('Disease Classification Dashboard')

    # Import file
    df = pd.read_csv("Food_and_Nutrition__.csv")

    # Pre-Processing
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Remove "useless" columns
    df = df.drop(columns=["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"])

    # Mapping activity levels
    activity_mapping = {
        'Sedentary': 1, 'Lightly Active': 2,
        'Moderately Active': 3, 'Very Active': 4,
        'Extremely Active': 5
    }
    df['Activity Level'] = df['Activity Level'].replace(activity_mapping)

    # Splitting diseases
    df["Disease List"] = df["Disease"].str.split(", ")
    
    # Creating a new column with diseases excluding some categories
    df["Disease"] = df["Disease List"].apply(lambda x: ", ".join([disease for disease in x if disease not in ["Acne", "Weight Loss", "Diabetes", "Weight Gain"]]))

    df_encoded = df.copy()
    df_encoded = df_encoded[df_encoded["Disease"].str.strip().ne('')]
    
    df_diseases = df_encoded["Disease"]
    label_encoder = LabelEncoder()
    df_diseases_encoded = label_encoder.fit_transform(df_diseases)

    # Features for LDA
    df_features = df_encoded[["Activity Level","Daily Calorie Target","Protein","Sugar","Sodium","Calories","Carbohydrates","Fiber","Fat"]]

    # LDA Dimensionality Reduction
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit_transform(df_features, df_diseases_encoded)

    # Visualization options
    vis_option = sl.sidebar.selectbox('Visualization Type', 
                                      ['Scatter Plot'])

    # Color palette
    colors = ["red", "blue", "m", "green"]
    unique_diseases = df_diseases.unique()

    # Plot
    plt.figure(figsize=(10, 6))
    
    if vis_option == 'Scatter Plot':
        for disease, color in zip(unique_diseases, colors):
            mask = df_diseases == disease
            plt.scatter(X_r[mask, 0], X_r[mask, 1], c=[color], label=disease, alpha=0.9, s=35)
        
        plt.xlabel('LDA Component 1')
        plt.ylabel('LDA Component 2')
        plt.title('LDA Visualization of Diseases')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    

    plt.tight_layout()
    sl.pyplot(plt)

    # Additional information
    sl.sidebar.header('Dataset Information')
    sl.sidebar.write(f'Total Samples: {len(df_encoded)}')
    sl.sidebar.write(f'Number of Diseases: {len(unique_diseases)}')
    sl.sidebar.write('Diseases:', ', '.join(unique_diseases))

if __name__ == '__main__':
    main()