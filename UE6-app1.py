import streamlit as sl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    # Load dataset
    df = pd.read_csv("Food_and_Nutrition__.csv")
    
    # Sidebar Navigation
    sl.sidebar.title("Navigation")
    page = sl.sidebar.radio("Go to", ["Disease Classification", "Data Description"])
    
    if page == "Disease Classification":
        disease_classification_page(df)
    elif page == "Data Description":
        data_description_page(df)


def disease_classification_page(df):
    sl.title('Disease Classification Dashboard')
    
    # Preprocessing the data
    scaler = StandardScaler()
    # numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    columns_used_lda = ["Activity Level", "Protein", "Sugar", "Sodium", "Carbohydrates", "Fiber", "Fat"]
    numerical_columns_used_lda = df[columns_used_lda].select_dtypes(include=["float64", "int64"]).columns
    df[numerical_columns_used_lda] = scaler.fit_transform(df[numerical_columns_used_lda])

    
    # Remove "useless" columns
    df = df.drop(columns=["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"])
    
    # Map activity levels
    activity_mapping = {
        'Sedentary': 1, 'Lightly Active': 2,
        'Moderately Active': 3, 'Very Active': 4,
        'Extremely Active': 5
    }
    df['Activity Level'] = df['Activity Level'].replace(activity_mapping)
    
    # Splitting diseases
    df["Disease List"] = df["Disease"].str.split(", ")
    df["Disease"] = df["Disease List"].apply(lambda x: ", ".join([disease for disease in x if disease not in ["Acne", "Weight Loss"]]))
    df_encoded = df[df["Disease"].str.strip().ne('')]
    
    df_diseases = df_encoded["Disease"]
    label_encoder = LabelEncoder()
    df_diseases_encoded = label_encoder.fit_transform(df_diseases)
    
    # Features for LDA
    df_features = df_encoded[columns_used_lda]
    
    # LDA Dimensionality Reduction
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit_transform(df_features, df_diseases_encoded)
    
    # Interactive user inputs
    sl.sidebar.header("Enter your patients details:")
    activity_level = sl.sidebar.selectbox("Activity Level", ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active'])
    protein = sl.sidebar.number_input("Protein (g)", min_value=0, value=140)
    sugar = sl.sidebar.number_input("Sugar (g)", min_value=0, value=126)
    sodium = sl.sidebar.number_input("Sodium (g)", min_value=0, value=28)
    carbohydrates = sl.sidebar.number_input("Carbohydrates (g)", min_value=0, value=262)
    fiber = sl.sidebar.number_input("Fiber (g)", min_value=0, value=30)
    fat = sl.sidebar.number_input("Fat (g)", min_value=0, value=70)
    
    activity_mapping = {
        'Sedentary': 1, 'Lightly Active': 2,
        'Moderately Active': 3, 'Very Active': 4,
        'Extremely Active': 5
    }
    activity_level = activity_mapping[activity_level]
    
    # Create a DataFrame for the user's input
    user_input = pd.DataFrame({
        "Activity Level": [activity_level],
        "Protein": [protein],
        "Sugar": [sugar],
        "Sodium": [sodium],
        "Carbohydrates": [carbohydrates],
        "Fiber": [fiber],
        "Fat": [fat]
    })
    print('\n\n\n')
    print(X_r[0])
    # Scale the user input using the same scaler
    user_input_scaled = scaler.transform(user_input.drop(columns=['Activity Level']))

    # Add activity level to the user_input scaled
    user_input_scaled = np.hstack([user_input[['Activity Level']].values, user_input_scaled])
    print(user_input_scaled)
    print('\n\n\n')

    # LDA Prediction for the user input
    user_lda_position = lda.transform(user_input_scaled)

    # Visualization options
    colors = ["red", "blue", "m", "green"]
    unique_diseases = df_diseases.unique()
    
    # Plotting
    plt.figure(figsize=(12, 8))

    for disease, color in zip(unique_diseases, colors):
        mask = df_diseases == disease
        plt.scatter(X_r[mask, 0], X_r[mask, 1], c=[color], label=disease, alpha=0.9, s=35)
    
    # Plot the user's position
    plt.scatter(user_lda_position[0, 0], user_lda_position[0, 1], c='black', marker='x', s=100, label="Patient")

    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('LDA Visualization of Diseases')
    plt.legend(loc='best')
    
    plt.tight_layout()
    sl.pyplot(plt)

def data_description_page(df):
    sl.title("Data Description")
    
    # Display the dataset
    sl.subheader("Dataset Overview")
    sl.dataframe(df.head())
    
    # Dataset Statistics
    sl.subheader("Summary Statistics")
    sl.write(df.describe())
    
    # Key Performance Indicators (KPIs)
    sl.subheader("Key Performance Indicators (KPIs)")
    total_records = len(df)
    gender_distribution = df['Gender'].value_counts()
    most_common_activity = df['Activity Level'].mode()[0]
    avg_calories = df['Calories'].mean()
    
    sl.metric("Total Records", total_records)
    sl.metric("Most Common Activity Level", most_common_activity)
    sl.metric("Average Calorie Consumption", round(avg_calories, 2))
    
    sl.subheader("Visualizations")
    
    # Gender distribution pie chart
    sl.write("### Gender Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'pink'])
    ax1.axis('equal')
    sl.pyplot(fig1)
    
    # Activity level bar chart
    sl.write("### Activity Level Distribution")
    activity_counts = df['Activity Level'].value_counts().reindex(['Sedentary','Lightly Active','Moderately Active','Very Active', 'Extremely Active'])
    fig2, ax2 = plt.subplots()
    sns.barplot(x=activity_counts.index, y=activity_counts.values, palette='viridis', ax=ax2)
    ax2.set_xlabel("Activity Level")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=10)
    ax2.set_ylabel("Count")
    ax2.set_title("Activity Level Distribution")
    sl.pyplot(fig2)
    
    # Line plots for numerical feature distributions by gender
    sl.write("### Line Plots for Numerical Distributions by Gender")
    numerical_features = ['Weight', 'Height', 'Calories']
    for feature in numerical_features:
        fig, ax = plt.subplots()
        for gender, color in zip(['Male', 'Female'], ['dodgerblue', 'hotpink']):
            sns.kdeplot(
                data=df[df['Gender'] == gender], 
                x=feature, 
                ax=ax, 
                label=gender, 
                color=color, 
                fill=True, 
                alpha=0.4
            )
        ax.set_title(f"{feature} Distribution by Gender")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(title="Gender")
        sl.pyplot(fig)
    
    # Disease distribution
    sl.write("### Disease Distribution")
    disease_counts = df['Disease'].str.split(", ").explode().value_counts()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=disease_counts.values, y=disease_counts.index, palette="cool", ax=ax3)
    ax3.set_xlabel("Count")
    ax3.set_ylabel("Disease")
    ax3.set_title("Distribution of Diseases")
    sl.pyplot(fig3)
    
    # Correlation heatmap
    sl.write("### Correlation Heatmap")
    corr_matrix = df[["Protein","Sugar","Sodium","Carbohydrates","Fiber","Fat"]].corr()
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Between Numerical Features")
    sl.pyplot(fig4)

if __name__ == '__main__':
    main()
