import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\DELL\Desktop\HeartFailurePrediction\heart.csv')

st.title('Heart Failure Prediction')

st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualization')
numerical_columns = df.select_dtypes(include=['number']).columns
st.bar_chart(df[numerical_columns])

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Feature engineering: Create age bins
df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 40, 60, 120], labels=['Young', 'Middle-aged', 'Older'])

# Convert age bins to dummy variables
df = pd.get_dummies(df, columns=['Age_Bin'], drop_first=True)

# Define features and target variable
x = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function for user inputs
def user_report():
    Age = st.sidebar.slider('Age', 1, 77, 1)
    Sex = st.sidebar.selectbox('Sex', ['M', 'F'])
    ChestPainType = st.sidebar.selectbox('Chest Pain Type', ['Atypical Angina(ATA)', 'Non-Anginal Pain(NAP)', 'Asymptonic(ASY)', 'Typical Angina(TA)'])
    RestingBP = st.sidebar.slider('Resting Blood Pressure', 0, 200, 0)
    Cholesterol = st.sidebar.slider('Cholesterol', 0, 603, 0)
    FastingBS = st.sidebar.selectbox('Fasting Blood Sugar', [0, 1])
    RestingECG = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    MaxHR = st.sidebar.slider('MaxHR', 60, 202, 0)
    ExerciseAngina = st.sidebar.selectbox('Exercise Angina', ['Y', 'N'])
    Oldpeak = st.sidebar.slider('Oldpeak', -2.6, 6.2, 0.1)
    ST_Slope = st.sidebar.selectbox('ST_Slope', ['Upsloping', 'Flat', 'Downsloping'])

    # Age binning for user input
    Age_Bin_Young = 1 if Age <= 40 else 0
    Age_Bin_MiddleAged = 1 if 40 < Age <= 60 else 0
    Age_Bin_Older = 1 if Age > 60 else 0

    user_report = {
        'Age': Age,
        'Sex_M': 1 if Sex == 'M' else 0,
        'ChestPainType_ASY': 1 if ChestPainType == 'Asymptonic(ASY)' else 0,
        'ChestPainType_ATA': 1 if ChestPainType == 'Atypical Angina(ATA)' else 0,
        'ChestPainType_NAP': 1 if ChestPainType == 'Non-Anginal Pain(NAP)' else 0,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG_LVH': 1 if RestingECG == 'LVH' else 0,
        'RestingECG_Normal': 1 if RestingECG == 'Normal' else 0,
        'MaxHR': MaxHR,
        'ExerciseAngina_Y': 1 if ExerciseAngina == 'Y' else 0,
        'Oldpeak': Oldpeak,
        'ST_Slope_Downsloping': 1 if ST_Slope == 'Downsloping' else 0,
        'ST_Slope_Flat': 1 if ST_Slope == 'Flat' else 0,
        'Age_Bin_Young': Age_Bin_Young,
        'Age_Bin_Middle-aged': Age_Bin_MiddleAged,
        'Age_Bin_Older': Age_Bin_Older
    }
    
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

# Media display
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg", "mp4", "wav"])
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
    elif uploaded_file.type.startswith('audio'):
        st.audio(uploaded_file)

# Show progress while training the model
with st.spinner('Training model...'):
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(x_train, y_train)
st.success('Model trained successfully!')

# Display feature importance
importances = rf.feature_importances_
feature_names = x.columns
st.subheader('Feature Importance')
fig, ax = plt.subplots()
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance', ax=ax)
st.pyplot(fig)

# Make predictions for the user's input
user_data = user_report()
user_data = user_data.reindex(columns=x.columns, fill_value=0)
user_result = rf.predict(user_data)

st.subheader('Your Report: ')
if user_result[0] == 0:
    st.write('You have Heart Disease')
else:
    st.write('You do not have Heart Disease')
