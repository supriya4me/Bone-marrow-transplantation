import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load your dataset
df = pd.read_csv('.\csv_result-bone+marrow+transplant+children.csv')  # Adjust file path if needed

# Define important features and target
important_features = ['survival_time', 'extcGvHD', 'Relapse', 'CD34kgx10d6',
    'PLTrecovery', 'ANCrecovery', 'Recipientage', 'Rbodymass']

# Load and preprocess data
if df.empty:
    st.stop()

# Separate features and target
X = df[important_features]
y = df['survival_status']

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create a column transformer with one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))
        ]), X.select_dtypes(include=['number']).columns)
    ],
    remainder='passthrough'
)

# Apply transformations
X_transformed = preprocessor.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
conf_matrix = confusion_matrix(y_test, y_pred)

st.title('Bone Marrow Transplantation Model Analysis')

st.subheader('Model Evaluation Metrics')
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

st.subheader('Confusion Matrix')
st.write(conf_matrix)

# User Input for Predictions
st.sidebar.header('User Input Parameters')

def user_input_features():
    survival_time = st.sidebar.number_input('Survival Time (days)', min_value=0, max_value=10000, value=1)
    extcGvHD = st.sidebar.selectbox('extcGvHD (0 = No, 1 = Yes)', options=[0, 1])
    Relapse = st.sidebar.selectbox('Relapse (0 = No, 1 = Yes)', options=[0, 1])
    CD34kgx10d6 = st.sidebar.number_input('CD34kgx10d6', min_value=0.0, max_value=1000.0, value=1.0)
    PLTrecovery = st.sidebar.selectbox('PLTrecovery (0 = No, 1 = Yes)', options=[0, 1])
    ANCrecovery = st.sidebar.selectbox('ANCrecovery (0 = No, 1 = Yes)', options=[0, 1])
    Recipientage = st.sidebar.number_input('Recipient Age (years)', min_value=0, max_value=120, value=1)
    Rbodymass = st.sidebar.number_input('Recipient Body Mass (kg)', min_value=30.0, max_value=200.0, value=60.0)

    data = {
        'survival_time': survival_time,
        'extcGvHD': extcGvHD,
        'Relapse': Relapse,
        'CD34kgx10d6': CD34kgx10d6,
        'PLTrecovery': PLTrecovery,
        'ANCrecovery': ANCrecovery,
        'Recipientage': Recipientage,
        'Rbodymass': Rbodymass
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Ensure user_input is defined before using it
user_input = user_input_features()

# Preprocess user input
user_input_transformed = preprocessor.transform(user_input)

# Display transformed user input
st.subheader('Debugging Information')
st.write("Transformed User Input:")
st.write(user_input_transformed)

# Predict using the model and display the prediction
user_prediction_prob = model.predict(user_input_transformed)
user_prediction = (user_prediction_prob > 0.5).astype(int).flatten()

st.subheader('Prediction for User Input')
st.write(f'The predicted survival status is: {"Survived" if user_prediction[0] == 1 else "Not Survived"}')
st.write(f'Probability of Survival: {user_prediction_prob[0][0]:.2f}')

# Check the class distribution
st.subheader('Target Variable Distribution')
st.write(df['survival_status'].value_counts())

# Print model summary
st.subheader('Model Summary')
st.write(model.summary())

# Check data consistency
st.subheader('User Input Data')
st.write(user_input)

# Verify preprocessing consistency
st.subheader('Preprocessing Steps Consistency')
st.write(f"Number of features after transformation: {X_transformed.shape[1]}")

# Additional debugging output
st.subheader('Model Debugging Output')

# Show distribution of predictions
st.write("Predicted Class Distribution on Test Set:")
st.write(pd.Series(y_pred).value_counts())

# Show the confusion matrix in more detail
st.write("Confusion Matrix on Test Set:")
st.write(conf_matrix)
