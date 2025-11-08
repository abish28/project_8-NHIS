import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import joblib

# Set page configuration with a custom icon and centered layout
st.set_page_config(
    page_title="Job Salary Predictor",
    page_icon="📈",
    layout="centered"
)

# ---- Custom CSS for a more attractive interface ----
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Main background and font - CHANGED FOR DARK MODE */
        .stApp {
            background-color: #1e1e1e; /* A deep charcoal background */
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #ecf0f1; /* Light text for readability */
        }
        
        /* Input text box styling */
        .stTextInput > div > div > input {
            background-color: #333; /* Darker input box */
            border: 2px solid #e74c3c;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            color: #ecf0f1; /* Light text within input box */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        }

        /* Button styling */
        .stButton button {
            background-color: #e74c3c;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 1.2em;
            border: none;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #c0392b;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.7);
            transform: translateY(-3px);
        }

        /* Prediction result styling */
        .prediction-result {
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: white;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
            margin-top: 20px;
        }
        
        /* Placeholder text styling - CHANGED */
        .placeholder-text {
            background-color: #333; /* Darker background */
            border-left: 5px solid #e74c3c;
            padding: 15px;
            border-radius: 8px;
            color: #ecf0f1; /* Light text for visibility */
            margin-top: 20px;
            font-size: 0.9em;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        /* The info tip at the bottom - CHANGED */
        .info-tip {
            background-color: #333; /* Darker background */
            border-left: 5px solid #3498db;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            color: #ecf0f1; /* Light text for visibility */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        /* General header and paragraph styling - CHANGED */
        h1.st-emotion-cache-17e997a, h2, .stMarkdown p {
            color: #ecf0f1; /* Set all general text to a light color */
        }
        h1.st-emotion-cache-17e997a {
            font-size: 4em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0;
        }
        h2 {
            text-align: center;
            font-weight: 300;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- Helper Functions ----
def identity_tokenizer(tokens):
    return tokens

@st.cache_resource
def load_resources():
    try:
        # Load the TF-IDF vectorizer
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        # Load the trained Keras model
        model = tf.keras.models.load_model("job_salary_model.keras")
        return vectorizer, model
    except (FileNotFoundError, AttributeError) as e:
        st.error(f"Error loading model files: {e}. Please ensure 'tfidf_vectorizer.pkl' and 'job_salary_model.keras' are in the same directory.")
        return None, None

# ---- Load Resources ----
vectorizer, model = load_resources()

if model is None:
    st.stop()

# ---- Streamlit App Interface ----
add_custom_css()

st.title("💰 Job Salary Predictor")
st.markdown("<h2 style='text-align:center;'>Your Career Compass</h2>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size: 1.2em;'>
    Enter a job title below and get an instant salary estimate based on our machine learning model.
    </p>
    """,
    unsafe_allow_html=True
)

# Create columns for a cleaner layout
col1, col2 = st.columns([3, 1])

with col1:
    user_input_title = st.text_input(
        "Enter Job Title:",
        placeholder="e.g., Senior Python Developer for a remote team",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Salary", type="primary", use_container_width=True):
        if not user_input_title:
            st.warning("Please enter a job title to get a prediction.")
        else:
            try:
                X_text_input = vectorizer.transform([user_input_title])
                placeholder_numeric = np.array([[0.0, 0.0, 0.0]])
                prediction = model.predict([X_text_input.toarray(), placeholder_numeric])
                predicted_salary = prediction[0][0]

                st.markdown("---")
                st.markdown(
                    f'<div class="prediction-result">Predicted Salary: ${predicted_salary:,.2f}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    """
                    <div class="placeholder-text">
                        This is a placeholder salary based on the assumption of average numeric features.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown(
    """
    <div class="info-tip">
    💡 <b>Tip:</b> Be as descriptive as possible in the job title for a more accurate prediction!
    </div>
    """,
    unsafe_allow_html=True
)