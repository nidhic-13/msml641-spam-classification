import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_text
import streamlit as st
import numpy as np


# Load model
@st.cache_resource
def load_model():
    """
    Load the TensorFlow SavedModel
    """
    model_path = "./spam_classifier_final"
    try:
        loaded = tf.saved_model.load(model_path)

        # Grab the default serving signature
        infer = loaded.signatures.get("serving_default", None)
        if infer is None:
            st.error(
                "SavedModel loaded, but no serving_default signature found."
            )
            return None

        return infer
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None


# Preprocessing and prediction
def preprocess_text(text: str) -> str:
    """
    Preprocess email text.
    """
    return text.lower().strip()


def predict_spam(model_fn, email_text: str):
    """
    Make prediction using the loaded SavedModel serving function.

    model_fn: a ConcreteFunction from loaded.signatures['serving_default']
    Returns:
        spam_prob (float) or None if there was an error.
    """
    try:
        processed = preprocess_text(email_text)

        # Batch of 1 string
        input_tensor = tf.constant([processed])

        # try simple positional call
        try:
            outputs = model_fn(input_tensor)
        except TypeError:
            # If above fails, inspect the signature and call with a keyword arg
            _, kw = model_fn.structured_input_signature
            if len(kw) == 1:
                input_name = list(kw.keys())[0]
                outputs = model_fn(**{input_name: input_tensor})
            else:
                raise

        # grab the first tensor
        if isinstance(outputs, dict):
            first_tensor = next(iter(outputs.values()))
        else:
            first_tensor = outputs

        pred = first_tensor.numpy()

        # Convert to a scalar 
        if pred.ndim == 2 and pred.shape[0] == 1:
            if pred.shape[1] == 1:
                # Single sigmoid output
                spam_prob = float(pred[0, 0])
        else:
            spam_prob = float(np.max(pred))

        return spam_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# Streamlit UI
st.set_page_config(
    page_title="Spam Email Classifier",
    layout="wide"
)

st.title("Email Spam Classifier")
st.markdown("Classify emails as spam or legitimate")

# Load model
model_fn = load_model()

if model_fn is not None:
    st.success("Model loaded successfully!")

    # Input section
    st.subheader("Enter Email Content")
    email_text = st.text_area(
        "Paste email content here:",
        height=200,
        placeholder="Enter the email text you want to classify..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_btn = st.button("Classify", type="primary")

    if classify_btn and email_text:
        with st.spinner("Analyzing email..."):
            spam_probability = predict_spam(model_fn, email_text)

            if spam_probability is not None:
                st.divider()
                st.subheader("Results")

                col1, col2 = st.columns(2)

                label = "SPAM" if spam_probability > 0.5 else "LEGITIMATE"
                conf_pct = spam_probability * 100.0

                with col1:
                    st.metric("Classification", label)

                with col2:
                    st.metric("Confidence (Spam Probability)", f"{conf_pct:.1f}%")

                # Progress bar
                st.progress(min(max(spam_probability, 0.0), 1.0))

    elif classify_btn:
        st.warning("Please enter email text to classify")

else:
    st.error(
        "Failed to load model.\n\n"
        "Make sure `./spam_classifier_final` is a TensorFlow SavedModel "
        "directory and was exported with a `serving_default` signature."
    )