
import streamlit as st
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Consumer Complaint AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f7f7f7;
    text-align: center;
    border: 1px solid #ddd;
}

.result {
    font-size: 22px;
    font-weight: bold;
    margin-top: 10px;
}

.confidence {
    font-size: 16px;
    color: #555;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS AND PREPROCESSING
# =========================================================

@st.cache_resource
def load_resources():

    # Tokenizer
    with open(r"C__Users_PRAVALLIKA_Downloads_nlp_nlp.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Label encoders
    with open(r"label_encoders (3).pkl", "rb") as f:
        encoders = pickle.load(f)

    # Models
    product_model = load_model("best_model (1)n.keras")
    issue_model = load_model("best_issue_model (1)n.keras")
    subissue_model = load_model("best_subissue_model (1)n.keras")

    return (
        tokenizer,
        encoders,
        product_model,
        issue_model,
        subissue_model
    )


tokenizer, encoders, product_model, issue_model, subissue_model = load_resources()

le_problem = encoders["problem"]
le_issue = encoders["issue"]
le_subissue = encoders["subissue"]


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def text_preprocessing(text):

    text = text.lower()

    # Remove standalone x
    text = re.sub(r'\bx+\b', '', text)

    # Replace money values
    text = re.sub(
        r'[$₹€£]\s?[\d,]+(?:\.\d+)?',
        ' money ',
        text
    )

    # Keep only letters, numbers and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Stop words
    stop_words = {
        "i", "me", "my", "we", "our", "you", "your",
        "he", "she", "it", "they", "them",
        "is", "am", "are", "was", "were",
        "the", "a", "an",
        "and", "or", "but",
        "to", "of", "in", "on", "for",
        "with", "this", "that",
        "have", "has", "had",
        "be", "been",
        "do", "does", "did"
    }

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_complaint(complaint):

    cleaned_text = text_preprocessing(complaint)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="pre",
        truncating="post"
    )

    # Predictions
    product_probability = product_model.predict(
        padded,
        verbose=0
    )[0]

    issue_probability = issue_model.predict(
        padded,
        verbose=0
    )[0]

    subissue_probability = subissue_model.predict(
        padded,
        verbose=0
    )[0]

    # Predicted indexes
    product_index = np.argmax(product_probability)
    issue_index = np.argmax(issue_probability)
    subissue_index = np.argmax(subissue_probability)

    # Convert indexes to original labels
    product = le_problem.inverse_transform(
        [product_index]
    )[0]

    issue = le_issue.inverse_transform(
        [issue_index]
    )[0]

    subissue = le_subissue.inverse_transform(
        [subissue_index]
    )[0]

    # Confidence
    product_confidence = product_probability[product_index] * 100
    issue_confidence = issue_probability[issue_index] * 100
    subissue_confidence = subissue_probability[subissue_index] * 100

    return (
        product,
        issue,
        subissue,
        product_confidence,
        issue_confidence,
        subissue_confidence
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🤖 Consumer Complaint AI</div>',
    unsafe_allow_html=True
)


# =========================================================
# EXAMPLE COMPLAINTS
# =========================================================

st.subheader("💬 Enter Consumer Complaint")

examples = [
    "I have been waiting for my refund but I have not received the money.",
    "My credit card payment was deducted but the payment was not completed.",
    "I received a call from a debt collector about a debt that I do not recognize."
]

selected_example = st.selectbox(
    "Or choose an example complaint:",
    ["Write your own complaint"] + examples
)


# =========================================================
# TEXT INPUT
# =========================================================

if selected_example == "Write your own complaint":

    complaint = st.text_area(
        "Complaint",
        placeholder="Example: I have not received my refund even though the transaction was cancelled.",
        height=150
    )

else:

    complaint = st.text_area(
        "Complaint",
        value=selected_example,
        height=150
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Complaint",
    use_container_width=True
):

    if complaint.strip() == "":

        st.warning("⚠️ Please enter a consumer complaint.")

    else:

        with st.spinner("Analyzing complaint..."):

            (
                product,
                issue,
                subissue,
                product_confidence,
                issue_confidence,
                subissue_confidence
            ) = predict_complaint(complaint)

        st.success("✅ Complaint analyzed successfully!")


        # =================================================
        # RESULTS
        # =================================================

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)


        # Product
        with col1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.write("🏷️ **Product**")

            st.markdown(
                f'<div class="result">{product}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence">'
                f'{product_confidence:.2f}% confidence'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(float(min(product_confidence / 100, 1.0)))

            st.markdown("</div>", unsafe_allow_html=True)


        # Issue
        with col2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.write("⚠️ **Issue**")

            st.markdown(
                f'<div class="result">{issue}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence">'
                f'{issue_confidence:.2f}% confidence'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(float(min(issue_confidence / 100, 1.0)))

            st.markdown("</div>", unsafe_allow_html=True)


        # Sub-issue
        with col3:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.write("🔎 **Sub-issue**")

            st.markdown(
                f'<div class="result">{subissue}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence">'
                f'{subissue_confidence:.2f}% confidence'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(float(min(subissue_confidence / 100, 1.0)))

            st.markdown("</div>", unsafe_allow_html=True)


        # =================================================
        # CLASSIFICATION FLOW
        # =================================================

        st.subheader("🔗 Classification Flow")

        st.write(
            f"**Complaint → Product → Issue → Sub-issue**"
        )


        
# extracting important information
def extract_info(text):

    info = {}

    # Date
    date = re.findall(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        text
    )

    # Remove duplicate dates
    date = list(dict.fromkeys(date))

    # Account number
    account = re.findall(
        r'\b\d{8,18}\b',
        text
    )

    # Amount
    amount = re.findall(
        r'(?:₹|Rs\.?|INR|\$)\s?\d+(?:,\d+)*(?:\.\d+)?',
        text,
        flags=re.IGNORECASE
    )

    if date:
        info["Date"] = date

    if account:
        info["Account Number"] = account

    if amount:
        info["Amount"] = amount

    return info
(
    product,
    issue,
    subissue,
    product_confidence,
    issue_confidence,
    subissue_confidence
) = predict_complaint(complaint)


extracted_info = extract_info(complaint)

st.subheader("📌 Important Information")

if extracted_info:

    for key, value in extracted_info.items():
        st.write(f"**{key}:** {', '.join(value)}")

else:

    st.write("No important information found.")

