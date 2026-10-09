# Consumer Complaint AI --- NLP Classification

**Dataset link:- https://www.consumerfinance.gov/data-research/consumer-complaints/
make sure that dataset is uploaded in streamlit

A Streamlit web application that uses deep-learning models to classify
consumer complaint narratives. It predicts three labels for a complaint:

-   **Product** --- the product/category associated with the complaint
-   **Issue** --- the main issue reported
-   **Sub-issue** --- a more specific description of the issue

The application also displays confidence scores and attempts to extract
dates, account-number-like digit sequences, and currency amounts from
the complaint text.

## Features

-   Enter a complaint manually or select one of the example complaints.
-   Clean and preprocess complaint text before prediction.
-   Predict Product, Issue, and Sub-issue using three trained Keras
    models.
-   Display a confidence percentage for each prediction.
-   Show a visual classification flow.
-   Extract dates, long digit sequences, and currency amounts from the
    complaint.

## How it works

1.  **Text preprocessing:** converts text to lowercase, replaces
    currency values with a `money` token, removes selected
    punctuation/characters, and removes a set of stop words.
2.  **Tokenization:** converts the cleaned text into integer sequences
    using the saved Keras tokenizer.
3.  **Padding:** pads or truncates each sequence to 200 tokens.
4.  **Classification:** three separately trained neural-network models
    predict the product, issue, and sub-issue.
5.  **Label decoding:** saved label encoders convert predicted class
    indexes back into readable labels.
6.  **Results:** Streamlit displays predicted labels and model
    confidence scores.

The notebook trains models using consumer complaint narratives, a
FastText word-vector model to build an embedding matrix, a Keras
tokenizer, and Bidirectional LSTM neural networks. The notebook also
evaluates the models using classification reports and explores
hyperparameter tuning with Optuna.

## Project files

Keep the following files together in the project directory unless you
update the paths in `appn.py`:

``` text
project-folder/
├── appn.py
├── nlp-customer-complaint-project.ipynb
├── C__Users_PRAVALLIKA_Downloads_nlp_nlp.pkl
├── label_encoders (3).pkl
├── best_model (1)n.keras
├── best_issue_model (1)n.keras
└── best_subissue_model (1)n.keras
```

**Important:** The Streamlit app expects the exact model and pickle
filenames shown above. The notebook, however, saves files under names
such as `nlp.pkl`, `label_encoders.pkl`, `best_model.keras`,
`best_issue_model.keras`, and `best_subissue_model.keras`. Rename/copy
the generated artifacts to the filenames expected by `appn.py`, or
update the file paths in `load_resources()` so they match your actual
files. The trained model files and pickle files are required to run the
app; they are not created automatically when Streamlit starts.

## Requirements

Use a Python environment compatible with your installed TensorFlow
version. Install the required packages:

``` bash
pip install streamlit numpy tensorflow scikit-learn nltk gensim regex pandas jupyter
```

The Streamlit app directly imports `streamlit`, `numpy`, and TensorFlow.
The notebook also uses packages including `pandas`, `scikit-learn`,
`nltk`, `gensim`, and `regex`. Depending on your TensorFlow/Keras
version, model loading may require compatible package versions.

## Run the application

1.  Place `appn.py` and the required tokenizer, label-encoder, and model
    files in the project folder.

2.  Open a terminal in that folder and activate your Python environment.

3.  Start Streamlit:

    ``` bash
    streamlit run appn.py
    ```

4.  Open the local URL printed in the terminal (usually
    `http://localhost:8501`).

5.  Enter a complaint or choose an example, then click **Analyze
    Complaint**.

## Training notebook

Open `nlp-customer-complaint-project.ipynb` in Jupyter Notebook,
JupyterLab, or VS Code to review the training workflow. The notebook
expects the complaint CSV dataset at the path specified in its
data-loading cell. Update that path to your local dataset location
before running it outside the original Kaggle environment.

The notebook's main stages are:

1.  Load and inspect the complaint dataset.
2.  Remove rows with missing values.
3.  Preprocess complaint narratives.
4.  Encode Product, Issue, and Sub-issue labels.
5.  Split the data into training and test sets.
6.  Train FastText vectors and prepare a tokenizer/embedding matrix.
7.  Train separate Bidirectional LSTM classifiers for the three targets.
8.  Save models and label encoders, and evaluate predictions.

## Notes and limitations

-   The displayed confidence is the selected model output probability.
    It should not be interpreted as a guarantee that a prediction is
    correct.
-   The information extractor uses regular expressions. Its "Account
    Number" field is based on detecting 8--18 consecutive digits and may
    identify unrelated numbers; review results carefully and avoid
    entering sensitive personal or financial information into a demo.
-   The notebook contains hard-coded Kaggle and Windows paths. Update
    them for your environment.
-   The notebook's Optuna/hyperparameter-tuning section appears to be
    exploratory and may need adjustment before it can be run end-to-end.
-   This project is a demonstration and should not replace official
    complaint review or financial/legal advice.

## Troubleshooting

**`FileNotFoundError` when starting the app** - Confirm all three
`.keras` files and both `.pkl` files are present. - Check that their
names exactly match the paths in `load_resources()` in `appn.py`.

**TensorFlow/Keras model-loading error** - Install a compatible
TensorFlow/Keras version for the environment used to train the models. -
If custom layers or custom objects were used during training, provide
them when loading the model.

**`ModuleNotFoundError`** - Activate the intended virtual environment
and install the missing package using `pip install package-name`.

**The prediction looks incorrect** - Check that the app uses the same
preprocessing logic and tokenizer that were used during training. -
Check that the label encoders correspond to the models and that the
output class counts match the encoded labels.

## Future improvements

-   Add a clear dataset and model-performance summary.
-   Add automated tests for preprocessing, prediction, and information
    extraction.
-   Improve sensitive-number detection and avoid displaying account-like
    numbers unnecessarily.
-   Add confidence thresholds and a message when a prediction is
    uncertain.
-   Package the model artifacts and environment versions for
    reproducible deployment.
