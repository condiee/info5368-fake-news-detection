import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions import compute_f1, fetch_dataset, compute_precision, compute_recall, compute_accuracy, apply_threshold
from sklearn.metrics import recall_score, precision_score, roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pages.B_Train_Model import split_dataset
import matplotlib.pyplot as plt

random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.title('Test Model')

#############################################

# Used to access model performance in dictionaries
METRICS_MAP = {
    'precision': compute_precision,
    'recall': compute_recall,
    'accuracy': compute_accuracy,
    'f1': compute_f1
}

# Checkpoint 9
def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function computes one or more metrics (precision, recall, accuracy, f1 score) using the model

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evaluate performance (string); 'precision', 'recall', 'accuracy'
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {}
    # Add code here
    # 1. Make a prediction using the model and input data
    pred = model.predict(X)
    # 2. Write a for loop that iterates through metrics, a list containings one or more strings
    # including ‘precision’, ‘recall’, ‘accuracy’
    for metric in metrics:
        # a. Check the metric name and compute it based on the string input. For example, if
        # metric=’precision’ then compute the precision on the predicted and input y_true.
        result = None
        if metric == 'precision':
            result = compute_precision(y_true, pred)
        elif metric == 'recall':
            result = compute_recall(y_true, pred)
        elif metric == 'accuracy':
            # result = np.sum(pred == y_true.to_numpy().reshape(-1)) / len(X)
            result = compute_accuracy(y_true, pred)
        elif metric == 'f1 score':
            # result = np.sum(pred == y_true.to_numpy().reshape(-1)) / len(X)
            result = compute_f1(y_true, pred)
        
        # b. Store the result in out_dict[metric_name]
        metric_dict[metric] = result

    return metric_dict


#confusion matrix
## referenced https://www.kaggle.com/code/sahrul/fake-news-detection-using-cnn
def plot_cmatrix(X, y_true, model):
    pred = model.predict(X)
    cmatrix = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = [False, True])
    cm_display.plot(ax=ax)
    st.pyplot(fig)
    #cm_display.plot()
   # plt.show()
   # cmatrix - pd.DataFrame(cmatrix, index = [0, 1], columns = [0, 1])
    #cm_cv.index.name = "Actual"
    #cm_cv.columns.name = "Predicted"
    #plt.figure(figsize = (10, 10))
    #sns.heatmap(cmatrix, cmap = "Blues", annot = True, fmt = '')

# Checkpoint 10
def plot_pr_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    """
    Plot the Precision/Recall curve between predicted and actual values for model names in trained_models on the training and validation datasets

    Input:
        - X_train: training input data
        - X_val: test input data
        - y_true: true targets
        - y_pred: predicted targets
        - model_names: trained model names
        - trained_models: trained models in a dictionary (accessed with model name)
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[model_name.__name__ + " Train Precision"] = train_precision_all
            - df[model_name.__name__ + " Train Recall"] = train_recall_all
            - df[model_name.__name__ + " Validation Precision"] = val_precision_all
            - df[model_name.__name__ + " Validation Recall"] = val_recall_all
    """
    # Set up figures
    fig = make_subplots(rows=len(trained_models), cols=1, shared_xaxes=True, vertical_spacing=0.15,
                         subplot_titles=model_names)

    # Intialize variables
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100) # 100 threshold values

    # Add code here
    for i, model in enumerate(trained_models):
    # 1. Use the trained model in trained_models[model_name] to:
        # i. Make predictions on the train set using predict_proba() function
        train_probabilities = model.predict_proba(X_train)
        train_precision_all = []
        train_recall_all = []

        # ii. Make predictions on the validation set using predict_proba() function
        val_probabilities = model.predict_proba(X_val)
        val_precision_all = []
        val_recall_all = []

        for threshold in threshold_values:
            # iii. Apply the threshold to the predictions on the training set using the apply_threshold function
            train_predictions = apply_threshold(train_probabilities, threshold)

            # iv. Apply the threshold to the predictions on the validation set using the apply_threshold function
            val_predictions = apply_threshold(val_probabilities, threshold)
            
            # v. Compute precision and recall on the training set using the predictions on the
            # training set (with threshold applied) and the true values (y_train). Use
            # precision_score (set zero_division=1) and recall_score functions.
            train_precision = precision_score(y_train, train_predictions, zero_division=1) # , pos_label=1, average=None
            train_precision_all += [train_precision]
            train_recall = recall_score(y_train, train_predictions)
            train_recall_all += [train_recall]

            # vi. Compute precision and recall on validation set using the predictions on the
            # validation set (with threshold applied) and the true values (y_val). Use
            # precision_score (set zero_division=1) and recall_score functions
            val_precision = precision_score(y_val, val_predictions, zero_division=1) # , pos_label=1, average=None
            val_precision_all += [val_precision]
            val_recall = recall_score(y_val, val_predictions)
            val_recall_all += [val_recall]

        # 2. Plot a Precision/Recall Curves showing the results on training and validation sets using the
        # train_precision_all, train_recall_all, val_precision_all, and val_recall_all. Plot precision on
        # the y-axis and recall on the x-axis (see code snippet below.
        fig.add_trace(go.Scatter(x=train_recall_all, y=train_precision_all, name="Train"), row=i+1, col=1) # use enumerated value i to align figures vertically
        fig.add_trace(go.Scatter(x=val_recall_all, y=val_precision_all, name="Validation"),row=i+1, col=1) # use enumerated value i
        fig.update_xaxes(title_text="Recall")
        fig.update_yaxes(title_text='Precision', row=i+1, col=1) # use enumerated value i
        fig.update_layout(title='Precision/Recall Curve', height=600) # title=model_names[i]+
    
        # 3. Save the results (train_precision_all, train_recall_all, val_precision_all, and val_recall_all) in df
        df[model_names[i] + " Train Precision"] = train_precision_all
        df[model_names[i] + " Train Recall"] = train_recall_all
        df[model_names[i] + " Validation Precision"] = val_precision_all
        df[model_names[i] + " Validation Recall"] = val_recall_all

    return fig, df

def plot_roc_curve(X_train, X_val, y_train, y_val, trained_models, model_names):
    """
    Plot the ROC curve between predicted and actual values for model names in trained_models on the training and validation datasets

    Input:
        - X_train: training input data
        - X_val: test input data
        - y_true: true targets
        - y_pred: predicted targets
        - model_names: trained model names
        - trained_models: trained models in a dictionary (accessed with model name)
    Output:
        - fig: the plotted figure
    """
    # Set up figures
    fig = make_subplots(rows=len(trained_models), cols=1, shared_xaxes=True, vertical_spacing=0.15,
                         subplot_titles=model_names)

    # Intialize variables
    df = pd.DataFrame()
    threshold_values = np.linspace(0.5, 1, num=100) # 100 threshold values

    # Add code here
    for i, model in enumerate(trained_models):
    # 1. Use the trained model in trained_models[model_name] to:
        # i. Make predictions on the train set using predict_proba() function
        train_probabilities = model.predict_proba(X_train)[::,1]
        # st.write('train',train_probabilities)
        # ii. Make predictions on the validation set using predict_proba() function
        val_probabilities = model.predict_proba(X_val)[::,1]
        # st.write('val', max(val_probabilities), min(val_probabilities))

        st.write(len(train_probabilities), len(y_val))

        train_fpr, train_tpr, _ = roc_curve(y_val, train_probabilities, pos_label=1)
        train_auc = roc_auc_score(y_val, train_probabilities)

        val_fpr, val_tpr, _ = roc_curve(y_val,  val_probabilities)
        val_auc = roc_auc_score(y_val, val_probabilities)

        # 2. Plot the ROC Curve showing the results on training and validation sets
        fig.add_trace(go.Scatter(x=train_fpr, y=train_tpr, name="Train"), row=i+1, col=1) # use enumerated value i to align figures vertically
        fig.add_trace(go.Scatter(x=val_fpr, y=val_tpr, name="Validation"),row=i+1, col=1) # use enumerated value i
        fig.update_xaxes(title_text="False Positive Rate")
        fig.update_yaxes(title_text='True Positive Rate', row=i+1, col=1) # use enumerated value i
        fig.update_layout(title='ROC Curve', height=600)

    return fig, df

# Page C
def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:
    # Restore dataset splits
    X_train, X_val, y_train, y_val = restore_data_splits(df)

    st.markdown("## Get Performance Metrics")
    metric_options = ['precision', 'recall', 'accuracy', 'f1']

    classification_methods_options = ['Logistic Regression',
                                      'Stochastic Gradient Descent with Logistic Regression',
                                      'Stochastic Gradient Descent with Cross Validation',
                                      'Random Forest', 
                                      'SVM', 
                                      'Naïve Bayes']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]
    st.session_state['trained_models'] = trained_models

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained classification models for evaluation',
        options=trained_models
    )
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('## Review Classification Model Performance')

            plot_options = ['Precision/Recall Curve', 'ROC Curve', 'Metric Results', 'Confusion Matrix']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=plot_options
            )

            st.write(df.label.value_counts())

            ############## Task 10: Compute evaluation metrics
            if 'Confusion Matrix' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]
                #trained_select = [st.session_state[model]
                #                 for model in model_select]
                for model in models:
                    plot_cmatrix(X_train, y_train, model)
            if 'Precision/Recall Curve' in review_plot:
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_pr_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            if 'ROC Curve' in review_plot:
                trained_select = [st.session_state[model]
                                  for model in model_select]
                fig, df = plot_roc_curve(
                    X_train, X_val, y_train, y_val, trained_select, model_select)
                st.plotly_chart(fig)

            ############## Task 11: Plot PR Curves
            if 'Metric Results' in review_plot:
                models = [st.session_state[model]
                          for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                # Select multiple metrics for evaluation
                metric_select = st.multiselect(
                    label='Select metrics for classification model evaluation',
                    options=metric_options,
                )
                if (metric_select):
                    st.session_state['metric_select'] = metric_select
                    st.write(
                        'You selected the following metrics: {}'.format(metric_select))

                    for idx, model in enumerate(models):
                        train_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_train, y_train, model, metric_select)
                        val_result_dict[model_select[idx]] = compute_eval_metrics(
                            X_val, y_val, model, metric_select)

                    st.markdown('### Predictions on the training dataset')
                    st.dataframe(train_result_dict)

                    st.markdown('### Predictions on the validation dataset')
                    st.dataframe(val_result_dict)

    st.write('Continue to Deploy Model')
