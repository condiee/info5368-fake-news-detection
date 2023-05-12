import numpy as np                      # pip install numpy
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import random
from helper_functions import fetch_dataset, set_pos_neg_reviews
random.seed(10)
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.title('Train Model')

#############################################

# Checkpoint 4

def split_dataset(df, number, target, feature_encoding, random_state=42):
    """
    This function splits the dataset into the training and test sets.

    Input:
        - X: training features
        - y: training targets
        - number: the ratio of test samples
        - target: article feature name 'rating'
        - feature_encoding: (string) 'Word Count' or 'TF-IDF' encoding
        - random_state: determines random number generation for centroid initialization
    Output:
        - X_train_sentiment: training features (word encoded)
        - X_val_sentiment: test/validation features (word encoded)
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train, X_val, y_train, y_val = [], [], [], []
    X_train_sentiment, X_val_sentiment = [], []

    try:
        # Split dataset into y (target='sentiment') and X (all other features)
        # Add code here
        X = df.loc[:, ~df.columns.isin([target])]
        y = df.loc[:, df.columns.isin([target])]

        # Split the train and test sets into X_train, X_val, y_train, y_val using X, y, number/100, and random_state
        # Add code here
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=random_state)

        # Use the column word_count and tf_idf_word_count as a feature prefix on X_train and X_val sets
        # Add code here
        if feature_encoding == 'Word Count':
            X_train_sentiment = X_train.loc[:,X_train.columns.str.startswith('word_count_')]
            X_val_sentiment = X_val.loc[:,X_val.columns.str.startswith('word_count_')]
        elif feature_encoding == 'TF-IDF':
            X_train_sentiment = X_train.loc[:,X_train.columns.str.startswith('tf_idf_word_count')]
            X_val_sentiment = X_val.loc[:,X_val.columns.str.startswith('tf_idf_word_count')]

        # Compute dataset percentages
        train_percentage = (len(X_train) /
                            (len(X_train)+len(X_val)))*100
        test_percentage = (len(X_val) /
                           (len(X_train)+len(X_val)))*100

        # Print dataset split result
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) \
                    and the test dataset contains {2:.2f} observations ({3:.2f}%).'
                    .format(len(X_train), train_percentage, len(X_val), test_percentage))

        # (Uncomment code) Save train and test split to st.session_state
        st.session_state['X_train'] = X_train_sentiment
        st.session_state['X_val'] = X_val_sentiment
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')

    return X_train_sentiment, X_val_sentiment, y_train, y_val

#logistic regression, random forest, SVM, naive bayes, CNN

# Checkpoint 5
def train_logistic_regression(X_train, y_train, model_name, params, random_state=42):
    """
    This function trains the model with logistic regression and stores it in st.session_state[model_name].

    Input:
        - X_train: training features (review features)
        - y_train: training targets
        - model_name: (string) model name
        - params: a dictionary with lg hyperparameters: max_iter, solver, tol, and penalty
        - random_state: determines random number generation for centroid initialization
    Output:
        - lg_model: the trained model
    """
    lg_model = None
    # Add code here
    # 1. Create a try and except block to train a logistic regression model.
    try:
        # 2. Create a LogisticRegression class object using the random_state as input.
        lg_model = LogisticRegression(random_state=random_state, max_iter=params['max_iter'], 
                                      solver=params['solver'], tol=params['tol'], penalty=params['penalty'])

        # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
        # Remember to create a continuous y_train array using np.ravel() function.
        lg_model.fit(X_train, np.ravel(y_train))
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = lg_model

    except:
        print('Exception thrown; cannot train logit model')

    # 5. Return the trained model
    return lg_model

def train_random_forest(X_train, y_train, model_name, params, random_state=42):

    rf_model = None
    # Add code here
    # 1. Create a try and except block to train a logistic regression model.
    try:
        # 2. Create a LogisticRegression class object using the random_state as input.
        rf_model = RandomForestClassifier(random_state=random_state, n_estimators=params['n_estimators'], 
                                      max_depth=params['max_depth'])

        # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
        # Remember to create a continuous y_train array using np.ravel() function.
        rf_model.fit(X_train, y_train)
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = rf_model

    except:
        print('Exception thrown; cannot train random forest model')

    # 5. Return the trained model
    return rf_model

def train_svm(X_train, y_train, model_name, params, random_state=42):
    
    svm_model = None
    # Add code here
    # 1. Create a try and except block to train a logistic regression model.
    try:
        # 2. Create a LogisticRegression class object using the random_state as input.
        svm_model = SVC(random_state=random_state, kernal=params['kernal'], 
                                      C=params['C'])

        # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
        # Remember to create a continuous y_train array using np.ravel() function.
        svm_model.fit(X_train, y_train)
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = svm_model

    except:
        print('Exception thrown; cannot train svm model')

    # 5. Return the trained model
    return svm_model

def train_naive_bayes(X_train, y_train, model_name, params, random_state=42):
    
    nb_model = None
    # Add code here
    # 1. Create a try and except block to train a logistic regression model.
    try:
        # 2. Create a LogisticRegression class object using the random_state as input.
        nb_model = GaussianNB()

        # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
        # Remember to create a continuous y_train array using np.ravel() function.
        nb_model.fit(X_train, y_train)
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = nb_model

    except:
        print('Exception thrown; cannot train naive bayes model')

    # 5. Return the trained model
    return nb_model

# Checkpoint 8

def inspect_coefficients(trained_models):
    """
    This function gets the coefficients of the trained models and displays the model name and coefficients

    Input:
        - trained_models: list of trained names (strings)
    Output:
        - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
    """
    out_dict = {'Logistic Regression': [],
                'Random Forest': [], 
                'SVM': [], 
                'Naïve Bayes': []}
    # Add code here
    for name, model in trained_models.items():
        if (model is not None):
            coef = model.coef_
            out_dict[name] = coef
            if coef is not None:
                st.write('### Coefficients inspection for {0}'.format(name))
                st.dataframe(out_dict[name])
                st.write('Total number of coefficients: {0}'.format(coef.shape[1]))
                st.write('Number of positive coefficients: {0}'.format(
                    len(coef[coef > 0])))
                st.write('Number of negative coefficients: {0}'.format(
                    len(coef[coef < 0])))
            else:
                st.write('Coefficients for {0} is None'.format(name))

            if ('cv_results_' in st.session_state):
                st.dataframe(st.session_state['cv_results_'])
                        
    return out_dict


###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display dataframe as table
    st.dataframe(df)

    st.markdown('### Do you want to predict whether an article is real or fake news, the article sentiment, or both?')
    options = ['Real or Fake News', 'Sentiment Analysis']
    select_options = st.multiselect(
        'Select what to predict:',
        options,
    )
    if (select_options):
        for option in select_options:
            # if option == options[1]: # sentiment analysis
                #do stuff
            if option == options[0]: # real/fake news
                st.session_state['target'] = 'label'

    # if (select_options):
    #     for option in select_options:
    #         if option == options[1]: # sentiment analysis
                # Select positive and negative articles
                # pos_neg_select = st.slider(
                #     'Select a range of ratings for negative reviews',
                #     1, 5, 3,
                #     key='pos_neg_selectbox')

                # if (pos_neg_select and st.button('Set negative sentiment upper bound')):
                #     df = set_pos_neg_reviews(df, pos_neg_select)

                #     st.write('You selected ratings positive rating greater than {}'.format(
                #         pos_neg_select))

    # Select variable to predict
    # feature_predict_select = st.selectbox(
    #     label='Select variable to predict',
    #     index=df.columns.get_loc(
    #         'sentiment') if 'sentiment' in df.columns else 0,
    #     options=df.columns,
    #     key='feature_selectbox',
    # )

    # st.session_state['target'] = feature_predict_select

    word_count_encoder_options = ['Word Count', 'TF-IDF']
    if ('word_encoder' in st.session_state):
        if (st.session_state['word_encoder'] is not None):
            word_count_encoder_options = st.session_state['word_encoder']
            st.write('Restoring selected encoded features {}'.format(
                word_count_encoder_options))

    # Select input features
    feature_input_select = st.selectbox(
        label='Select features for classification input',
        options=word_count_encoder_options,
        key='feature_select'
    )

    st.session_state['feature'] = feature_input_select

    # st.write('You selected input {} and output {}'.format(
    #     feature_input_select, feature_predict_select))

    # Task 4: Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    X_train, X_val, y_train, y_val = [], [], [], []
    # Compute the percentage of test and training data
    if (feature_predict_select in df.columns):
        X_train, X_val, y_train, y_val = split_dataset(
            df, number, feature_predict_select, feature_input_select)

    classification_methods_options = ['Logistic Regression',
                                      'Random Forest', 
                                      'SVM', 
                                      'Naïve Bayes']

    trained_models = [
        model for model in classification_methods_options if model in st.session_state]

    # Collect ML Models of interests
    classification_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        classification_model_select))

    # Add parameter options to each regression method

    # Task 5: Logistic Regression
    if (classification_methods_options[0] in classification_model_select or classification_methods_options[0] in trained_models):
        st.markdown('#### ' + classification_methods_options[0])

        lg_col1, lg_col2 = st.columns(2)

        with (lg_col1):
            # solver: algorithm to use in the optimization problem
            solvers = ['liblinear', 'lbfgs', 'newton-cg',
                       'newton-cholesky', 'sag', 'saga']
            lg_solvers = st.selectbox(
                label='Select solvers for SGD',
                options=solvers,
                key='lg_reg_solver_multiselect'
            )
            st.write('You select the following solver(s): {}'.format(lg_solvers))

            # penalty: 'l1' or 'l2' regularization
            lg_penalty_select = st.selectbox(
                label='Select penalty for SGD',
                options=['l2', 'l1'],
                key='lg_penalty_multiselect'
            )
            st.write('You select the following penalty: {}'.format(
                lg_penalty_select))

        with (lg_col2):
            # tolerance: stopping criteria for iterations
            lg_tol = st.text_input(
                label='Input a tolerance value',
                value='0.01',
                key='lg_tol_textinput'
            )
            lg_tol = float(lg_tol)
            st.write('You select the following tolerance value: {}'.format(lg_tol))

            # max_iter: maximum iterations to run the LG until convergence
            lg_max_iter = st.number_input(
                label='Enter the number of maximum iterations on training data',
                min_value=1000,
                max_value=5000,
                value=1000,
                step=100,
                key='lg_max_iter_numberinput'
            )
            st.write('You set the maximum iterations to: {}'.format(lg_max_iter))

        lg_params = {
            'max_iter': lg_max_iter,
            'penalty': lg_penalty_select,
            'tol': lg_tol,
            'solver': lg_solvers,
        }
        if st.button('Train Logistic Regression Model'):
            train_logistic_regression(
                X_train, y_train, classification_methods_options[0], lg_params)

        if classification_methods_options[0] not in st.session_state:
            st.write('Logistic Regression Model is untrained')
        else:
            st.write('Logistic Regression Model trained')

    # # Task 6: Stochastic Gradient Descent with Logistic Regression
    # if (classification_methods_options[1] in classification_model_select or classification_methods_options[2] in trained_models):
    #     st.markdown('#### ' + classification_methods_options[1])

    #     # Loss: 'log' is logistic regression, 'hinge' for Support Vector Machine
    #     sdg_loss_select = 'log'

    #     sgd_col1, sgd_col2 = st.columns(2)

    #     with (sgd_col1):
    #         # max_iter: maximum iterations to run the iterative SGD
    #         sdg_max_iter = st.number_input(
    #             label='Enter the number of maximum iterations on training data',
    #             min_value=1000,
    #             max_value=5000,
    #             value=1000,
    #             step=100,
    #             key='sgd_max_iter_numberinput'
    #         )
    #         st.write('You set the maximum iterations to: {}'.format(sdg_max_iter))

    #         # penalty: 'l1' or 'l2' regularization
    #         sdg_penalty_select = st.selectbox(
    #             label='Select penalty for SGD',
    #             options=['l2', 'l1'],
    #             key='sdg_penalty_multiselect'
    #         )
    #         st.write('You select the following penalty: {}'.format(
    #             sdg_penalty_select))

    #     with (sgd_col2):
    #         # alpha=0.001: Constant that multiplies the regularization term. Ranges from [0 Inf)
    #         sdg_alpha = st.text_input(
    #             label='Input one alpha value',
    #             value='0.001',
    #             key='sdg_alpha_numberinput'
    #         )
    #         sdg_alpha = float(sdg_alpha)
    #         st.write('You select the following alpha value: {}'.format(sdg_alpha))

    #         # tolerance: stopping criteria for iterations
    #         sgd_tol = st.text_input(
    #             label='Input a tolerance value',
    #             value='0.01',
    #             key='sgd_tol_textinput'
    #         )
    #         sgd_tol = float(sgd_tol)
    #         st.write('You select the following tolerance value: {}'.format(sgd_tol))

    #     sgd_params = {
    #         'loss': sdg_loss_select,
    #         'max_iter': sdg_max_iter,
    #         'penalty': sdg_penalty_select,
    #         'tol': sgd_tol,
    #         'alpha': sdg_alpha,
    #     }

    #     if st.button('Train Stochastic Gradient Descent Model'):
    #         train_sgd_classifer(
    #             X_train, y_train, classification_methods_options[1], sgd_params)

    #     if classification_methods_options[1] not in st.session_state:
    #         st.write('Stochastic Gradient Descent Model is untrained')
    #     else:
    #         st.write('Stochastic Gradient Descent Model trained')

    # # ############## Task 7: Stochastic Gradient Descent with Logistic Regression with Cross Validation
    # if (classification_methods_options[2] in classification_model_select or classification_methods_options[2] in trained_models):
    #     st.markdown('#### ' + classification_methods_options[2])

    #     # Loss: "squared_error": Ordinary least squares, huber": Huber loss for robust regression, "epsilon_insensitive": linear Support Vector Regression.
    #     sdgcv_loss_select = 'log_loss'

    #     sdg_col, sdgcv_col = st.columns(2)

    #     # Collect Parameters
    #     with (sdg_col):
    #         # max_iter: maximum iterations to run the iterative SGD
    #         sdgcv_max_iter = st.number_input(
    #             label='Enter the number of maximum iterations on training data',
    #             min_value=1000,
    #             max_value=5000,
    #             value=1000,
    #             step=100,
    #             key='sgdcv_max_iter_numberinput'
    #         )
    #         st.write('You set the maximum iterations to: {}'.format(sdgcv_max_iter))

    #         # penalty: 'l1' or 'l2' regularization
    #         sdgcv_penalty_select = st.selectbox(
    #             label='Select penalty for SGD',
    #             options=['l2', 'l1'],
    #             # default='l1',
    #             key='sdgcv_penalty_select'
    #         )
    #         st.write('You select the following penalty: {}'.format(
    #             sdgcv_penalty_select))

    #         # tolerance: stopping criteria for iterations
    #         sgdcv_tol = st.text_input(
    #             label='Input a tolerance value',
    #             value='0.01',
    #             key='sdgcv_tol_numberinput'
    #         )
    #         sgdcv_tol = float(sgdcv_tol)
    #         st.write(
    #             'You select the following tolerance value: {}'.format(sgdcv_tol))

    #     # Collect Parameters
    #     with (sdgcv_col):
    #         # alpha=0.01: Constant that multiplies the regularization term. Ranges from [0 Inf)
    #         sdgcv_alphas = st.text_input(
    #             label='Input alpha values, separate by comma',
    #             value='0.001,0.0001',
    #             key='sdgcv_alphas_textinput'
    #         )
    #         sdgcv_alphas = [float(val) for val in sdgcv_alphas.split(',')]
    #         st.write(
    #             'You select the following alpha value: {}'.format(sdgcv_alphas))

    #         sgdcv_params = {
    #             'loss': [sdgcv_loss_select],
    #             'max_iter': [sdgcv_max_iter],
    #             'penalty': [sdgcv_penalty_select],
    #             'tol': [sgdcv_tol],
    #             'alpha': sdgcv_alphas,
    #         }

    #         st.markdown('Select SGD Cross Validation Parameters')
    #         # n_splits: number of folds
    #         sgdcv_cv_n_splits = st.number_input(
    #             label='Enter the number of folds',
    #             min_value=2,
    #             max_value=len(df),
    #             value=3,
    #             step=1,
    #             key='sdgcv_cv_nsplits'
    #         )
    #         st.write('You select the following split value(s): {}'.format(
    #             sgdcv_cv_n_splits))

    #         sgdcv_cv_params = {
    #             'n_splits': sgdcv_cv_n_splits,
    #         }

    #     if st.button('Train Stochastic Gradient Descent Model with Cross Validation'):
    #         train_sgdcv_classifer(
    #             X_train, y_train, classification_methods_options[2], sgdcv_params, sgdcv_cv_params)

    #     if classification_methods_options[2] not in st.session_state:
    #         st.write(
    #             'Stochastic Gradient Descent Model with Cross Validation is untrained')
    #     else:
    #         st.write(
    #             'Stochastic Gradient Descent Model with Cross Validation trained')

    # Task 9: Inspect classification coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select features for classification input',
        options=classification_model_select,
        key='inspect_multiselect'
    )

    models = {}
    for model_name in inspect_models:
        if (model_name in st.session_state):
            models[model_name] = st.session_state[model_name]
    coefficients = inspect_coefficients(models)

    st.write('Continue to Test Model')
