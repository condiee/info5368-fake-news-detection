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
     #     # Add code here
#     # 1. Create a try and except block to train a logistic regression model.
    try:
#         # 2. Create a LogisticRegression class object using the random_state as input.
        lg_model = LogisticRegression(random_state=random_state, max_iter=params['max_iter'], 
                                       solver=params['solver'], tol=params['tol'], penalty=params['penalty'])

#             # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
#             # Remember to create a continuous y_train array using np.ravel() function.
        lg_model.fit(X_train, np.ravel(y_train))
    
#         # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = lg_model

    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train logit model', e)

#     # 5. Return the trained model
    return lg_model

# logistic regression, random forest, SVM, naive bayes

def train_grid_logistic_regression(X_train, y_train, model_name):
    lg = None
    try:
        lg = LogisticRegression()
        param_grid = {'solver' : ['liblinear', 'saga'], # 'lbfgs', 'sag', 
                       'penalty': ['l1', 'l2'],
                        'tol' : [0.0001, 0.001, 0.01,0.1 ], 
                        'max_iter' : [1000, 3000, 5000]}
        lg_model = GridSearchCV(lg, param_grid, cv=10)
        lg_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = lg_model
        st.write("tuned hpyerparameters: (best parameters) ",lg_model.best_params_)
        st.write("accuracy :",lg_model.best_score_)
    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train logit model. ERROR:', e)

    # 5. Return the trained model
    return lg_model

def train_grid_random_forest(X_train, y_train, model_name):
    rf = None
    try:
        rf = RandomForestClassifier()
        param_grid = {'n_estimators' : [50,100,150,200], 'max_depth' : [10,20,30,40,50]}
        rf_model = GridSearchCV(rf, param_grid, cv=5)
        rf_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = rf_model
        st.write("tuned hpyerparameters: (best parameters) ", rf_model.best_params_)
        st.write("accuracy :", rf_model.best_score_)
    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train random forest model. ERROR:', e)

    # 5. Return the trained model
    return rf_model

def train_random_forest(X_train, y_train, model_name, params, random_state=42):

    rf_model = None
    # Add code here
    try:
        rf_model = RandomForestClassifier(random_state=random_state, n_estimators=params['n_estimators'], 
                                      max_depth=params['max_depth'])

            # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
            # Remember to create a continuous y_train array using np.ravel() function.
        rf_model.fit(X_train, np.ravel(y_train))
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = rf_model

    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train random forest model. ERROR:', e)

    # 5. Return the trained model
    return rf_model

def train_grid_svm(X_train, y_train, model_name):
    svm = None
    try:
        svm = SVC(probability=True)
        param_grid = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C' : [0.001, 0.01, 0.1, 1, 10, 100]}        
        svm_model = GridSearchCV(svm, param_grid, cv=5)
        svm_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = svm_model
        st.write("tuned hpyerparameters: (best parameters) ", svm_model.best_params_)
        st.write("accuracy :", svm_model.best_score_)
    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train svm model. ERROR:', e)
    # 5. Return the trained model
    return svm_model

def train_svm(X_train, y_train, model_name, params, random_state=42):
    
    svm_model = None
    # Add code here
    try:
        svm_model = SVC(random_state=random_state, kernel=params['kernel'], 
                                      C=params['C'])

            # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
            # Remember to create a continuous y_train array using np.ravel() function.
        svm_model.fit(X_train, np.ravel(y_train))
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = svm_model

    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train svm model. ERROR:', e)

    # 5. Return the trained model
    return svm_model

def train_grid_naive_bayes(X_train, y_train, model_name):
    nb = None
    try:
        nb = GaussianNB()
        param_grid = {'var_smoothing' : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}       
        nb_model = GridSearchCV(nb, param_grid, cv=5)
        nb_model.fit(X_train, np.ravel(y_train))
        st.session_state[model_name] = nb_model
        st.write("tuned hpyerparameters: (best parameters) ", nb_model.best_params_)
        st.write("accuracy :", nb_model.best_score_)
    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
        print('Exception thrown; cannot train naive bayes model. ERROR:', e)

    # 5. Return the trained model
    return nb_model

def train_naive_bayes(X_train, y_train, model_name, params, random_state=42):
    
    nb_model = None
    # Add code here
    # 1. Create a try and except block to train a logistic regression model.
    try:
        nb_model = GaussianNB(var_smoothing=params['var_smoothing'])

            # 3. Fit the model to the data using the fit() function with input data X_train, y_train.
            # Remember to create a continuous y_train array using np.ravel() function.
        nb_model.fit(X_train, y_train)
    
        # 4. Save the model in st.session_state[model_name].
        st.session_state[model_name] = nb_model

    except Exception as e:
        st.write("Please choose different hyperparameters:",e)
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

    # print("MODELS:", trained_models)
    # 1. Write a for loop through the model names and trained models.
    for name, model in trained_models.items():
        # 2. In the for loop,
        # a. check that the model is not None
        # assert model is not None
        # b. If the model is valid, store the coefficients in out_dict[name] using model.coef
        # (same for all models) and display the coefficients.
        if model is not None:
            out_dict[name] = model.coef_
            st.write(f"**{name}** coefficents: {model.coef_[0]}")

            # c. Compute and print the following values:
            # i. Total number of coefficients
            st.write(f"There are {len(model.coef_[0])} total coefficients.")

            # ii. Number of positive coefficients
            st.write(f"There are {sum([x for x in model.coef_[0]>=0])} positive coefficients.")

            # iii. Number of negative coefficients
            st.write(f"There are {sum([x for x in model.coef_[0]<0])} negative coefficients.")

    # 3. Display ‘cv_results_’ in st.session_state[‘cv_results_’] if it exists (from Checkpoint 7)
    if 'cv_results_' in st.session_state:
        st.write("**Cross Validation Results:**", st.session_state['cv_results_'])
                        
    return out_dict



###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display dataframe as table
    st.write('### Displaying cleaned and processed data:')
    st.dataframe(df)

    # Select variable to predict
    # feature_predict_select = st.selectbox(
    #     label='Select variable to predict',
    #     index=df.columns.get_loc(
    #         'sentiment') if 'sentiment' in df.columns else 0,
    #     options=df.columns,
    #     key='feature_selectbox',
    # )
    feature_predict_select = 'label'
    st.session_state['target'] = feature_predict_select

    word_count_encoder_options = ['Word Count', 'TF-IDF']
    if ('word_encoder' in st.session_state):
        if (st.session_state['word_encoder'] is not None):
            word_count_encoder_options = st.session_state['word_encoder']
            st.write('Restoring selected encoded features {}'.format(
                word_count_encoder_options))

    # Select input features
    feature_input_select = st.selectbox(
        label='Select (encoded) features for classification input',
        options=word_count_encoder_options,
        key='feature_select'
    )

    st.session_state['feature'] = feature_input_select

    # TODO: fix this so it isn't misleading what you're training on (article text!)
    st.write('You selected **{}** as input and **{}** as predicted output.'.format(
        feature_input_select, feature_predict_select))

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
    coefficient_models_options = ['Logistic Regression', 
                                      'SVM', 
                                      'Naïve Bayes']


    trained_models = [
        model for model in classification_methods_options if model in st.session_state]

    # Collect ML Models of interests
    classification_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=classification_methods_options,
    )
    st.write('You selected the following models to train: {}'.format(
        classification_model_select))

    # Add parameter options to each regression method
    hyp_training_options = ['Use Grid Search Cross Validation to find the best parameters', 'Enter Parameters Manually']

    # Task 5: Logistic Regression
    if (classification_methods_options[0] in classification_model_select or classification_methods_options[0] in trained_models):
        st.markdown('#### ' + classification_methods_options[0])

        logit_param_options = st.selectbox('Select how to choose hyperparameters', 
                                           options=hyp_training_options,
                                           key='logit')
        if logit_param_options == hyp_training_options[0]:
            if st.button('Train Logistic Regression Model using Grid Search'):
                train_grid_logistic_regression(
                    X_train, y_train, classification_methods_options[0])
        elif logit_param_options == hyp_training_options[1]:
            lg_col1, lg_col2 = st.columns(2)
            with (lg_col1):
                 # solver: algorithm to use in the optimization problem
                 solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                 lg_solvers = st.selectbox(
                 label='Select solvers',
                 options=solvers,
                 key='lg_reg_solver_multiselect'
                 )
                 st.write('You select the following solver(s): {}'.format(lg_solvers))

        #         # penalty: 'l1' or 'l2' regularization
                 lg_penalty_select = st.selectbox(
                 label='Select penalty',
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
            if st.button("Clear Trained Logistic Regression Model"):
                del st.session_state[classification_methods_options[0]]

    # Task 6: Random Forest
    if (classification_methods_options[1] in classification_model_select or classification_methods_options[1] in trained_models):
        st.markdown('#### ' + classification_methods_options[1])

        rf_param_options = st.selectbox('Select how to choose hyperparameters',
                                        options=hyp_training_options,
                                        key='rf')
        if rf_param_options == hyp_training_options[0]:
            if st.button('Train Random Forest Model using Grid Search'):
                train_grid_random_forest(
                    X_train, y_train, classification_methods_options[1])
        elif rf_param_options == hyp_training_options[1]:
            rf_col1, rf_col2 = st.columns(2)
            with (rf_col1):
                n_est = st.number_input(
                label='Enter the number of estimators on training data',
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                key='rf_n_est_numberinput'
                )
                st.write('You set the number of estimators to: {}'.format(n_est))
            with (rf_col2):
                max_dep = st.number_input(
                label='Enter the max depth on training data',
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                key='rf_max_dep_numberinput'
                )
                st.write('You set the max depth to: {}'.format(max_dep))

            rf_params = {
            'n_estimators': n_est,
            'max_depth': max_dep,
            }
            if st.button('Train Random Forest Model'):
                train_random_forest(
                X_train, y_train, classification_methods_options[1], rf_params)

        if classification_methods_options[1] not in st.session_state:
            st.write('Random Forest Model is untrained')
        else:
            st.write('Random Forest Model trained')
            if st.button("Clear Trained Random Forest Model"):
                del st.session_state[classification_methods_options[1]]

    # Task 7: SVM
    if (classification_methods_options[2] in classification_model_select or classification_methods_options[2] in trained_models):
        st.markdown('#### ' + classification_methods_options[2])
            
        svm_param_options = st.selectbox('Select how to choose hyperparameters', 
                                           options=hyp_training_options,
                                           key='svm')
        if svm_param_options == hyp_training_options[0]:
            if st.button('Train SVM Model using Grid Search'):
                train_grid_svm(
                X_train, y_train, classification_methods_options[2])
        elif svm_param_options == hyp_training_options[1]:
            svm_col1, svm_col2 = st.columns(2)
            with (svm_col1):
                c = st.number_input(
                label='Enter the value of C',
                min_value=0.0001,
                max_value=100.0,
                value=0.1,
                step=0.1,
                key='svm_c_numberinput'
                )
                st.write('You set c to: {}'.format(c))
            with (svm_col2):
                kernel = st.selectbox(
                label='Select kernel for SVM',
                options=['linear', 'poly', 'rbf', 'sigmoid'],
                key='svm_kernal_multiselect'
                )
                st.write('You set the kernel to: {}'.format(kernel))

            svm_params = {
            'C': c,
            ### changing to e here
            'kernel': kernel,
            }
            if st.button('Train SVM Model'):
                train_svm(
                X_train, y_train, classification_methods_options[2], svm_params)

        if classification_methods_options[2] not in st.session_state:
            st.write('SVM Model is untrained')
        else:
            st.write('SVM Model trained')
            if st.button("Clear Trained SVM Model"):
                del st.session_state[classification_methods_options[2]]

    # Task 8: Naive Bayes
    if (classification_methods_options[3] in classification_model_select or classification_methods_options[3] in trained_models):
        st.markdown('#### ' + classification_methods_options[3])

        nb_param_options = st.selectbox('Select how to choose hyperparameters',
                                        options=hyp_training_options,
                                        key='nb')
        if nb_param_options == hyp_training_options[0]:
            if st.button('Train Naïve Bayes Model using Grid Search'):
                train_grid_naive_bayes(
                    X_train, y_train, classification_methods_options[3])
        elif nb_param_options == hyp_training_options[1]:
            var = st.number_input(
            label='Enter the value of var smoothing',
            min_value=0.0000001,
            max_value=0.1,
            value=0.1,
            step=0.01,
            key='nb_var_numberinput'
            )
            st.write('You set var smoothing to: {}'.format(var))

            nb_params = {
            'var_smoothing': var,
            }
            if st.button('Train Naive Bayes Model'):
                train_naive_bayes(
                X_train, y_train, classification_methods_options[3], nb_params)

        if classification_methods_options[3] not in st.session_state:
            st.write('Naieve Bayes Model is untrained')
        else:
            st.write('Naive Bayes Model trained')
            if st.button("Clear Trained Naive Bayes Model"):
                del st.session_state[classification_methods_options[3]]

    # st.write("#### DANGER ZONE: Clear all trained models?")
    # if st.button("Clear All Trained Models"):
    #     for model in classification_methods_options:
    #         st.write('checking', model)
    #         if model in st.session_state:
    #             st.write('in stsate')
    #             del st.session_state[model]
    #             st.write('del', st.session_state) # doesn't refresh

    # Task 9: Inspect classification coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select features for classification input',
        options= coefficient_models_options,
        key='inspect_multiselect'
    )

    models = {}
    for model_name in inspect_models:
        if (model_name in st.session_state):
            models[model_name] = st.session_state[model_name]
    coefficients = inspect_coefficients(models)

    st.write('Continue to Test Model')
