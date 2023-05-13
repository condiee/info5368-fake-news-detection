import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import string
from pandas.api.types import is_string_dtype
import plotly.express as px
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import  SentimentIntensityAnalyzer
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")
st.markdown("## Real or Fake News?")

#############################################

st.markdown("On this page, you can use one of the models you have trained to predict whether a news article is real or fake news. You can also detect the sentiment of the article.")

#############################################
enc = OrdinalEncoder()

# Checkpoint 11
def deploy_model(text):
    """
    Restore the trained model from st.session_state[‘deploy_model’] 
                and use it to classify the text data   
    Input: 
        - text
    Output: 
        - classification, +1 or -1 to indicate fake/real news OR sentiment
    """
    classification = None
    model = None

    # Add code here
    # 1. Restore the model for deployment in st.session_state[‘deploy_model’]
    if('deploy_model' in st.session_state):
        model = st.session_state['deploy_model']
        
    if(model):
        classification = model.predict(text)

    # 2. Predict the product sentiment of the input text using the predict function e.g.,
    classification = model.predict(text)
    
    # return product sentiment
    return classification

###################### FETCH DATASET #######################
df = None

if 'data' in st.session_state and 'trained_models' in st.session_state and st.session_state['trained_models'] != []:
    df = st.session_state['data']
    # st.write(st.session_state)

    # Select a model to deploy from the trained models
    st.markdown("### Choose which model you would like to deploy to predict article sentiment and integrity:")
    model_select = st.selectbox(
        label='Select from your trained models:',
        options = st.session_state['trained_models'],
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

else:
    st.write('*Please upload a dataset on page A and train and test a machine learning model on pages B and C in order to deploy your model on this page.*')

# Deploy App!
if df is not None:
    # Input review
    st.markdown('### Use a trained classification method to automatically predict the sentiment of a news article and whether it is real and fake news.')
    
    user_input = st.text_input(
        "Enter the body of a news article (you're welcome to include the title at the beginning as well):",
        key="user_review",
    )
    if (user_input):
        # st.write(user_input)

        translator = str.maketrans('', '', string.punctuation)
        # check if the feature contains string or not
        user_input_updates = user_input.translate(translator)
        
        if 'count_vect' in st.session_state:
            count_vect = st.session_state['count_vect']            
            text_count = count_vect.transform([user_input_updates])
            
            if 'tfidf_transformer' in st.session_state:
                tfidf_transformer = st.session_state['tfidf_transformer']
                tfidf = tfidf_transformer.transform(text_count)
                if model_select == 'SVM':
                    encoded_user_input = pd.DataFrame(tfidf.toarray())
                else:
                    encoded_user_input = tfidf             
            else: # word count encoder
                if model_select == 'SVM':
                    encoded_user_input = pd.DataFrame(text_count.toarray())
                else: 
                    encoded_user_input = text_count

            # SENTIMENT ANALYSIS
            sentiment = SentimentIntensityAnalyzer()
            sentiment_score = sentiment.polarity_scores(user_input)
            sentiment, score = max(sentiment_score, key=sentiment_score.get), max(sentiment_score.values())
            if sentiment == 'neu':
                sentiment = "neutral"
            elif sentiment == 'pos':
                sentiment= "positive"
            elif sentiment == 'neg':
                sentiment = "negative"

            classification = deploy_model(encoded_user_input)
            # st.write("classification:", classification[0])
            if(classification == 1):
                decision = "fake"
            elif classification == 0:
                decision = "real"

            st.write(f"This article text is predicted to be **{decision}** news with a **{sentiment}** sentiment (score = {score*100}%).")

            

          

