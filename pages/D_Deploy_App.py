import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import string
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
                and use it to predict the sentiment of the input data    
    Input: 
        - text
    Output: 
        - product_sentiment: product sentiment class, +1 or -1
    """
    product_sentiment = None

    # Add code here
    # 1. Restore the model for deployment in st.session_state[‘deploy_model’]
    model = st.session_state['deploy_model']

    # 2. Predict the product sentiment of the input text using the predict function e.g.,
    product_sentiment = model.predict(text)
    
    # return product sentiment
    return product_sentiment

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
    #df.dropna(inplace=True)
    st.markdown('### Introducing the ML Powered Review Application')

    # Input review
    st.markdown('### Use a trained classification method to automatically predict the sentiment of a news article and whether it is real and fake news.')
    
    user_input = st.text_input(
        "Enter the body of a news article (you're welcome to include the title at the beginning as well):",
        key="user_review",
    )
    if (user_input):
        st.write(user_input)

        translator = str.maketrans('', '', string.punctuation)
        # check if the feature contains string or not
        user_input_updates = user_input.translate(translator)
        
        if 'count_vect' in st.session_state:
            count_vect = st.session_state['count_vect']
            text_count = count_vect.transform([user_input_updates])
            # Initialize encoded_user_input with text_count as default
            encoded_user_input = text_count
            if 'tfidf_transformer' in st.session_state:
                tfidf_transformer = st.session_state['tfidf_transformer']
                encoded_user_input = tfidf_transformer.transform(text_count)
            
            #product_sentiment = st.session_state["deploy_model"].predict(encoded_user_input)
            product_sentiment = deploy_model(encoded_user_input)
            sentiment = None

            st.write("**Sentiment prediction:**", product_sentiment[0])

            if(product_sentiment == 1):
                sentiment = "positive"
            elif product_sentiment == -1:
                sentiment = "negative"

            if sentiment:
                st.write(f'Your trained model predicts that the news article has a {sentiment} sentiment.')



