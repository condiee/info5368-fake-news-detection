import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import string
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from helper_functions import fetch_dataset, clean_data, summarize_review_data, display_review_keyword, remove_review
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################

# Checkpoint 1
def remove_punctuation(df, features):
    """
    This function removes punctuation from features (i.e., product reviews)
    Input: 
        - df: the pandas dataframe
        - feature: the features to remove punctation
    Output: 
        - df: dataframe with updated feature with removed punctuation
    """

    def remove_punc(text):
        try: # python 2.x
            text = text.translate(None, string.punctuation) 
        except: # python 3.x
            translator = text.maketrans('', '', string.punctuation)
            text = text.translate(translator)
        return text

    for feature in features:
    # check if the feature contains string or not
    # Add code here
        if type(df[feature][0]) == str:

    # applying translate method eliminating punctuations
    # Add code here
            df[feature] = df[feature].apply(remove_punc)

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Confirmation statement
    st.write('Punctuation was removed from {}'.format(features))
    return df

# Checkpoint 2
def word_count_encoder(df, feature, word_encoder):
    """
    This function performs word count encoding on feature in the dataframe
    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform word count encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with word count feature
    """
    # Add code here
    # 1. Use the CountVectorizer() to create a count vectorizer class object.
    count_vect = CountVectorizer()

    # for feature in features:
    # st.write(feature)
    # 2. Use the count vectorizer transform() function to the feature in df to create frequency
    # counts for words.
    frequency_counts = count_vect.fit_transform(df[feature])
    
    # 3. Convert the frequency counts to an array using the toarray() function and convert the
    # array to a pandas dataframe.
    word_count_df = pd.DataFrame(frequency_counts.toarray())

    # 4. Add a prefix to the column names in the data frame created in Step 3 using add_prefix()
    # pandas function with ‘word_count_’ as the prefix.
    word_count_df = word_count_df.add_prefix('word_count_')

    # 5. Add the word count dataframe to df using the pd.concat() function.
    df = pd.concat([df, word_count_df], axis=1)

    # 6. Update the confirmation statement to show the length of the word_count dataframe
    # (Uncomment code) Show confirmation statement
    st.write('Feature {} has been word count encoded from {} reviews.'.format(
    feature, len(word_count_df)))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('Word Count')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect

    return df

# Checkpoint 3
def tf_idf_encoder(df, feature, word_encoder):
    """
    This function performs tf-idf encoding on the given features
    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform tf-idf encoding
        - word_encoder: list of strings with word encoding names 'TF-IDF', 'Word Count'
    Output: 
        - df: dataframe with tf-idf encoded feature
    """
    # Add code here
    # 1. Use the CountVectorizer() to create a count vectorizer class object.
    count_vect = CountVectorizer()

    # 2. Use the count vectorizer transform() function to the feature in df to create frequency
    # counts for words.
    frequency_counts = count_vect.fit_transform(df[feature])

    # 3. Use the TfidfTransformer() to create a TF-IDF transformer class object.
    tfidf_transformer = TfidfTransformer()

    # 4. Transform the frequency counts (from Step 2) into TF-IDF features using the
    # TfidfTransformer object.
    tfidf = tfidf_transformer.fit_transform(frequency_counts)

    # 5. Create a pandas dataframe for the TF-IDF features which takes the TF-IDF features
    # array as input so convert the TF-IDF features to an array using the toarray() function.
    word_count_df = pd.DataFrame(tfidf.toarray())
    
    # 6. Add a prefix to the column names in the data frame created in Step 3 using add_prefix()
    # pandas function with ‘tf_idf_word_count_’ as the prefix.
    word_count_df = word_count_df.add_prefix('tf_idf_word_count_')
    
    # 7. Add the TF-IDF dataframe to df using the pd.concat() function
    df = pd.concat([df, word_count_df], axis=1)
    
    # (Uncomment code) Show confirmation statement
    st.write(
       'Feature {} has been TF-IDF encoded from {} reviews.'.format(feature, len(word_count_df)))

    # (Uncomment code) Store new features in st.session_state
    st.session_state['data'] = df

    # (Uncomment code) Save variables for restoring state
    word_encoder.append('TF-IDF')
    st.session_state['word_encoder'] = word_encoder
    st.session_state['count_vect'] = count_vect
    st.session_state['tfidf_transformer'] = tfidf_transformer
    return df


#def generate_word_cloud(text):
#    font_path = '/System/Library/Fonts/Geneva.ttf'
#    wordcloud = WordCloud(
#        width = 3000,
#        height = 2000,
#        background_color = 'black',
#        #font_path = font_path
#    ).generate(str(text)),
#    fig = plt.figure(
#        figsize = (40, 30),
#        facecolor = 'k',
#       edgecolor = 'k')
#    plt.imshow(wordcloud, interpolation = 'bilinear')
#    plt.axis('off')
#    plt.tight_layout(pad=0)
#    plt.show()

def generate_word_cloud(text):
    wordcloud = WordCloud().generate(str(text))
    #plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud)
    st.pyplot(fig)
    #plt.show()
###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('Upload successful. See the unprocessed dataset below.')

    st.dataframe(df)

    # Remove irrelevant features
    if st.button('Clean Dataset'):
        df, data_cleaned = clean_data(df)
        if (data_cleaned):
            st.markdown('Cleaned Dataset')
            st.dataframe(df)

    st.markdown('### Visualize Features')

    ############## World Cloud
    st.markdown('## Word Cloud')
    st.markdown('Select subject to create word cloud')
    article_subjects = list(set(df.subject.values))
    article_subjects.append('all')
    cloud_feature = st.multiselect(
        'Select article subject or all to see all subjects to create word cloud',
        article_subjects,
    )
    if (cloud_feature):
        for x in cloud_feature:
            if x == 'all':
                generate_word_cloud(df.text.values)
            else:
                t = df.loc[df['subject'] == x]
                generate_word_cloud(t.text.values)
    
    st.markdown('## Feature Graphs')
    st.markdown('Select feature to view graphically')
    graph_sub = ['label', 'subject']
    graph_feature = st.multiselect(
        'Select feature to view',
        graph_sub,
    )
    if (graph_feature):
        for x in graph_feature:
            if x == 'label':
                st.bar_chart(df.label.value_counts())
            elif x == 'subject':
                st.bar_chart(df.subject.value_counts())


    # Summarize reviews
    st.markdown('### Describe Data')
    object_columns = df.select_dtypes(include=['object']).columns
    summarize_reviews = st.selectbox(
        'Select the feature from the dataset',
        object_columns,
    )
    if(summarize_reviews):
        # Show summary of reviews
        summary = summarize_review_data(df, summarize_reviews)

    # Inspect Reviews
    st.markdown('### Inspect Articles')

    review_keyword = st.text_input(
        "Enter a keyword to search in articles",
        key="review_keyword",
    )

    # Display dataset
    st.dataframe(df)

    if (review_keyword):
        displaying_review = display_review_keyword(df, review_keyword)
        st.write(displaying_review)

    # Remove Reviews: number_input for index of review to remove
    st.markdown('### Remove Irrelevant/Useless Articles')
    review_idx = st.number_input(
        label='Enter article index',
        min_value=0,
        max_value=len(df),
        value=0,
        step=1)

    if (review_idx):
        df = remove_review(df, review_idx)
        st.write('Article at index {} has been removed'.format(review_idx))

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)
    word_encoder = []

    word_count_col, tf_idf_col = st.columns(2)

    ############## Task 2: Perform Word Count Encoding
    with (word_count_col):
        text_feature_select_int = st.selectbox(
            'Select text features for encoding word count',
            string_columns,
        )
        if (text_feature_select_int and st.button('Word Count Encoder')):
            df = word_count_encoder(df, text_feature_select_int, word_encoder)

    ############## Task 3: Perform TF-IDF Encoding
    with (tf_idf_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for encoding TF-IDF',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('TF-IDF Encoder')):
            df = tf_idf_encoder(df, text_feature_select_onehot, word_encoder)


    # Show updated dataset
    if (text_feature_select_int or text_feature_select_onehot):
        st.write(df)

    # Save dataset in session_state
    st.session_state['data'] = df

    st.write('Continue to Train Model')