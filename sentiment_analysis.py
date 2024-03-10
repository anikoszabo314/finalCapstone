# This program performs sentiment analysis on a dataset of product reviews

# ================================== Importing packages ================================

# Import the packages we'll need
import numpy as np
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the small English language model
nlp = spacy.load('en_core_web_sm')

# Add the SpacyTextBlob component to the pipeline
nlp.add_pipe('spacytextblob')

# ==================================== Loading the data ==================================

# Load the dataset
df = pd.read_csv('amazon_product_reviews.csv')
#print(df) # Uncomment to check whether the data has been read in correctly

# ================================ Preprocessing the data =================================

# Remove missing values
clean_df = df.dropna(subset=['reviews.text'])

# Put all words in lowercase and remove trailing whitespaces
clean_df['cleaned_reviews'] = clean_df['reviews.text'].apply(lambda x: str(x).lower().strip())
#print(clean_df)

# ================================== Checking similarity ==================================

# Select the first two reviews to check their similarity
review1 = clean_df['cleaned_reviews'][0]
review2 = clean_df['cleaned_reviews'][1]

# Tokenize the reviews
doc1 = nlp(review1)
doc2 = nlp(review2)

# Calculate the similarity score
similarity_score = doc1.similarity(doc2)

# Print the reviews and their similarity score
print(f"Review 1: {review1}")
print(f"Review 2: {review2}")
print(f"Similarity Score: {similarity_score:.2f}")

# =================================== Creating a function ==================================

# Define a function to perform the sentiment analysis
def analyse_sentiment(review):
    # Tokenize the review
    doc = nlp(review)

    # Retrieve the sentiment and polarity scores
    sentiment = doc._.blob.sentiment
    polarity = doc._.blob.polarity

    return sentiment, polarity

# =================================== Testing the model =====================================

# Test the model on some positive sample reviews
positive_reviews = [
    "I absolutely love this product! The design is sleek, and it adds a touch of sophistication to my workspace.",
    "This is the best purchase I've made in a long time. The features are outstanding, and it has exceeded my expectations.",
    "Incredible value for the price! The performance is outstanding, and it has made my daily tasks much easier.",
    "I can't express how happy I am with this product. The customer service is excellent, and the quality is top-notch.",
    "What a fantastic buy! The packaging was eco-friendly, and the setup was a breeze. Highly recommended!"
]

for review in positive_reviews:
    sentiment, polarity = analyse_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print("\n")

# Test the model on some neutral reviews
neutral_reviews = [
    "This product meets the basic requirements but lacks some advanced features.",
    "Neutral experience with this product. It does what it's supposed to do without any standout features.",
    "I neither love it nor hate it. It serves its purpose adequately, but I wouldn't go out of my way to recommend it.",
    "This is an average product. It meets basic requirements but lacks some advanced features.",
    "The product is okay. It doesn't stand out, but it gets the job done."
]

for review in neutral_reviews:
    sentiment, polarity = analyse_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print("\n")

# Test the model on some negative reviews
negative_reviews = [
    "I'm not impressed with this product. The quality is subpar, and it failed to meet my expectations.",
    "Regret purchasing this product. It doesn't live up to the advertised features, and the performance is lacking.",
    "Disappointed with the overall experience. The user interface is confusing, and the product didn't meet my needs.",
    "This product is a letdown. The build quality is poor, and it doesn't perform as expected.",
    "I wish I had chosen a different product. This one falls short in terms of quality and functionality."
]

for review in negative_reviews:
    sentiment, polarity = analyse_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print("\n")

# =============================== Checking for different products ===========================

# Before we analyse the dataset, we'll need to restrict it
# because there are too many reviews (4999),
# so it would take too long to perform the analysis on all of them.

# Check unique product ID values
unique_product_ids = clean_df['id'].unique()

# Display the number of unique product IDs
print(f"Number of unique product IDs: {len(unique_product_ids)}")
print("Unique product IDs:")
print(unique_product_ids)

# ================== Performing the analysis on different products separately ===================

# Separate the dataset based on different product IDs
for product_id in unique_product_ids:
    product_reviews = clean_df[clean_df['id'] == product_id]['cleaned_reviews']
    product_name = clean_df[clean_df['id'] == product_id]['name'].iloc[0]
    
    sentiments = []
    polarities = []
    for review in product_reviews:
        
        # Tokenize the review
        doc = nlp(review)
        
        # Remove stopwords
        cleaned_review = ' '.join([token.text for token in doc if not token.is_stop])
        
        # Perform sentiment analysis
        sentiment, polarity = analyse_sentiment(cleaned_review)
        sentiments.append(sentiment)
        polarities.append(polarity)
        #print(polarities)
        #print(sentiments) # If you want to have a look at all the numbers

    # Calculate the mean polarity to gauge the average sentiment
    if len(polarities) > 0:
        mean_polarity = sum(polarities) / len(polarities)
    else:
        mean_polarity = 0

    print("\n===================================================\n")
    print(f"The mean polarity is {mean_polarity:.2f}.")
    if mean_polarity > 0:
        print(f"The reviews for {product_name} are positive.")
    elif mean_polarity < 0:
        print(f"The reviews for {product_name} are negative.")
    else:
        print(f"The reviews for {product_name} are neutral.")

# I haven't used the sentiment output of the function,
# but it's there in case you want to have a look at it.