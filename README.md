# Final Report
- https://github.com/danlee0528/Two-Sigma-Connect-Rental-Listing-Inquiries/blob/master/Project%20Final%20Report.pdf

# Dataset
- https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/overview

## Dataset Description
Predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City.

The target variable, interest_level, is defined by the number of inquiries a listing has in the duration that the listing was live on the site. 

- train.json - the training set
- test.json - the test set
- sample_submission.csv - a sample submission file in the correct format
- images_sample.zip - listing images organized by listing_id (a sample of 100 listings)
- Kaggle-renthop.7z - (optional) listing images organized by listing_id. Total size: 78.5GB compressed. Distributed by BitTorrent (Kaggle-renthop.torrent). 

### Data fields
- bathrooms: number of bathrooms
- bedrooms: number of bathrooms
- building_id
- created
- description
- display_address
- features: a list of features about this apartment
- latitude
- listing_id
- longitude
- manager_id
- photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
- price: in USD
- street_address
- interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'


# Project Description
## Phase1. Exploratory data analysis and data pre-processing
- Perform the initial analysis and exploration on the dataset to summarize its main characteristics. This step is a
great practice to see what the data can tell you beyond the formal modelling or hypothesis
testing task, like discovering the potential patterns, spotting outliers and so on. To this aim, apply any meaningful visualization methods and statistical tests.

- In addition, in this phase, perform data pre-processing, which is the practice of
detecting and correcting corrupt or inaccurate records from the dataset, by identifying
incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying,
or deleting the dirty or coarse data.

- Finally, extract features from the unstructured text and images associated with the
dataset, optionally use traditional feature extraction methods. Neural network-based methods that have recently become very
popular for processing natural language and images are not allowed in this project.

## Phase 2. Training Models
Train on the data that has been preprocessed in milestone 1. Choose among the following three classifiers:
1. Decision Tree
2. Logistic Regression
3. SVM

## Phase 3. Advanced Models
Develop more advanced classifiers of own choice. The only restriction is that only the classifiers used in milestone 2 should be engineered.
