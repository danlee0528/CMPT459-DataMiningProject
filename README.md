# Objective
The main goal of this project is predicting how popular an apartment rental listing will be on the basis of data mentioned above. 
However, there are some intermediate stages which enhance your analytical skills and prepare you for the final goal.

# Dataset
- https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/overview

## Dataset Description
In this competition, you will predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City.

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

Like all the data mining and machine learning pipelines, we expect you to perform the initial
analysis and exploration on the dataset to summarize its main characteristics. This step is a
great practice to see what the data can tell you beyond the formal modelling or hypothesis
testing task, like discovering the potential patterns, spotting outliers and so on. To this aim, you
can apply any meaningful visualization methods and statistical tests. It is worth noting, that
you may incrementally improve your exploration section, as you work on the next steps.

In addition, in this phase, you need to perform data pre-processing, which is the practice of
detecting and correcting corrupt or inaccurate records from the dataset, by identifying
incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying,
or deleting the dirty or coarse data.

Finally, you need to extract features from the unstructured text and images associated with the
dataset. You can use traditional feature extraction methods such as those presented in class.
You are not expected to use neural network-based methods that have recently become very
popular for processing natural language and images.

# Submissions 
## 1. Exploratory data analysis (Use the training dataset (. . . train) to perform EDA)
  - [ ] Plot histograms for the following numeric columns: Price, Latitude & Longitude.
  - [ ] Plot hour-wise listing trend and find out the top 5 busiest hours of postings.
  - [ ] Visualization to show the proportion of target variable values.
## 2. Dealing with missing values, outliers
  - [ ] Find out the number of missing values in each variable.
  - [ ] Find out the number of outliers in each variable. Plot visualizations to demonstrate them. You can either remove the outliers or provide a short argument as to why outlier detection is not meaningful for that attribute.
  - [ ] Can we safely drop the missing values? If not, how will you deal with them?
## 3. Feature extraction from images and text
  - [ ] Extract features from the images and transform it into data that’s ready to be used in the model for classification.
  - [ ] Extract features from the text data and transform it into data that’s ready to be used in the model for classification.

