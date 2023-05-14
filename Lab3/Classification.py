import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Read features from the .pkl file
features_from_pkl_file = pd.read_pickle('results.pkl')

# Define the properties to extract from the file
properties_from_file = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

# Extract unique iter_texture categories
T_images = features_from_pkl_file['category'].unique()

# Create empty lists for features and labels
features = []
labels = []

# Iterate over each iter_texture category
for iter_texture in T_images:
    # Filter the data for the current iter_texture category
    texture_data = features_from_pkl_file[features_from_pkl_file['category'] == iter_texture]

    # Extract the properties for the current iter_texture category
    properties = texture_data[properties_from_file].values

    # Flatten and add the properties to the features list
    features.extend([np.hstack(prop).flatten() for prop in properties])

    # Add labels for each sample in the current iter_texture category
    labels.extend([iter_texture] * len(properties))

# Split the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels)

# Create an SVM classifier
classifier = svm.SVC(kernel='linear')

# Train the model using the training sets
classifier.fit(x_train, y_train)

# Predict the response for the test dataset
predict_response = classifier.predict(x_test)

# Calculate the model accuracy
accuracy = metrics.accuracy_score(y_test, predict_response) * 100
print('Accuracy:', round(accuracy, 2), '%')
