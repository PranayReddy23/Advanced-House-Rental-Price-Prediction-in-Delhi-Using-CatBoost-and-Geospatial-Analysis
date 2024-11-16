from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import pickle

app = Flask(__name__)

# Load models and encoders once
with open('kmeans_model.pkl', 'rb') as file:
    loaded_kmeans = pickle.load(file)

with open('onehot_encoder_cat.pkl', 'rb') as file:
    loaded_encoder = pickle.load(file)

with open('std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('cat_model.pkl', 'rb') as file:
    cat_model = pickle.load(file)

# Initialize Nominatim API once
geolocator = Nominatim(user_agent="geoapi")

# Load the house dataset only if needed
house_data = pd.read_csv('Indian_housing_Delhi_data.csv')  # Assuming house_data.csv is the main dataset
location_means = house_data.groupby('location')['price'].mean()
global_mean = house_data['price'].mean()
location_ranks = house_data.groupby('location')['price'].mean().rank().to_dict()

def get_location_features(location):
    city_center = (28.6139, 77.2090)  # New Delhi center coordinates
    location_geocode = geolocator.geocode(location)
    
    if location_geocode:
        location_coords = (location_geocode.latitude, location_geocode.longitude)
        distance = geodesic(location_coords, city_center).km
        cluster = loaded_kmeans.predict([[location_coords[0], location_coords[1]]])[0]
    else:
        distance = np.nan
        cluster = np.nan
    
    mean_encoded = location_means.get(location, global_mean)
    rank_encoded = location_ranks.get(location, np.nan)
    
    return distance, cluster, mean_encoded, rank_encoded

@app.route('/')
def home_page():
    return render_template('house_rentals.html')

@app.route('/', methods=['POST'])
def predict():
    # Get form data
    form_data = {
        'location': request.form.get('location'),
        'property_type': request.form.get('property_type'),
        'bhk': request.form.get('bhk'),
        'house_size': request.form.get('house_size'),
        'status': request.form.get('status'),
        'deposit_amount': request.form.get('deposit_amount'),
        'no_of_balconies': request.form.get('no_of_balconies'),
        'no_of_bathrooms': request.form.get('no_of_bathrooms')
    }
    
    # Prepare new data
    new_data = pd.DataFrame({
        'numBathrooms': form_data['no_of_bathrooms'],
        'numBalconies': form_data['no_of_balconies'],
        'SecurityDeposit': form_data['deposit_amount'],
        'Status': form_data['status'],
        'house_size(sq_ft)': form_data['house_size'],
        'BHK': form_data['bhk'],
        'property_type': form_data['property_type']
    }, index=[0])
    
    if form_data['location']:
        distance, cluster, mean_encoded, rank_encoded = get_location_features(form_data['location'])
        new_data['distance_from_center'] = distance
        new_data['location_cluster'] = cluster
        new_data['location_mean_encoded'] = mean_encoded
        new_data['location_rank_encoded'] = rank_encoded
    else:
        new_data[['distance_from_center', 'location_cluster', 'location_mean_encoded', 'location_rank_encoded']] = np.nan

    # Encode categorical and scale numerical features
    new_data_cat = new_data[['Status', 'property_type']]
    new_data_cat_encoded = pd.DataFrame(loaded_encoder.transform(new_data_cat), columns=loaded_encoder.get_feature_names_out())

    new_data_num = new_data.drop(['Status', 'property_type'], axis=1)
    new_data_num_encoded = pd.DataFrame(scaler.transform(new_data_num), columns=scaler.get_feature_names_out())
    
    # Combine encoded and scaled features
    new_data_encoded = pd.concat([new_data_num_encoded, new_data_cat_encoded], axis=1)
    
    # Predict using the loaded CatBoost model
    result = cat_model.predict(new_data_encoded)[0]

    # Assuming the prediction is a log-transformed value
    final_result = np.exp(result)  # Remove int casting to preserve accuracy

    # Render the result on result.html template
    return render_template('house_rentals.html', final_result=f"{final_result:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
