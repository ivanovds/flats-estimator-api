import xgboost as xgb
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import mpmath as math
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle


def get_azimuth(latitude, longitude):
    """Returns angle between north and city center."""
    city_center_coordinates = [50.450075, 30.524205]
    rad = 6372795

    llat1 = city_center_coordinates[0]
    llong1 = city_center_coordinates[1]
    llat2 = latitude
    llong2 = longitude

    lat1 = llat1 * math.pi / 180.
    lat2 = llat2 * math.pi / 180.
    long1 = llong1 * math.pi / 180.
    long2 = llong2 * math.pi / 180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.power(cl2 * sdelta, 2) + math.power(cl1 * sl2 - sl1 * cl2 * cdelta, 2))
    x = sl1 * sl2 + cl1 * cl2 * cdelta
    ad = math.atan2(y, x)

    x = (cl1 * sl2) - (sl1 * cl2 * cdelta)
    y = sdelta * cl2
    z = math.degrees(math.atan(-y / x))

    if (x < 0):
        z = z + 180.

    z2 = (z + 180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2 * math.pi) * math.floor((z2 / (2 * math.pi))))
    angledeg = (anglerad2 * 180.) / math.pi

    return round(angledeg, 2)


def get_metro_distance(latitude, longitude):
    """Returns distance to the nearest metro station."""
    flat_coordinates = [latitude, longitude]
    distances = list(map(lambda x, y: geodesic(flat_coordinates, [x, y]).meters,
                         metro_stations['latitude'], metro_stations['longitude']))

    metro_distance = min(distances)

    return int(metro_distance)


def get_center_distance(latitude, longitude):
    city_center_coordinates = [50.450075, 30.524205]
    center_distance = geodesic(city_center_coordinates, [latitude, longitude]).meters

    return int(center_distance)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


def print_metrics(prediction, val_y):
    val_mae = mean_absolute_error(val_y, prediction)
    median_AE = median_absolute_error(val_y, prediction)
    r2 = r2_score(val_y, prediction)

    print('')
    print('R\u00b2: {:.2}'.format(r2))
    print('')
    print('Середня абсолютна похибка: {:.3} %'.format(mean_absolute_percentage_error(val_y, prediction)))
    print('Медіанна абсолютна похибка: {:.3} %'.format(median_absolute_percentage_error(val_y, prediction)))


flats_path = 'kyiv_flats_dataset.csv'
df = pd.read_csv(flats_path)
df.head(5)
df.info()

metro_path = 'kyiv_metro_stations.csv'
metro_stations = pd.read_csv(metro_path)
metro_stations.info()
metro_stations.head(5)

flats_path = 'kyiv_flats_dataset.csv'
df = pd.read_csv(flats_path)

df['price_metr'] = df['price_total_usd'] / df['total_square_meters']

city_center_coordinates = [50.450075, 30.524205]
df['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), df['latitude'], df['longitude']))

df = df.loc[(df['center_distance'] < 15000)]
df['azimuth'] = df['azimuth'].round(0)

df.info()
df.head(5)
df.describe()

# Deleting of outliers
first_quartile = df.quantile(q=0.25)
third_quartile = df.quantile(q=0.75)
IQR = third_quartile - first_quartile
outliers = df[(df > (third_quartile + 1.5 * IQR)) | (df < (first_quartile - 1.5 * IQR))].count(axis=1)
outliers.sort_values(axis=0, ascending=False, inplace=True)

outliers = outliers.head(2000)
df.drop(outliers.index, inplace=True)

df.info()

# Assign the target variable
y = df['price_metr']

# List of features which we will use building the models
features = [
    'floor',
    'floors_count',
    'total_square_meters',
    'center_distance',
    'metro_distance',
    'azimuth',
]

X = df[features]

# Perform random sampling of data ( 0.75/0.25 )
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=300,
                                 n_jobs=-1,
                                 bootstrap=False,
                                 criterion='mse',
                                 max_features=3,
                                 random_state=1,
                                 max_depth=30,
                                 min_samples_split=5
                                 )

rf_model.fit(train_X, train_y)
rf_prediction = rf_model.predict(val_X).round(0)

print_metrics(rf_prediction, val_y)

# XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:gamma',
                             learning_rate=0.01,
                             max_depth=20,
                             n_estimators=2000,
                             nthread=-1,
                             eval_metric='gamma-nloglik',
                             )

xgb_model.fit(train_X, train_y)
xgb_prediction = xgb_model.predict(val_X).round(0)

print_metrics(xgb_prediction, val_y)

# Final model (combination of Random Forest and XGBoost)
prediction = rf_prediction * 0.5 + xgb_prediction * 0.5

print_metrics(prediction, val_y)

# Importance of attributes in the Random forest model
importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Рейтинг важливості ознак:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

plt.figure()
plt.title("Важливість ознак")
plt.bar(range(X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Evaluation of single apartments
# Appartments URL: https://dom.ria.com/ru/realty-prodaja-kvartira-kiev-predslavinskaya-ulitsa-16232758.html
# Total price: 320 000 USD.


flat = pd.DataFrame({
    'floor': [4],
    'floors_count': [7],
    'total_square_meters': [183],
    'latitude': [50.424018],
    'longitude': [30.520902]
})

flat['center_distance'] = get_center_distance(flat['latitude'][0], flat['longitude'][0])
flat['metro_distance'] = get_metro_distance(flat['latitude'][0], flat['longitude'][0])

flat['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), flat['latitude'], flat['longitude']))
flat['azimuth'] = flat['azimuth'].round(0)

flat = flat.drop('latitude', axis=1)
flat = flat.drop('longitude', axis=1)

rf_prediction_price_metr = rf_model.predict(flat).round(2)
xgb_prediction_price_metr = xgb_model.predict(flat).round(2)

rf_total_price = rf_prediction_price_metr * flat['total_square_meters'][0]
xgb_total_price = xgb_prediction_price_metr * flat['total_square_meters'][0]

avg_price_metr = rf_prediction_price_metr * 0.5 + xgb_prediction_price_metr * 0.5
avg_total_price = (rf_prediction_price_metr * 0.5 + xgb_prediction_price_metr * 0.5) * flat['total_square_meters'][0]

print(f'Ціна за метр, предсказанная моделью Random Forest: {int(rf_prediction_price_metr)} дол.')
print(f'Ціна за метр, передбачена моделюю XGBoost: {int(xgb_prediction_price_metr)} дол.')
print('Середня ціна за метр по двум моделям: {int(avg_price_metr)} дол.')

print(f'Ціна за всю квартиру, передбачена моделюю Random Forest: {int(rf_total_price)} дол.')
print(f'Ціна за всю квартиру, передбачена моделюю XGBoost: {int(xgb_total_price)} дол.')
print(f'Середня ціна за всю квартиру по двум моделям: {int(avg_total_price)} дол.')

# Save the model to disk with pickle
filename = 'rf_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))
filename = 'xgb_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))