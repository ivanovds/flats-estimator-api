import os
import pickle
import pandas as pd
from geopy.distance import geodesic
import mpmath as math

metro_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../static/data/kyiv_metro_stations.csv')
metro_stations = pd.read_csv(metro_path)

rf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../static/data/rf_model.sav')
loaded_rf_model = pickle.load(open(rf_path, 'rb'))
xgb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../static/data/xgb_model.sav')
loaded_xgb_model = pickle.load(open(xgb_path, 'rb'))


def get_azimuth(latitude, longitude):
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


def predict_price(data):
    flat = pd.DataFrame({
        'floor': [data['floor']],
        'floors_count': [data['floors_count']],
        'total_square_meters': [data['total_square_meters']],
        'center_distance': [data['center_distance']],
        'metro_distance': [data['metro_distance']],
        'azimuth': [data['azimuth']],
    })

    rf_prediction_price_metr = loaded_rf_model.predict(flat).round(2)
    xgb_prediction_price_metr = loaded_xgb_model.predict(flat).round(2)

    rf_total_price = rf_prediction_price_metr * flat['total_square_meters'][0]
    xgb_total_price = xgb_prediction_price_metr * flat['total_square_meters'][0]

    avg_total_price = (rf_total_price * 0.5 + xgb_total_price * 0.5)

    return int(avg_total_price)

