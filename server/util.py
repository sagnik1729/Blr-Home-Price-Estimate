import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None
def get_estimated_price(location,sqft,bhk,bath,balcony ):
    try:
        loc_index = __data_columns.index(location)
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3]=sqft
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def get_loc_name():
    return __locations


def get_data_columns():
    return __data_columns


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]  # first 3 columns are "bath", "balcony", "BHK", "final_area",
    global __model
    if __model is None:
        with open('./artifacts/blr_home_price_est.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

if __name__ == "__main__":
    load_saved_artifacts()
    # print(get_loc_name())
    print(get_estimated_price('location_1st Phase JP Nagar', 1000, 3, 3,0))