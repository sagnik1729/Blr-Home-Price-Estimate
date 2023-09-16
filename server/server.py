# print('hello sagnik, we now going to build this')
from flask import Flask , request, jsonify
import util
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "HII"



@app.route('/get_loc_name', methods=['GET'])
def get_loc_name():
    # print(util.get_loc_name())
    locations = util.get_loc_name()
    response=jsonify({
        'locations':locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    balcony=int(request.form['balcony'])
    estimated_price=util.get_estimated_price(location, total_sqft, bhk, bath,balcony)
    response = jsonify({
        'estimated_price':estimated_price
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print('Starting the Flask server')
    util.load_saved_artifacts()

    app.run()



