"""
Flask app for EverythingHoops API
"""

from tricktracker_api import TrickTrackerAPI
from flask import Flask, request, jsonify

app = Flask(__name__)

# create TrickTrackerAPI object
trick_api = TrickTrackerAPI("../Ollie108.mov")


@app.route("/")
def home():
    """
    Home page
    """
    return "Welcome to TrickTracker!"


@app.route("/trick", methods=['GET'])
def trick():
    """
    Trick prediction page
    """

    # get prediction
    prediction = trick_api.predict()

    # return jsonified statline
    response = jsonify(prediction)
    return response


if __name__ == "__main__":
    app.run(debug=True, port=8000)