"""
Flask app for EverythingHoops API
"""

from tricktracker_api import TrickTrackerAPI
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def home():
    """
    Home page
    """
    return "Welcome to TrickTracker!"

@app.route("/trick", methods=['POST','GET'])
def trick():
    """
    Trick prediction page
    """

    if request.method == "POST":
        # get the file
        video = request.files["video"]

        # save the file
        video.save("../webcam_capture.mov")

    # create TrickTrackerAPI object
    trick_api = TrickTrackerAPI("../webcam_capture.mov")

    # get prediction
    classification = trick_api.classify()

    # predict height
    # prediction = trick_api.predict_height()

    # append classification dict to prediction dict
    # prediction.update(classification)

    print(classification)

    # return jsonified statline 
    response = jsonify(classification)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(debug=True, port=8000)