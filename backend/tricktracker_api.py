import tensorflow as tf
import cv2
import numpy as np
import os
import tqdm
from sklearn.preprocessing import LabelBinarizer
from ultralytics import YOLO

SEQUENCE_LENGTH = 40

# labels for the tricks
LABELS = ['Ollie','Kickflip','Shuvit'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

class TrickTrackerAPI:
    """
    TrickTracker API
    """

    def __init__(self, video_path):
        """
        Load the model
        """

        # load model from h5
        self.model = tf.keras.models.load_model('trick_tracker.h5')
        
        # set the video path
        # global VIDEO_PATH
        self.video_path = video_path

    def _frame_generator(self):
        """
        Generator that yields a tuple of (image, label) for a given video
        
        :yield: (image, label) tuple
        :rtype: tuple
        """
        
        # open the video and get the number of frames
        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # sample frames based on SEQUENCE_LENGTH
        sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
        current_frame = 0

        # loop through each frame where we have a sample
        max_images = SEQUENCE_LENGTH
        while True:

            # read the frame from the video
            success, frame = cap.read()

            # break if we are no longer reading a video
            if not success:
                break
            
            # only use every sample_every_frame frame
            if current_frame % sample_every_frame == 0:
                
                # convert to RGB (opencv uses BGR)
                frame = frame[:, :, ::-1]

                # resize to 299x299 for inceptionv3
                img = tf.image.resize(frame, (299, 299))

                # preprocess using the inception_v3 preprocess_input function
                img = tf.keras.applications.inception_v3.preprocess_input(img)
                max_images -= 1

                # get preprocessed image and video path
                yield img, self.video_path

            # break if we have enough images
            if max_images == 0:
                break
                
            # increment counter
            current_frame += 1

    def _make_generator(self):
        """
        Return generator function that yields (padded_sequence, label) tuples
        """

        def generator():
            """
            Generator function that yields (padded_sequence, label) tuples

            :yield: (padded_sequence, label) tuple
            :rtype: tuple
            """

            # get the full path and label
            full_path = os.path.join(self.video_path).replace('.mov', '.npy')
            label = os.path.basename(os.path.dirname(self.video_path))

            # load the features
            features = np.load(full_path)

            # pad the sequence if necessary
            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            # transform the label
            transformed_label = encoder.transform([label])

            # yield the padded sequence and label
            yield padded_sequence, transformed_label[0]

        # return the generator    
        return generator
    
    def classify(self):
        """
        Make a trick prediction on a video
        """

        # create a dataset from our generator
        dataset = tf.data.Dataset.from_generator(self._frame_generator,
                    output_types=(tf.float32, tf.string),
                    output_shapes=((299, 299, 3), ()))

        # batch the dataset and prefetch
        dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

        # load the inception_v3 model
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        # output tensor from inception_v3 to add layers
        x = inception_v3.output

        # add a global spatial average pooling layer
        pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

        # model with input from inception_v3 and pooling layer
        feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)

        # initialize current_path and all_features
        current_path = self.video_path
        counter = 0
        all_features = []

        # iterate over the dataset
        for img, batch_paths in tqdm.tqdm(dataset):

            # get the features for the images using the inception_v3 model
            batch_features = feature_extraction_model(img)

            # reshape the tensor
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1))
            
            # iterate over each feature and path
            for features, path in zip(batch_features.numpy(), batch_paths.numpy()):

                # check if we are done with a video
                if counter == batch_features.shape[0] - 1:

                    # save the features as a numpy array and clear the all_features list
                    output_path = current_path.replace('.mov', '.npy')
                    np.save(output_path, all_features)
                    all_features = []
                
                # update the current path and counter
                current_path = path.decode()
                counter += 1

                # append the features to the all_features list
                all_features.append(features)

        # create prediction dataset from generator
        prediction_dataset = tf.data.Dataset.from_generator(self._make_generator(),
                            output_types=(tf.float32, tf.int16),
                            output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS)))) 

        # batch the dataset and prefetch
        prediction_dataset = prediction_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

        # make predictions on the prediction dataset
        predictions = self.model.predict(prediction_dataset)

        # get the index of the max prediction
        max_index = np.argmax(predictions[0])

        # return the prediction
        return {"Prediction": LABELS[max_index], "Accuracy": str(predictions[0][max_index])}
    
    def predict_height(self):

        # initialize YOLO model
        model = YOLO("yolov8n-seg.pt")

        # model predict
        results = model.predict(source=self.video_path, show=True)

        # find fps of video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # initialize numpy array
        skateboard = np.array([])
        is_skateboard = True

        # loop through results
        for result in results:
            for b in result.boxes:

                # if object is skateboard add to numpy array
                if result.names[int(b.cls)] == "skateboard":
                    skateboard = np.append(skateboard, b.xyxy)

        # if no skateboard is detected return 0
        if len(skateboard) == 0:
            return {"is_skateboard": False, "height": 0}

        # reshape the numpy array
        skateboard = np.reshape(skateboard, (-1, 4))

        # function to find center of bounding box
        def find_center(box):
            # get the x and y coordinates
            x1, y1, x2, y2 = box

            # find the center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # return the center
            return x_center, y_center

        # find the center of each bounding box
        skateboard_centers = np.apply_along_axis(find_center, 1, skateboard)

        # get the y coordinates
        y = skateboard_centers[:, 1]

        # Smooth the data using a moving average
        window_size = 3  # Can be adjusted based on the amount of smoothing you want
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

        # Calculate the derivative
        dy = np.diff(y_smooth)

        # Identify breakpoints where the derivative changes sign
        breakpoints = np.where(np.diff(np.sign(dy)))[0] + window_size // 2

        # find largest gap between consecutive breakpoints
        largest_gap = 0

        for i in range(len(breakpoints) - 1):
            gap = breakpoints[i + 1] - breakpoints[i]
            if gap > largest_gap:
                largest_gap = gap

        # calculate time between breakpoints in seconds
        time_between = largest_gap / fps

        # calculate final height
        final_y = 4.9 * (time_between ** 2)

        # return the final height
        return {"is_skateboard": is_skateboard, "height": final_y}
    
def main():
    # initialize the api
    trick_tracker = TrickTrackerAPI('../Ollie108.mov')

    # make a prediction
    classification = trick_tracker.classify()

    # predict height
    prediction = trick_tracker.predict_height()

    # print the prediction
    print(classification)
    print(prediction)
    
if __name__ == '__main__':
    main()