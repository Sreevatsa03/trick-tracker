import tensorflow as tf
import cv2
import numpy as np
import os
import tqdm
from sklearn.preprocessing import LabelBinarizer

VIDEO_PATH = '../Ollie108.mov'
SEQUENCE_LENGTH = 40
LABELS = ['Ollie','Kickflip','Shuvit'] 

encoder = LabelBinarizer()
encoder.fit(LABELS)

# load model from h5
model = tf.keras.models.load_model('trick_tracker.h5')

# summarize model
model.summary()

def frame_generator():
    """
    Generator that yields a tuple of (image, label) for a given video
    
    :yield: (image, label) tuple
    :rtype: tuple
    """
    
    # open the video and get the number of frames
    cap = cv2.VideoCapture(VIDEO_PATH)
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
            yield img, VIDEO_PATH

        # break if we have enough images
        if max_images == 0:
            break
            
        # increment counter
        current_frame += 1

# create a dataset from our generator
dataset = tf.data.Dataset.from_generator(frame_generator,
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
current_path = VIDEO_PATH
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

def make_generator():
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
        full_path = os.path.join(VIDEO_PATH).replace('.mov', '.npy')
        label = os.path.basename(os.path.dirname(VIDEO_PATH))

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

# create prediction dataset from generator
prediction_dataset = tf.data.Dataset.from_generator(make_generator(),
                    output_types=(tf.float32, tf.int16),
                    output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS)))) 

# batch the dataset and prefetch
prediction_dataset = prediction_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# make predictions on the prediction dataset
predictions = model.predict(prediction_dataset)

# get the index of the max prediction
max_index = np.argmax(predictions[0])

# map labels to predictions
print(LABELS[max_index])