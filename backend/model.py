"""
Train and validate TensorFlow model to recognize skateboarding tricks
"""

# import libraries
import tensorflow as tf
import os
import cv2
import numpy as np
import tqdm
from sklearn.preprocessing import LabelBinarizer

BASE_PATH = 'Tricks'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mov')
SEQUENCE_LENGTH = 40

# labels for the tricks
LABELS = ['Ollie','Kickflip','Shuvit'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

def frame_generator():
    """
    Generator that yields a tuple of (image, label) for videos in the tricks directory

    :yield: (image, label) tuple
    :rtype: tuple
    """

    # get all the video video_paths and shuffle
    video_paths = tf.io.gfile.glob(VIDEOS_PATH)
    np.random.shuffle(video_paths)

    # iterate over each video
    for video in video_paths:

        # open the video and get the number of frames
        cap = cv2.VideoCapture(video)
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
                yield img, video

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
current_path = None
all_features = []

# iterate over the dataset
for img, batch_paths in tqdm.tqdm(dataset):

    # get the features for the images using the inception_v3 model
    batch_features = feature_extraction_model(img)

    # reshape the tensor
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1))
    
    # iterate over each feature and path
    for features, path in zip(batch_features.numpy(), batch_paths.numpy()):

        # check if we are working on a new video
        if path != current_path and current_path is not None:

            # save the features as a numpy array and clear the all_features list
            output_path = current_path.decode().replace('.mov', '.npy')
            np.save(output_path, all_features)
            all_features = []
        
        # update the current path
        current_path = path

        # append the features to the all_features list
        all_features.append(features)

# model with layers:
# 1. Masking to skip timestep of 0
# 2. LSTM with 512 units
# 3. Dense layer with 256 units and relu activation
# 4. Dropout layer to randomly set 50% of the units to 0
# 5. Output layer with softmax activation to output probability distribution over the 3 classes
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

# compile the model with Adam optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# load the test list
with open('test.txt') as f:
    test_list = [row.strip() for row in list(f)]

# load the train list
with open('train.txt') as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]


def make_generator(file_list):
    """
    Return generator function that yields (padded_sequence, label) tuples

    :param file_list: list of files to generate from
    :type file_list: list
    """

    def generator():
        """
        Generator function that yields (padded_sequence, label) tuples

        :yield: (padded_sequence, label) tuple
        :rtype: tuple
        """

        # shuffle the file list
        np.random.shuffle(file_list)

        # iterate over each file
        for path in file_list:

            # get the full path and label
            full_path = os.path.join(BASE_PATH + '/', path).replace('.mov', '.npy')
            label = os.path.basename(os.path.dirname(path))

            # load the features
            features = np.load(full_path)

            # pad the features if necessary
            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            # transform the label with encoder
            transformed_label = encoder.transform([label])

            # yield the padded sequence and transformed label
            yield padded_sequence, transformed_label[0]
    
    # return the generator
    return generator


print("Training on: " + str(len(train_list)) + " samples")

# create a train dataset from our generator
train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                                               output_types=(tf.float32, tf.int16),
                                               output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))

# batch the dataset and prefetch
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

print("Validating on: " + str(len(test_list)) + " samples")

# create a validation dataset from our generator
valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))

# batch the dataset and prefetch
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# tensorboard callback to log metrics
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)

# fit the model on the train and validation dataset
model.fit(train_dataset, epochs=17, callbacks=[tensorboard_callback], validation_data=valid_dataset)

# save model to h5
model.save('trick_tracker.h5')