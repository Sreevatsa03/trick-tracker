"""
Script to split the dataset into train and test set
"""

# import libraries
import os
import random

BASE_PATH = 'tricks'

# set the random seed
random.seed(123)

# initialize list of files
files = list()

# iterate over the directory structure
for root, dirs, files in os.walk(BASE_PATH, topdown=True):

    # iterate over the files
    for name in files:

        # check if the file is a .mov file
        if name.endswith(".mov"):

            # initialize the files list if empty
            files = list() if files is None else files

            # get the path of the file
            path = os.path.join(root,name).replace("\\",'/')
            path = path.strip("tricks/")
            files.append(path)

# shuffle the files
random.shuffle(files)

# split the files into train and test set
train_list = len(files)*0.80
train_list = int(train_list)

# write the train file
with open("train.txt","w") as f: 
    for i in range(0,train_list):
        f.write(files[i]+"\n")
        
# write the test file
with open("test.txt","w") as f:
    for j in range(train_list+1, len(files)):
        f.write(files[j]+"\n")