"""
Script to split the dataset into train and test set
"""

# import libraries
import os
import random

BASE_PATH = 'backend/tricks'

# set the random seed
random.seed(123)

# initialize list of files_list
files_list = list()

# iterate over the directory structure
for root, dirs, files in os.walk(BASE_PATH, topdown=True):

    # iterate over the files
    for name in files:

        # check if the file is a .mov file
        if name.endswith(".mov"):

            # initialize the files_list list if empty
            files_list = list() if files_list == None else files_list

            # get the path of the file
            path = os.path.join(root,name).replace("\\", '/')
            path = path.strip(BASE_PATH + "/")
            files_list.append(path)

# shuffle the files_list
random.shuffle(files_list)

# split the files_list into train and test set
train_list = len(files_list)*0.80
train_list = int(train_list)

# write the train file
with open("train.txt","w") as f: 
    for i in range(0,train_list):
        f.write(files_list[i]+"\n")
        
# write the test file
with open("test.txt","w") as f:
    for j in range(train_list+1, len(files_list)):
        f.write(files_list[j]+"\n")