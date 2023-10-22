from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import time

model = YOLO("yolov8n-seg.pt")

start = time.time()
results = model.predict(source="Ollie13.mov", show=True)
end = time.time()

t = end - start

# find fps of video
cap = cv2.VideoCapture("Ollie13.mov")
relative_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) / t
fps = cap.get(cv2.CAP_PROP_FPS)

time_ratio = fps / relative_fps

# result = results[0]

skateboard = np.array([])

for result in results:
    for b in result.boxes:

        # if object is skateboard add to numpy array
        if result.names[int(b.cls)] == "skateboard":
            # print("Skateboard coordinates:", b.xyxy)
            skateboard = np.append(skateboard, b.xyxy)

# print(skateboard)

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

# get the x and y coordinates
x = np.array(range(len(skateboard_centers[:, 0])))
y = skateboard_centers[:, 1]

# x_data = np.linspace(x.min(), x.max(), 100)
# y_data = y

# Smooth the data using a moving average
window_size = 3  # Can be adjusted based on the amount of smoothing you want
y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

# Calculate the derivative
dy = np.diff(y_smooth)

# Identify breakpoints where the derivative changes sign
breakpoints = np.where(np.diff(np.sign(dy)))[0] + window_size // 2

# find largest gap between consecutive breakpoints
largest_gap = 0
largest_idx = 0
largest_idx_2 = 0

for i in range(len(breakpoints) - 1):
    gap = breakpoints[i + 1] - breakpoints[i]
    if gap > largest_gap:
        largest_idx = breakpoints[i]
        largest_idx_2 = breakpoints[i + 1]
        largest_gap = gap

print("Largest gap:", largest_gap)

print("FPS:", fps)

# calculate time between breakpoints in seconds
time_between = largest_gap / fps

# calculate final height
final_y = 4.9 * (time_between ** 2)

print("Final height:", final_y)

plt.scatter(x, y)
for bp in breakpoints:
    plt.axvline(x=x[bp], color='red', linestyle='--')

plt.show()

# plot the parabola
# plt.plot(x, y, '-', color='red')
# plt.show()
