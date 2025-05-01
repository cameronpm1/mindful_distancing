import cv2
import numpy as np

from scripts.polar_map import polarMap
from scripts.hmm import hmm
from scripts.movenet import movenet



hmm1 = hmm()
movenet1 = movenet()

pose = movenet1.predict('misc/frame_00463.jpeg')
prediction = hmm1.predict(pose)
map1 = polarMap(theta_bins=10)

map1.map_behavior('crouch',distance=0,angle=0)
map1.map_decay()
print(map1.map)
map1.map_decay()
print(map1.map)