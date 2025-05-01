import cv2
import numpy as np

from scripts.polar_map import polarMap
from scripts.hmm import hmm
from scripts.movenet import movenet


#initialize hmm, movenet, and probability map classes
hmm1 = hmm()
movenet1 = movenet()
map1 = polarMap()

#load a test image and get hmm prediction
pose = movenet1.predict('misc/frame_00463.jpeg')
prediction = hmm1.predict(pose)

#load predicted behavior into the map assuming distance of 2ft and directly ahead
map1.map_behavior(prediction,distance=2,angle=0)
#propagate map by 1 timestep (previous probabilities slowly go to 0)
map1.map_decay()
