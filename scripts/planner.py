import  numpy as np

from scripts.hmm import hmm
from scripts.movenet import movenet
from scripts.polar_map import polarMap

class planner():

    def __init__(
        self,
        goal_angle : float = 0.0,
        init_angle : float = 0.0,
    ) -> None:
        
        self.angle = init_angle
        self.goal_angle = goal_angle
        self._step = 0

        self.hmm = hmm()
        self.map = polarMap()
        self.movenet = movenet()

    def plan(
            self,
    ) -> None:
        '''
        function for computing desired control imput, and tracking expected angle of turtlebot
        '''

        angle_change = self.goal_angle - self.angle
        turtlebot_angle_change = self.vfh(target_angle=angle_change)

        '''
        ENTER CODE HERE
        execute change in turtle bot angle and move forward, you must then
        update self.angle to the new angle of the turtle bot
        '''

        return turtlebot_angle_change

    def step(
            self,
    ) -> None:
        '''
        step function for turtle bot, handles all necissary functions to 
        sense the human, predict poses/behavior, map obstacles and take a 
        forward step
        '''
        
        '''
        ENTER CODE HERE
        load image/save image, self.movenet.predict can take either a string of 
        and image file location, or a loaded jpeg image using cv2. set image to 
        one of these two types of varaibles
        '''

        image = 'misc/frame_00463.jpeg' #overwrite

        #take image, compute pose, and predict behavior
        pose = self.movenet.predict(image)
        prediction = self.hmm.predict(pose)

        '''
        ENTER CODE HERE
        need to compute expected distance of human using turtlebot lidar, and overwrite
        assumed distance in self.map.map_behavior (currently set to 2 ft).
        '''

        #load predicted behavior into the map assuming distance of 2ft and directly ahead
        self.map.map_behavior(prediction,distance=5,angle=0)

        #compute needed change in robot angle and give move command
        turtlebot_angle_change = self.plan()

        #propagate map by 1 timestep (previous probabilities slowly go to 0)
        self.map.map_decay()

        return turtlebot_angle_change


    def vfh(
            self,
            target_angle : float,
    ):
        '''
        implementation of Vetor Feild Histogram method, takes target angle
        assuming the probability map is fixed to the robot, and computs the
        desired change in angle

        input
        -----
        target_angle:float
            the desired change in angle for the robot

        output
        ------
        list[list[float]]
            desired point on map to travel to, if map has multiple bins will
            be a list of points. points have the format [theta, distance]
        '''

        path = []
        for i in range(self.map.depth_bins):
            angle_dif = abs(self.map.theta_points - target_angle)
            sorted_indexes = np.argsort(angle_dif)

            for idx in sorted_indexes:
                if self.map.map[idx,i] == 0:
                    path.append([self.map.theta_points[idx],self.map.depth_points[i]])
                    break

        return path