import numpy as np
from scipy.stats import multivariate_normal

behavior_distributions = {
    'crouch' : {
        'means' : [[0.0,0.0]],
        'stds' : [[3.0,3.0]]
    },
    'walk_left' : {
        'means' : [[-1.0,0.0],[-2.0,0.0]],
        'stds' : [[1.0,1.0],[2.0,2.0]]
    },
    'walk_right' : {
        'means' : [[1.0,0.0],[2.0,0.0]],
        'stds' : [[1.0,1.0],[2.0,2.0]]
    },
    'clear' : {
        'means' : [],
        'stds' : []
    }
}

class polarMap():

    def __init__(
            self,
            theta_bins : int = 50,
            depth_bins : int = 1,
            depth_range : float = 5,
    ):
        '''
        input
        -----
        theta_bins:int
            number of angle bins in mape
        depth_bins:int
            number of depth bins in map
        depth_range:float
            depth range of the map
        '''
        
        self.depth_range = depth_range

        self.map = np.zeros((theta_bins,depth_bins))
        self.theta_points = np.linspace(0,np.pi*2*(theta_bins-1)/theta_bins,theta_bins) + np.pi*2/(theta_bins*2)
        self.depth_points = np.linspace(0,depth_range*(depth_bins-1)/depth_bins,depth_bins) + depth_range/(depth_bins*2)


    def map_decay(
            self,
            decay_factor : float = 0.5,
            decay_tol : float = 0.005,
    ) -> None:
        '''
        decay map weighting each timestep, set a values belowe decay_tol to 0

        input
        -----
        decay_factor:float
            value to multiple all map weights by
        decay_tol:float
            tolerance for setting weights to 0
        '''

        self.map = self.map*decay_factor
        self.map[self.map < decay_tol] = 0

    def map_behavior(
            self,
            behavior : str,
            distance : float,
            angle : float,
    ) -> None:
        '''
        takes string of behavior, and average location of human, and adds
        predefined gaussian distribution to map

        input
        -----
        behavior:str
            the behavior to be mapped (crouch, walk left, walk right, clear)
        distance:float
            distance from obstacle
        angle:float
            angle from obstacle (head on is 0 degrees, clockwise is negative)
        '''

        theta = np.abs(self.theta_points - angle).argmin()
        depth_point = np.abs(self.depth_points - distance).argmin()
        obs_pos = np.array([np.cos(angle),np.sin(angle)]) * distance

        #update map to current timestep
        self.map_decay()

        #update map with identified behavioral distribution
        bd = behavior_distributions[behavior]
        for i,theta in enumerate(self.theta_points):
            for j,depth in enumerate(self.depth_points):
                for k,dist in enumerate(bd['means']):
                    point = np.array([np.cos(theta),np.sin(theta)]) * depth
                    self.map[i,j] += multivariate_normal.pdf(point - obs_pos,dist,bd['stds'][k])

    def get_map(self):
        return self.map