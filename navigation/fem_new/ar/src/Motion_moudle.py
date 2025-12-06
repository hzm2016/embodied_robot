import numpy as np

class Motion:
    def __init__(self) :
        pass


    def Rotate(self, u, theta):
        return [[np.cos(theta) + u[0]**2 * (1-np.cos(theta)), 
             u[0] * u[1] * (1-np.cos(theta)) - u[2] * np.sin(-theta), 
             u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(-theta)],
            [u[0] * u[1] * (1-np.cos(theta)) + u[2] * np.sin(-theta),
             np.cos(theta) + u[1]**2 * (1-np.cos(theta)),
             u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(-theta)],
            [u[0] * u[2] * (1-np.cos(theta)) - u[1] * np.sin(-theta),
             u[1] * u[2] * (1-np.cos(theta)) + u[0] * np.sin(-theta),
             np.cos(theta) + u[2]**2 * (1-np.cos(theta))]]

    def Included_Angle(self, m):
        if np.shape(m) == (3,):
            n = np.array((m, m, m))
            coordinate_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
            angle_matrix = np.arccos(np.multiply(n, coordinate_mat)/np.linalg.norm(m))
            return [angle_matrix[0,0], angle_matrix[1,1], angle_matrix[2,2]]
        else:
            print("Included Angle Calculating error")

    
    def rotation_angle(self, m):
        angle_with_y = np.arctan(m[0]/m[2])
        xx = np.array((-np.cos(angle_with_y), 0, np.sin(angle_with_y)))
        angle_with_xx = np.arctan(m[1]/np.linalg.norm(np.array((m[0],m[2]))))
        return [angle_with_y, angle_with_xx, xx]
        


    def matrix_to_Transfer(self, m_left, m_up, m_orientation):

        Transfer = np.array([-m_left.T, m_up.T, -m_orientation.T])
        Transfer = np.c_[Transfer,np.array([0,0,0])]
        Transfer = np.r_[Transfer,np.array([[0,0,0,1]])]
        return Transfer
