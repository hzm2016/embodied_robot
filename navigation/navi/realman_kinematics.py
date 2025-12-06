import numpy as np

def mod_dh_matrix(alpha_prev, a_prev, d, theta):
    """
    Calculate the Modified DH transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha_prev)
    sa = np.sin(alpha_prev)
    
    return np.array([
        [ct, -st, 0, a_prev],
        [st*ca, ct*ca, -sa, -sa*d],
        [st*sa, ct*sa, ca, ca*d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(theta_values, robot_type="RM75-B"):
    """
    Calculate forward kinematics for the 7-DOF robot
    Args:
        theta_values: List of 7 joint angles in radians
        robot_type: "RM75-B" or "RM75-6F"
    Returns:
        4x4 transformation matrix for end-effector
    """
    # Modified DH Parameters
    alpha = [0, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2]  # α_(i-1)
    a = [0, 0, 0, 0, 0, 0, 0]  # a_(i-1)
    d = [240.5, 0, 256, 0, 210, 0, 144 if robot_type=="RM75-B" else 172.5]  # d_i
    
    # Initialize transformation matrix
    T = np.eye(4)
    
    # Calculate forward kinematics
    for i in range(7):
        Ti = mod_dh_matrix(alpha[i], a[i], d[i], theta_values[i])
        T = T @ Ti
    
    return T

if __name__ =="__main__":
    # Test forward kinematics
    theta_values = [0, 0, 0, 0, 0, 0, 0]
    T = forward_kinematics(theta_values)
    print(T)