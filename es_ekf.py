
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion


#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:   
#       user-defined data_type = <class 'data.data.Data'>
#     a: Acceleration of the vehicle, in the inertial frame. 
#        data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     v: Velocity of the vehicle, in the inertial frame. 
#        data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     p: Position of the vehicle, in the inertial frame. 
#        data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     alpha: Rotational acceleration of the vehicle, in the inertial frame. 
#            data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     w: Rotational velocity of the vehicle, in the inertial frame. 
#        data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame. 
#        data_type = <class 'numpy.ndarray'> shape = (8734, 3)
#     _t: Timestamp in ms. 
#         data_type = <class 'numpy.ndarray'> shape = (10920,)
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame i.e IMU frame). 
#          Type of 'imu_f': <class 'data.utils.StampedData'>
#     data: The actual data. 
#           Type of 'imu_f.data': <class 'numpy.ndarray'>, Shape of 'imu_f.data': (10918, 3)
#     t: Timestamps in ms. 
#        Type of 'imu_f.t': <class 'numpy.ndarray'>, Shape of 'imu_f.t': (10920,)
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame i.e IMU frame).
#     data: The actual data. 
#           Type of 'imu_w.data': <class 'numpy.ndarray'>, Shape of 'imu_w.data': (10918, 3)
#     t: Timestamps in ms. 
#        Type of 'imu_w.t': <class 'numpy.ndarray'>, Shape of 'imu_w.t': (10920,)
#   gnss: StampedData object with the GNSS data.
#     data: The actual data. 
#           Type of 'gnss.data': <class 'numpy.ndarray'>, Shape of 'gnss.data': (55, 3)
#     t: Timestamps in ms. 
#        Type of 'gnss.t': <class 'numpy.ndarray'>, Shape of 'gnss.t': (55,)
#   lidar: StampedData object with the LIDAR data (positions only). (data is in Lidar frame)
#     data: The actual data. 
#           Type of 'lidar.data': <class 'numpy.ndarray'>, Shape of 'lidar.data': (521, 3)
#     t: Timestamps in ms. 
#        Type of 'lidar.t': <class 'numpy.ndarray'>, Shape of 'lidar.t': (521,)
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']


################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()


################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
# we've got the RPY values from the calibration process
# for PART1 and PART3
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# for PART2
'''C_li = np.array([
    [ 0.9975 , -0.04742,  0.05235],
    [ 0.04992,  0.99763, -0.04742],
    [-0.04998,  0.04992,  0.9975 ]
])'''

# we've got the translation vector values from the calibration process
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
# this specific transformation formula tailored to this particular vehicle and its sensor setup.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
# The variance values are typically determined through a combination of 
#theoretical analysis, empirical testing, and sometimes manual tuning.
################################################################################################
# for Part1
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 1.00

# for PART2
'''var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 8.9'''

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
'''print('g shape is: ', g.shape)
print('g type is: ', type(g))'''
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
# l_jac is
# [[0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 1.]]
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian
# h_jac is
# [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0.]]

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates. shape is (8734, 3) i.e each row has x, y, z coords
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates. shape is (8734, 3) i.e each row has v_x, v_y, v_z components of velocity
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions. shape is (8734, 4) i.e each row has 4 params of a quaternion. 1 scalar part + 3 vector parts
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep. shape is (8734, 9, 9) i.e 8734 9x9 matrices. each matrix is 9x9 i.e 3 positioon, 3 velocity, 3 orientation error state. Later we convert 3x1 orientation to 4x1 quaternion 
# While a simple 3x1 orientation vector may seem intuitive, using quaternions provides benefits in terms of efficiency, stability, interpolation, and mathematical properties. These advantages make quaternions a preferred choice for representing orientation

# Set initial values.
p_est[0] = gt.p[0]
'''print('p_est[0] is: ', p_est[0])
print('p_est[0] shape is: ', p_est[0].shape)'''
v_est[0] = gt.v[0]
print('gt.r[0] is: ', gt.r[0])
print('shape of gt.r[0] is :', gt.r[0].shape)
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
print('q_est[0] is: ', q_est[0])
print('shape of q_est[0] is :', q_est[0].shape)
p_cov[0] = np.zeros(9)  # covariance of estimate
'''print('p_cov[0] is: ', p_cov[0])
print('p_cov[0] shape is: ', p_cov[0].shape)'''
gnss_i  = 0
lidar_i = 0

R_gnss = np.eye(3)*var_gnss
R_lidar = np.eye(3)*var_lidar


#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################



def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain 
    K_k = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + sensor_var)

    # 3.2 Compute error state
    del_x_k = K_k @ np.reshape(y_k - p_check, (3, 1))

    # 3.3 Correct predicted state
    p_hat = p_check + np.reshape(del_x_k[0:3], (3, ))
    v_hat = v_check + np.reshape(del_x_k[3:6], (3, ))
    q_hat = Quaternion(axis_angle=angle_normalize(del_x_k[6:])).quat_mult_left(q_check)

    # 3.4 Compute corrected covariance
    I = np.eye(9)
    p_cov_hat = (I - K_k @ h_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat


#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    #print(k)
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    
    p_k_1 = p_est[k-1]
    v_k_1 = v_est[k-1]
    q_k_1 = q_est[k-1]
    p_cov_k_1 = p_cov[k-1]

    # Convert quaternion to rotation matrix
    C_ns = Quaternion(*q_k_1).to_mat()
    
    imu_f_transformed = C_ns.dot(imu_f.data[k - 1])
    

    # 1. Update state with IMU inputs
    p_check = p_k_1 + delta_t * v_k_1 + 0.5 * (delta_t ** 2) * (imu_f_transformed + g)
    v_check = v_k_1 + delta_t * (imu_f_transformed + g)
    
    omega = angle_normalize(imu_w.data[k-1]*delta_t)

    q_check = Quaternion(*q_k_1).quat_mult_left(Quaternion(axis_angle=omega))
    
    # 1.1 Linearize the motion model and compute Jacobians
    col2_row_1 = -delta_t * skew_symmetric(imu_f_transformed)
    
    F_k_1 = np.block([[np.eye(3), np.eye(3) * delta_t, np.zeros((3, 3))],
              [np.zeros((3, 3)), np.eye(3), col2_row_1],
              [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])
              
    L_k_1 = np.block([[np.zeros((3, 3)), np.zeros((3, 3))],
              [np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), np.eye(3)]])
              
    Q_k_1 = delta_t**2 * np.block([[var_imu_f*np.eye(3), np.zeros((3, 3))],
              [np.zeros((3, 3)), var_imu_w*np.eye(3)]])
    

    # 2. Propagate uncertainty
    p_cov_check = F_k_1 @ p_cov_k_1 @ F_k_1.T + L_k_1 @ Q_k_1 @ L_k_1.T
    

    # 3. Check availability of GNSS and LIDAR measurements
    if imu_f.t[k] in gnss.t:  # we are trusting gnss more than lidar if both data's are available at time k becuase var_gnss < var_lidar 
        index = list(gnss.t).index(imu_f.t[k])
        y_k = gnss.data[index]
        p_hat, v_hat, q_hat, p_cov_hat = measurement_update(R_gnss, p_cov_check, y_k, p_check, v_check, q_check)
    elif imu_f.t[k] in lidar.t:
        index = list(lidar.t).index(imu_f.t[k])
        y_k = lidar.data[index]
        p_hat, v_hat, q_hat, p_cov_hat = measurement_update(R_lidar, p_cov_check, y_k, p_check, v_check, q_check)
    else:
        p_hat, v_hat, q_hat, p_cov_hat = p_check, v_check, q_check, p_cov_check
    
    # Update states (save)
    p_est[k] = p_hat
    v_est[k] = v_hat
    q_est[k] = q_hat
    p_cov[k] = p_cov_hat



print('MAIN FILTER LOOP DONE!!!')

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]
# p2_str = ''
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])
# with open('pt2_submission.txt', 'w') as file:
#     file.write(p2_str)

# Pt. 3 submission
# p3_indices = [6800, 7600, 8400, 9200, 10000]
# p3_str = ''
# for val in p3_indices:
#     for i in range(3):
#         p3_str += '%.3f ' % (p_est[val, i])
# with open('pt3_submission.txt', 'w') as file:
#     file.write(p3_str)