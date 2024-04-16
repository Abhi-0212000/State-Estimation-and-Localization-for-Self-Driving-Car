# State-Estimation-and-Localization-for-Self-Driving-Car

This project centers on the implementation of an Extended Kalman Filter (EKF) sensor fusion pipeline for state estimation and localization in a self-driving car context.

## Overview

This repository contains the implementation of the main functionality responsible for sensor fusion and the prediction and correction of the car's position in the navigation frame. The core logic of the Extended Kalman Filter, along with the sensor data processing, has been developed as part of this project.

## Data and Credits

The majority of the code and data, including ground truth and sensor data, have been provided by the University of Toronto. The project builds upon this foundation, with the main focus being the implementation of the state estimation pipeline.

### Motion Model

**Position Equation:**

$$
p_k=p_{k-1}+\mathrm{\Delta t}v_{k-1}+\frac{\mathrm{\Delta}t^2}{2}\left[C_{ns}f_{k-1}+g\right]
$$

**Velocity Equation:**

$$
v_k=v_{k-1}+\mathrm{\Delta t}\left[c_{ns}f_{k-1}+g\right]
$$

**Orientation (Quaternion Update):**

$$
q_k=q_{k-1}\otimes q\left(\omega_k\mathrm{\Delta t}\right)=\mathrm{\Omega}\left(q\left(\omega_{k-1}\mathrm{\Delta t}\right)\right)q_{k-1}
$$

where,

$$
C_{ns}=C_{ns}\left(q_{k-1}\right)
\ \ \ \ \ \ \ \ \, \ \ \ \ \ \ \ \ \ 
\Omega\left(\begin{bmatrix} 
q_{\omega} \\ 
q_{\text{v}} 
\end{bmatrix}\right) = q_{\omega} \cdot \mathbf{I} + 
\begin{bmatrix} 
0 & -q_{\text{v}}^T \\ 
q_{\text{v}} & -[q_{\text{v}}]_{\times}
\end{bmatrix}
\ \ \ \ \ \ \ \ \, \ \ \ \ \ \ \ \ \
q(θ)=\begin{bmatrix} 
\cos\frac{|\theta|}{2} \\ 
\frac{\theta}{|\theta|}\sin\frac{|\theta|}{2}
\end{bmatrix}
$$


### Motion Model

**Error State**

$$
\delta x_k = 
\begin{bmatrix}
\delta p_k \\
\delta v_k \\
\delta y_k
\end{bmatrix} 
\in \mathbb{R}^9
$$

**Error Dynamics**

$$
\delta x_k=F_{k-1}\delta x_{k-1}+L_{k-1}n_{k-1}
$$

where,

$$
F_{k-1} = \begin{bmatrix}
I & I\Delta t & 0 \\
0 & I & -[C_{ns}f_{k-1}]_{\times} \Delta t \\
0 & 0 & I
\end{bmatrix}
\ \ \ \ \ \ \ \ \, \ \ \ \ \ \ \ \ \
L _{k-1}=\begin{bmatrix}
0 & 0 \\
I & 0 \\
0 & I
\end{bmatrix}
\ \ \ \ \ \ \ \ \, \ \ \ \ \ \ \ \ \
n_k \sim N(0, Q_k)
\sim N(0, \Delta t^2 \begin{bmatrix}
\sigma _{acc}^2 & 0 \\
0 & \sigma _{gyro}^2
\end{bmatrix}
)
$$

where,

- `delta_p_k` : position error
- `delta_v_k` : velocity error
- `delta_Phi_k` : Orientation error
- `n_k_1` : Measurement Noise
- `I` : Identity matrix of size `3x3`
- `Sigma_acc^2` : diagonal acceleration variance matrix of size `3x3`
- `Sigma_gyro^2` : diagonal gyroscope variance matrix of size `3x3`

### Measurement Model

**Measurement Model | GNSS**

$$
y_k=h(x_k)+ν_k \ \ \ \==> \ \ \ \  y_k=H_k x_k + ν_k \ \ \ \==> \ \ \ \ y_k=\begin{bmatrix} I & 0 & 0 \end{bmatrix} x_k+ν_k \ \ \ \ \ \ \,\ \ \ \ \ \ \ ν_k \sim N(0, R_{GNSS})
$$

**Measurement Model | LIDAR**

$$
y_k=h(x_k)+ν_k \ \ \ \==> \ \ \ \  y_k=H_k x_k + ν_k \ \ \ \==> \ \ \ \ y_k=\begin{bmatrix} I & 0 & 0 \end{bmatrix} x_k+ν_k \ \ \ \ \ \ \,\ \ \ \ \ \ \ ν_k \sim N(0, R_{LIDAR})
$$


## EKF  |  IMU + GNSS + LIDAR  
### Filter Loop Equations

**1. Update State with IMU Inputs**

$$
\stackrel{\vee}{x_k} = 
\begin{bmatrix}
\stackrel{\vee}{p_k} \\
\stackrel{\vee}{v_k} \\
\stackrel{\vee}{q_k}
\end{bmatrix} 
$$

$$
\stackrel{\vee}{p_k}=p_{k-1}+\Delta t v_{k-1}+\frac{\Delta t^2}{2}\left(C_{ns}f_{k-1}+g_n\right) 
$$

$$
\stackrel{\vee}{v_k}=v_{k-1}+\mathrm{\Delta t}\left(C_{ns}f_{k-1}+g_n\right)	
$$

$$
\stackrel{\vee}{q_k}=\mathrm{\Omega}\left(q\left(\omega_{k-1}\mathrm{\Delta t}\right)\right)q_{k-1}=q_{k-1}\otimes q\left(\omega_{k-1}\mathrm{\Delta t}\right)
$$

**2. Propagate Uncertainity**

$$
\stackrel{\vee}{P_k} = F_{k-1}P_{k-1}F_{k-1}^T+L_{k-1}Q_{k-1}L_{k-1}^T
$$

**3. Compute Kalman Gain**

$$
K_k=\stackrel{\vee}{P_k} H_k^T\left(H_k \stackrel{\vee}{P_k} H_k^T+R\right)^{-1}
$$

**4. Compute Error State**

$$
\delta x_k=K_k\left(y_k-\stackrel{\vee}{p_k}\right)
$$

**5. Compute Predicted State**

$$
\overset{\wedge}{p_k}=\stackrel{\vee}{p_k}+\delta p_k
\ \ \ \ \ \ \,\ \ \ \ \ \ 
\overset{\wedge}{v_k}=\stackrel{\vee}{v_k}+\delta v_k
\ \ \ \ \ \ \,\ \ \ \ \ \ 
\overset{\wedge}{q_k}=q\left(\delta\emptyset\right)\otimes\stackrel{\vee}{q_k}
$$

**6. Computed Corrcted Covariance**

$$
\overset{\wedge}{P_k}=\left(I-K_kH_k\right)\stackrel{\vee}{P_k}
$$


## Results

The key results of this project include:

- **Ground Truth and Estimated Trajectories Plot**: Visual representation of the ground truth trajectory alongside the estimated trajectory derived from the implemented sensor fusion pipeline.
  
- **Error Plot**: Analysis of the error in each degree of freedom, with uncertainty estimates included. This provides insights into the performance of the state estimation process.


<div>
    <img src="https://github.com/Abhi-0212000/State-Estimation-and-Localization-for-Self-Driving-Car/assets/70425157/1ad2d3ad-420a-4cd2-aa20-80c9db46ceed" alt="GroundTruthAndEstimatedTrajectory" style="width:45%; float:left; margin-right:5%">
    <img src="https://github.com/Abhi-0212000/State-Estimation-and-Localization-for-Self-Driving-Car/assets/70425157/0fc08add-5ed6-4531-8453-5b520ddb6063" alt="ErrorPlots" style="width:45%; float:left;">
</div>

