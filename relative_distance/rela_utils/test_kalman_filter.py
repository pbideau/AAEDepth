from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_kalman(measurements,filtered_state_means):

    # Plot the original measurements
    plt.figure(figsize=(12, 6))
    plt.plot(measurements, label='Original Measurements', marker='o', linestyle='-', color='b')

    # Plot the filtered output
    plt.plot(filtered_state_means[:, 0], label='Filtered Output', marker='.', linestyle='-', color='r')

    # Add labels and legend
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Kalman Filter: Original Measurements vs Filtered Output')
    plt.legend()

    # Show the plot
    plt.show()


def simple_test_plot():

    # Define the initial state and transition matrix
    x_init = np.array([0, 0])
    transition_matrix = [[1, 1], [0, 1]]

    # Define the measurement matrix and measurement noise
    measurement_matrix = [[1, 0]]
    measurement_noise = np.array([[0.1]])

    # Define the process noise
    process_noise = np.array([[0.1, 0.1], [0.1, 0.2]])

    # Create the Kalman filter
    kf = KalmanFilter(transition_matrices=transition_matrix,
                    observation_matrices=measurement_matrix,
                    initial_state_mean=x_init,
                    observation_covariance=measurement_noise,
                    transition_covariance=process_noise)

    # Generate some fake measurements
    measurements = np.random.randn(50, 1)  # shape (50,1)

    # Run the Kalman filter
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)




# def test():
def kalman_filter(est_obj_depths):
    x_init = np.array([est_obj_depths[0]])
    # transition_matrix = [[1, 1], [0, 1]]
    transition_matrix = [1]

    # Define the measurement matrix and measurement noise
    # measurement_matrix = [[1, 0]]
    measurement_matrix = [1]
    measurement_noise = np.array([5])

    # Define the process noise
    process_noise = np.array([0.1])

    # Create the Kalman filter
    kf = KalmanFilter(transition_matrices=transition_matrix,
                    observation_matrices=measurement_matrix,
                    initial_state_mean=x_init,
                    observation_covariance=measurement_noise,   # R
                    transition_covariance=process_noise)
    measurements = est_obj_depths[1:]
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    plot_kalman(measurements,filtered_state_means)

if __name__ == '__main__':   
    # simple_test_plot() 
    # quit()


    res_obj_dict_dir = '/Users/cainan/Desktop/active_perception/event_depth_esti/code/stppp_and_output/x-student-events/output/2024-02-19_NewRes_0.05dt'
    sequences = ['scene_03_00_000000', 'scene_03_01_000000', 'scene_03_02_000000', 'scene_03_02_000001',
            'scene_03_02_000002', 'scene_03_02_000003', 'scene_03_03_000000', 'scene_03_03_000001',
            'scene_03_03_000002', 'scene_03_04_000000']
    sequence = sequences[0]
    rel_res_obj_dict_file = glob.glob(os.path.join(res_obj_dict_dir, '*'+sequence+'*'+ 'rel_res_obj_dict' +'*.npy')) 
    if len(rel_res_obj_dict_file) != 1:
        print('!!!!len(depth_files) != 1!!!!!')
        quit()

    rel_res_obj_dict = np.load(rel_res_obj_dict_file[0], allow_pickle=True).item()
    object_ids = list(rel_res_obj_dict.keys())
    object_ids.sort()
    object_id = object_ids[0]
    est_obj_depths = rel_res_obj_dict[object_id]['distance']  # list len 946
    est_obj_time = rel_res_obj_dict[object_id]['ts']         # list len 946


    x_init = np.array([est_obj_depths[0]])
    # transition_matrix = [[1, 1], [0, 1]]
    transition_matrix = [1]

    # Define the measurement matrix and measurement noise
    # measurement_matrix = [[1, 0]]
    measurement_matrix = [1]
    measurement_noise = np.array([5])

    # Define the process noise
    process_noise = np.array([0.1])

    # Create the Kalman filter
    kf = KalmanFilter(transition_matrices=transition_matrix,
                    observation_matrices=measurement_matrix,
                    initial_state_mean=x_init,
                    observation_covariance=measurement_noise,   # R
                    transition_covariance=process_noise)
    measurements = est_obj_depths[1:]
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    plot_kalman(measurements,filtered_state_means)
    print('1')


from pykalman import KalmanFilter

def plot_kalman(measurements,filtered_state_means):

    # Plot the original measurements
    plt.figure(figsize=(12, 6))
    plt.plot(measurements, label='Original Measurements', marker='o', linestyle='-', color='b')

    # Plot the filtered output
    plt.plot(filtered_state_means[:, 0], label='Filtered Output', marker='.', linestyle='-', color='r')

    # Add labels and legend
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Kalman Filter: Original Measurements vs Filtered Output')
    plt.legend()

    # Show the plot
    plt.show()

def kalman_filter(est_obj_depths):
    x_init = np.array([est_obj_depths[0]])
    transition_matrix = [1]
    measurement_matrix = [1]
    measurement_noise = np.array([200])
    process_noise = np.array([0.1])
    kf = KalmanFilter(transition_matrices=transition_matrix,
                    observation_matrices=measurement_matrix,
                    initial_state_mean=x_init,
                    observation_covariance=measurement_noise,   # R
                    transition_covariance=process_noise)
    measurements = est_obj_depths[1:]
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    # plot_kalman(measurements,filtered_state_means)
    filtered_state_means = filtered_state_means.flatten().tolist()
    filtered_state_means.insert(0, est_obj_depths[0])

    return filtered_state_means
