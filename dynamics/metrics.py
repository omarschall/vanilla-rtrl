import numpy as np

def assign_time_points_to_stages(signal_dict, performance_criterion,
                                 topological_criterion, window_duration=10):
    """Takes a dictionary of signals, which must include 'test_loss' and
    'aligned_graph_distances', and returns an array of stage assignments.

    Uses the 'XOR' criterion, where he stages are determined by whether or not
    performance is at criterion in combination with whether or not the
    topology is constant over time.

    Stage 1:
        constant topology
        poor performance
    Stage 2:
        changing topology
        poor performance
    Stage 3:
        changing topology
        good performance
    Stage 4:
        constant topology
        good performance

    We measure this by doing a causal (backwards-looking) convolution of the
    signal"""

    ### --- Reflect test loss over 0 time point for causal convolution --- ###

    kernel = np.ones(window_duration) / window_duration
    loss = signal_dict['test_loss']
    reflected_loss = np.concatenate([loss[window_duration-1:0:-1], loss])
    convolved_loss = np.convolve(reflected_loss, kernel, mode='valid')

    good_performance = (convolved_loss < performance_criterion)[:-1]

    ### --- Zero-pad top. metric for causal convolution --- ###

    top_metric = signal_dict['aligned_graph_distances']
    padded_metric = np.concatenate([np.zeros(window_duration - 1), top_metric])
    convolved_metric = np.convolve(padded_metric, kernel, mode='valid')

    constant_topology = convolved_metric < topological_criterion

    ### --- Assign time points to stages --- ###

    stage_assignments = np.zeros_like(top_metric)

    stage_assignments[np.where(np.logical_and(constant_topology,
                                              np.invert(good_performance)))] = 1
    stage_assignments[np.where(np.logical_and(np.invert(constant_topology),
                                              np.invert(good_performance)))] = 2
    stage_assignments[np.where(np.logical_and(np.invert(constant_topology),
                                              good_performance))] = 3
    stage_assignments[np.where(np.logical_and(constant_topology,
                                              good_performance))] = 4

    stage_assignments = stage_assignments.astype(np.int)

    return stage_assignments

