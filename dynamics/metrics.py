import numpy as np

def assign_time_points_to_stages(signal_dict, performance_criterion,
                                 topological_criterion, loss_window=10,
                                 topological_window=10,
                                 post_process=True, time_point_trigger=4):
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

    topological_kernel = np.ones(topological_window) / topological_window
    loss = signal_dict['test_loss']
    reflected_loss = np.concatenate([loss[topological_window-1:0:-1], loss])
    convolved_loss = np.convolve(reflected_loss, topological_kernel, mode='valid')

    good_performance = (convolved_loss < performance_criterion)[:-1]

    ### --- Zero-pad top. metric for causal convolution --- ###

    loss_kernel = np.ones(loss_window) / loss_window
    top_metric = signal_dict['aligned_graph_distances']
    padded_metric = np.concatenate([np.zeros(loss_window - 1), top_metric])
    convolved_metric = np.convolve(padded_metric, loss_kernel, mode='valid')

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

    if post_process:
        sa, tst = post_process_stage_assignments(stage_assignments,
                                                 time_point_trigger=time_point_trigger)

        #sa = stage assignments
        #tst = time of stage transitions

        return sa, tst
    else:
        return stage_assignments

def post_process_stage_assignments(stage_assignments, time_point_trigger=4):
    """For a given set of stage assignments, performs post processing by making
    each period contiguous.

    Starting with stage 2, we look for the first time point with 4 following
    consecutive time points that were initially categorized as stage 2. Then
    this is considered the "transition point," and we repeat for stage 3 and 4.
    """

    #Copy original stage assignments
    ret = stage_assignments.copy()
    t_stage_transition_prev = 0
    t_stage_transitions = []

    for i_stage in [2, 3, 4]:

        #Find points assigned to stage i_stage
        stage_i_points = (stage_assignments == i_stage)

        #Transition times
        stage_i_transitions = (stage_assignments == i_stage)

        #Roll array a number of time steps forward as specified by time_point_trigger
        for i_roll in range(1, time_point_trigger):
            stage_roll = np.roll(stage_i_points, -i_roll)
            stage_i_transitions = np.logical_and(stage_i_transitions,
                                                 stage_roll)

        try:
            t_stage_transition = sorted(np.where(stage_i_transitions)[0])[0]
        except IndexError:
            t_stage_transition = None

        #Put everything up to stage n transition in stage n-1
        if t_stage_transition_prev is not None:
            ret[t_stage_transition_prev:t_stage_transition] = i_stage - 1

        t_stage_transition_prev = t_stage_transition

        t_stage_transitions.append(t_stage_transition)

    if t_stage_transition_prev is not None:
        ret[t_stage_transition_prev:] = 4

    return ret, t_stage_transitions