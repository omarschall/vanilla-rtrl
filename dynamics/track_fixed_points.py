import numpy as np
from scipy.spatial import distance

class Node_Trajectory:

    def __init__(self, checkpoints_dict, distance_threshold=0.01):
        """Initializes an instance of a Node Trajectory by specifying
        a dictionary of checkpoints to build from."""

        self.distance_threshold = distance_threshold
        self.unique_node_trajectories = self.align_nodes(checkpoints_dict)

    def align_nodes(self, checkpoints_dict):
        """Takes a dictionary of checkpoints and turns it into a list of
        unique node trajectories.

        A unique node trajectory is a dictionary of time indices (for the
        duration of the node's "lifespan" and its corresponding occupancies
        in state space."""

        self.nodes_dict = {int(key.split('_')[-1]) : checkpoints_dict[key]['nodes']
                           for key in checkpoints_dict
                           if 'checkpoint' in key}

        #List of unique no
        unique_node_trajectories = []

        ### --- Initialize with first nodes --- ###

        initial_nodes = self.nodes_dict[0]
        active_node_idx = list(range(initial_nodes.shape[0]))
        n_total_nodes = len(active_node_idx)
        for node in initial_nodes:
            unique_node_trajectories.append({'time_steps': [0],
                                             'node_coordinate': [node]})
        prev_nodes = initial_nodes
        for time_step, nodes_array in self.nodes_dict.items():
            #print(time_step)

            n_nodes = nodes_array.shape[0]
            n_prev_nodes = prev_nodes.shape[0]
            I_x = []
            I_y = []
            D = distance.cdist(prev_nodes, nodes_array)
            D[np.where(D > self.distance_threshold)] = np.inf
            while len(I_x) < n_nodes:

                d = np.argmin(D)
                d_min = np.min(D)
                if d_min == np.inf:
                    break

                x, y = (d // n_nodes), (d % n_nodes)

                I_x.append(x)
                I_y.append(y)

                D[x, :] = np.inf
                D[:, y] = np.inf

            I_ = [I_y[i_x] for i_x in np.argsort(I_x)]
            I_f = sorted(I_x)
            I_b = [i for i in I_]
            extra_indices = list(range(n_nodes))
            [extra_indices.remove(i) for i in I_]
            I = I_ + extra_indices
            n_matched_nodes = len(I_x)
            #Keep only active nodes, in same relative order
            active_node_idx = [active_node_idx[i_x] for i_x in I_f]
            #Add extra new nodes
            n_new_nodes = n_nodes - n_matched_nodes
            new_nodes = list(range(n_total_nodes, n_total_nodes + n_new_nodes))
            active_node_idx += new_nodes
            n_total_nodes += n_new_nodes

            for i_active_node, active_node in enumerate(active_node_idx):
                node = nodes_array[I[i_active_node]]

                if i_active_node < n_matched_nodes:
                    unique_node_trajectories[active_node]['time_steps'].append(time_step)
                    unique_node_trajectories[active_node]['node_coordinate'].append(node)
                else:
                    unique_node_trajectories.append({})
                    unique_node_trajectories[active_node]['time_steps'] = [time_step]
                    unique_node_trajectories[active_node]['node_coordinate'] = [node]


            # if n_prev_nodes == n_nodes:
            #
            #
            #
            # if n_prev_nodes > n_nodes:
            #
            #     #Keep only the active nodes
            #     active_node_idx = [active_node_idx[i_x] for i_x in I_x]
            #
            #     for i_active_node, active_node in enumerate(active_node_idx):
            #
            #         node = nodes_array[I[i_active_node]]
            #         unique_node_trajectories[active_node]['time_steps'].append(time_step)
            #         unique_node_trajectories[active_node]['node_coordinate'].append(node)
            #
            # if n_prev_nodes < n_nodes:
            #
            #     for i_active_node, active_node in enumerate(active_node_idx):
            #
            #         node = nodes_array[I[i_active_node]]
            #         unique_node_trajectories[active_node]['time_steps'].append(time_step)
            #         unique_node_trajectories[active_node]['node_coordinate'].append(node)
            #
            #     n_new_nodes = n_nodes - n_prev_nodes
            #     new_nodes = list(range(n_total_nodes, n_total_nodes + n_new_nodes))
            #     n_old_nodes = len(active_node_idx)
            #     active_node_idx += new_nodes
            #     n_total_nodes += n_new_nodes
            #
            #     for i_active_node, active_node in enumerate(new_nodes):
            #         node = nodes_array[I[i_active_node + n_old_nodes]]
            #         unique_node_trajectories.append({})
            #         unique_node_trajectories[active_node]['time_steps'] = [time_step]
            #         unique_node_trajectories[active_node]['node_coordinate'] = [node]

            prev_nodes = nodes_array[I]

        return unique_node_trajectories