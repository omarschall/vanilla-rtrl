import numpy as np

def generate_ergodic_markov_task(one_hot_dim=4, n_inputs=4, idem=True):
    """Generates a random set of fixed_points as one hots and associated
    transitions under idempotency constraint (or not) until it finds an
    ergodic instance."""

    FPs = [np.eye(one_hot_dim)[i] for i in range(one_hot_dim)]
    n_states = len(FPs)

    done = False

    while not done:

        T_dict = {}

        for i_input in range(n_inputs):

            T = np.zeros((n_states, n_states))
            perm = np.random.permutation(range(n_states))

            for i_fp in range(n_states):
                T[i_fp, perm[i_fp]] = 1

            if idem:
                #force to be idempotent
                for i_fp in np.random.permutation(range(n_states)):
                    if T[:,i_fp].sum() > 0:
                        T[i_fp, :] = np.eye(n_states)[i_fp]

                #check if idempotent
                assert (T == T.dot(T)).all()

            T_dict['input_{}'.format(i_input)] = T

        T_avg = sum([T for T in T_dict.values()]) / len(T_dict.keys())


        T_power = np.round(np.linalg.matrix_power(T_avg, 1000), 2)

        if (T_power[0] > 0).all():
            done = True

    return FPs, T_dict

def concatenate_datasets(data_1, data_2):
    """Takes in two data dicts of form in gen_data and concatenates the data
    sequentially in time."""

    data = {'train': {}, 'test': {}}

    for dataset, io in product(['train', 'test'], ['X', 'Y']):
        data[dataset][io] = np.concatenate([data_1[dataset][io],
                                            data_2[dataset][io]], axis=0)

    return data