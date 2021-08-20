import os, pickle

def unpack_analysis_results(data_path):
    """For a path to results, unpacks the data into a dict of checkpoints
    and sorted corresponding indices."""

    done = False
    checkpoints = {}
    i = 0
    i_missing = 0
    while not done:

        file_path = os.path.join(data_path, 'result_{}'.format(i))

        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
            checkpoints.update(result)
        except FileNotFoundError:
            i_missing += 1

        i += 1
        if i_missing > 5:
            done = True
    indices = sorted([int(k.split('_')[-1]) for k in
                      checkpoints.keys() if 'checkpoint' in k])

    return indices, checkpoints
