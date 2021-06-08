def write_main_script(analyze=True, distances=True):

    code = "from wrappers import analyze_training_run\n"
         + "import os\n"

    if analyze:

        code += ("analyze_training_run()\n"
              + "result['i_job'] = i_job\n"
              + "result['config'] = params\n"
              + "save_dir = os.environ['SAVEPATH']\n")


        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'result_' + str(i_job))

        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
