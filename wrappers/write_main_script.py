def write_main_script(analyze=True, distances=True):

    code = 'from wrappers import '

    if analyze:

        code += 'analyze_training_run()\n'
