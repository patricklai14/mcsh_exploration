import json
import pathlib
import pickle
import shutil
import subprocess
import time

import constants
import evaluate_mcsh_model
import utils

class model_metrics:
    def __init__(self, train_errors, test_errors):
        self.train_errors = train_errors
        self.test_errors = test_errors

def evaluate_models(config_files, train_imgs, test_imgs=None, eval_mode="cv", enable_parallel=True, workspace=None):

    if len(train_imgs) == 0:
        raise RunTimeError("train dataset is empty, cannot evaluate model")

    if enable_parallel:
        if not parallel_workspace:
            raise RunTimeError("workspace cannot be None in parallel mode")

        #prepare directories
        print("Parallel processing enabled. Initializing workspace")
        workspace_subdirs = []
        
        config_path = pathlib.Path(workspace) / constants.CONFIG_DIR
        workspace_subdirs.append(config_path)

        pbs_path = pathlib.Path(workspace) / constants.PBS_DIR
        workspace_subdirs.append(pbs_path)

        temp_output_path = pathlib.Path(workspace) / constants.OUTPUT_DIR
        workspace_subdirs.append(temp_output_path)

        training_path = pathlib.Path(workspace) / constants.TRAINING_DIR
        workspace_subdirs.append(training_path)

        data_path = pathlib.Path(workspace) / constants.DATA_DIR
        workspace_subdirs.append(data_path)

        for subdir in workspace_subdirs:
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)
            
            subdir.mkdir(parents=True, exist_ok=False)

        #write data to disk
        train_data_file = data_path / constants.TRAIN_DATA_FILE
        pickle.dump(train_imgs, open(data_file, "wb" ))

        #create pace pbs files
        pbs_files = {}
        job_names = []
        model_eval_script_dir = pathlib.Path(__file__).parent.absolute()
        for config_file in config_files:
            config = json.load(open(config_file, "r"))
            job_name = config["name"]

            if job_name in pbs_files:
                raise RunTimeError("duplicate job name: {}".format(job_name))

            model_eval_script = model_eval_script_dir / constants.EVAL_MODEL_SCRIPT
            command_str = "python {} --workspace {} --job_name {} --data {} --config {}".format(
                            model_eval_script, parallel_workspace, order, train_data_file, config_file)
            pbs_file = utils.create_pbs(pbs_path, job_name, command_str, time="00:10:00")

            pbs_files[job_name] = pbs_file
            job_names.append(job_name)

        #submit jobs on pace
        for _, pbs_file in pbs_files.items():
            subprocess.run(["qsub", pbs_file])

        #collect results
        results = []
        for name in job_names:
            curr_output_file = temp_output_path / "output_{}.json".format(name)
            while not curr_output_file.exists():
                print("results for job {} not ready. Sleeping for 20s".format(name))
                time.sleep(20)

            result_dict = json.load(open(curr_output_file, "r"))
            train_mse = result_dict[TRAIN_MSE]
            test_mse = result_dict[TEST_MSE]

            results.append(model_metrics(train_mse, test_mse))

        #clear workspace
        for subdir in workspace_subdirs:
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)

        return results