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
    def __init__(self, train_error, test_error):
        self.train_error = train_error
        self.test_error = test_error

def evaluate_models(config_files, train_dataset, test_dataset=None, eval_mode="cv", enable_parallel=True, workspace=None):

    if train_dataset.empty():
        raise RuntimeError("train dataset is empty, cannot evaluate model")

    if enable_parallel:
        if not workspace:
            raise RuntimeError("workspace cannot be None in parallel mode")

        #prepare directories
        workspace_path = pathlib.Path(workspace)
        print("Parallel processing enabled. Initializing workspace in {}".format(workspace_path))
        workspace_subdirs = []
        
        config_path = workspace_path / constants.CONFIG_DIR
        workspace_subdirs.append(config_path)

        pbs_path = workspace_path / constants.PBS_DIR
        workspace_subdirs.append(pbs_path)

        temp_output_path = workspace_path / constants.OUTPUT_DIR
        workspace_subdirs.append(temp_output_path)

        training_path = workspace_path / constants.TRAINING_DIR
        workspace_subdirs.append(training_path)

        data_path = workspace_path / constants.DATA_DIR
        workspace_subdirs.append(data_path)

        for subdir in workspace_subdirs:
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)
            
            subdir.mkdir(parents=True, exist_ok=False)

        #write data to disk
        train_data_file = data_path / constants.TRAIN_DATA_FILE
        pickle.dump(train_dataset, open(train_data_file, "wb" ))

        #create pace pbs files
        pbs_files = {}
        job_names = []
        model_eval_script_dir = pathlib.Path(__file__).parent.absolute()
        for config_file in config_files:
            config = json.load(open(config_file, "r"))
            job_name = config["name"]

            if job_name in pbs_files:
                raise RuntimeError("duplicate job name: {}".format(job_name))

            model_eval_script = model_eval_script_dir / constants.EVAL_MODEL_SCRIPT
            command_str = "python {} --workspace {} --job_name {} --data {} --config {}".format(
                            model_eval_script, workspace, job_name, train_data_file, config_file)
            pbs_file = utils.create_pbs(pbs_path, job_name, command_str, time="00:10:00")

            pbs_files[job_name] = pbs_file
            job_names.append(job_name)

        #submit jobs on pace
        for name, pbs_file in pbs_files.items():
            print("Submitting job: {}".format(name))
            subprocess.run(["qsub", pbs_file])

        #collect results
        results = []
        for name in job_names:
            curr_output_file = temp_output_path / "output_{}.json".format(name)
            while not curr_output_file.exists():
                print("results for job {} not ready. Sleeping for 20s".format(name))
                print("looking for: {}".format(curr_output_file))
                time.sleep(20)

            result_dict = json.load(open(curr_output_file, "r"))
            train_mse = result_dict[constants.TRAIN_MSE]
            test_mse = result_dict[constants.TEST_MSE]

            results.append(model_metrics(train_mse, test_mse))

        #clear workspace
        if workspace_path.exists() and workspace_path.is_dir():
            shutil.rmtree(workspace_path) 

        return results
