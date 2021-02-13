from ase import Atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory
from ase.io import read

import numpy as np
import pandas as pd

import argparse
import copy
import os
import pdb

import evaluate_mcsh_model
import selection_methods

def main():
    parser = argparse.ArgumentParser(description="Run feature selection")
    parser.add_argument("--mcsh_param", type=str, required=True, 
                        help="MCSH parameters on which to perform feature selection")
    parser.add_argument("--method", type=str, required=True,
                        help="feature selection method to use")

    args = parser.parse_args()
    args_dict = vars(args)

    dir_prefix = "D:\\Work\\sandbox\\vip"
    OUTPUT_DIR = "D:\\Work\\sandbox\\vip\\outputs"

    #setup dataset
    np.random.seed(3)
    distances = np.linspace(2, 5, 100)
    images = []
    for i in range(len(distances)):
        l = distances[i]
        image = Atoms(
            "CuCO",
            [
                (-l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0)),
                (0, 0, 0),
                (l * np.sin(0.65), l * np.cos(0.65), np.random.uniform(low=-4.0, high=4.0))
            ],
        )

        image.set_cell([10, 10, 10])
        image.wrap(pbc=True)
        image.set_calculator(EMT())
        images.append(image)

    elements = ["Cu","C","O"]
    atom_gaussians = {"C": os.path.join(dir_prefix, "config\\MCSH_potential\\C_coredensity_5.g"),
                      "O": os.path.join(dir_prefix, "config\\MCSH_potential\\O_totaldensity_7.g"),
                      "Cu": os.path.join(dir_prefix, "config\\MCSH_potential\\Cu_totaldensity_5.g")}
    data = evaluate_mcsh_model.dataset(images, elements, atom_gaussians)

    if args_dict['mcsh_param'] == "orders":
        if args_dict['method'] == "forward":
            selection_methods.forward_selection(OUTPUT_DIR, data)

        if args_dict['method'] == "backward":
            selection_methods.backward_elimination(OUTPUT_DIR, data)

if __name__ == "__main__":
    main()

