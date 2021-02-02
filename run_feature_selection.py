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

    OUTPUT_DIR = "D:\\Work\\sandbox\\vip\\outputs"

    if args_dict['mcsh_param'] == "orders":
        if args_dict['method'] == "forward":
            selection_methods.forward_selection(OUTPUT_DIR)

        if args_dict['method'] == "backward":
            selection_methods.backward_elimination(OUTPUT_DIR)

if __name__ == "__main__":
    main()
