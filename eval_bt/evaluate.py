from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--evaluate", "-e",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--plot", "-p",
    action="store_true",
    default=False,
)

def run_evaluation():
    # models = ["resnet50", "bagnet33", "simclr_resnet50x1"]
    models = ["alexnet", "alexnet_s", "alexnet_b", "alexnet_bs", "alexnet_b2s"]
    # models = ["alexnet16_s"]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "bt-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # You can edit plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    import time

    args = parser.parse_args()

    if args.evaluate:
        # 1. evaluate models on out-of-distribution datasets
        t1 = time.time()
        run_evaluation()
        t2 = time.time()
        elapsed_time_eval = t2 - t1
        print(f"elapsed time: {elapsed_time_eval}")

    if args.plot:
        # 2. plot the evaluation results
        t3 = time.time()
        run_plotting()
        t4 = time.time()
        elapsed_time_plot = t4 - t3
        print(f"elapsed time: {elapsed_time_plot}")
