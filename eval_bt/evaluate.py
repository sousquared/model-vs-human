from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    # models = ["resnet50", "bagnet33", "simclr_resnet50x1"]
    models = ["alexnet", "alexnet_s", "alexnet_b", "alexnet_bs", "alexnet_b2s"]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # You can edit plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    import time
    t1 = time.time()

    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    t2 = time.time()
    elapsed_time_eval = t2 - t1
    print(f"elapsed time: {elapsed_time_eval}")

    # 2. plot the evaluation results
    run_plotting()
    t3 = time.time()
    elapsed_time_plot = t3 - t2

    print(f"elapsed time: {elapsed_time_eval}")
    print(f"elapsed time: {elapsed_time_plot}")
