# flash_detection
Individual flash detection in LArTPC optical waveforms.
Some paths still need to be updated in files and notebooks since reorganization - will fix as I encounter.

**Main Workflow**
* `notebooks/simulating_data.ipynb`: Generate a sample of optical waveform data
* `run_training.py`: Instantiate a model architecture & training configuration; train the model.
* `performance_analysis.py`: Evaluate the performance of model on various benchmarks (Configure which models to include and which evaluations to run in `model_list.yaml` and `performance_analysis_config.yaml`. Also log file paths to evaluation results in `model_list.yaml`.)
* `final_analysis.ipynb`: Load in evaluation results & plot them for specified model versions.
* `compare_networks.ipynb`: Visualize single-waveform predictions for various model versions side-by-side.