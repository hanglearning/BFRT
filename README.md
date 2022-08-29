
# BFRT (FL Part)

## Introduction
<b>BFRT</b> is a <b>B</b>lockchained  <b>F</b>ederated  learning  architecture  for  a <b>R</b>eal-time  <b>T</b>raffic flow  prediction system. This repo hosts the code of the real-time federated learning portion of the work. 

Please refer to [CCGrid '22](https://fcrlab.unime.it/ccgrid22/) [BFRT Paper](https://ieeexplore.ieee.org/document/9826108) for detailed explanations and the algorithms.

## Run the Simulation
### <ins>Suggested</ins> Running Environment
We recommend running the code files on Google Colab, the easiest and fastest way to get them to run. With the resume functionality built in the code (introduced later), the free version of Colab is good enough to handle an entire simulation. We have provided a [sample Colab file](https://colab.research.google.com/drive/1oW4hnAT8cYOFc11pQilLIkGFplqm0zca#scrollTo=G6BMOEhVnPed) to demonstrate the running steps. You may find detailed instructions in this Colab file. 

If you wish to set up the code to run on your local machine, we assume you are familiar with setting up numpy, pandas and TensorFlow for your OS. Then you could use the same commands to run the code. Running steps are provided in the next section.

### 1. Steps to run the simulation
The gateway to running the simulation is [main.py](https://github.com/hanglearning/BFRT/blob/main/main.py). The code will scan the folder containing your traffic data (csv) files, recognize each file as a detector, and process the "volume" feature inside the data files as the input to the federated LSTM or GRU models. The logic of data processing is written in [this file](https://github.com/hanglearning/BFRT/blob/main/process_data.py). After loading the data files, the federated learning simulation begins. The code will calculate the maximum possible FL communication rounds and stop after the last possible round. Each detector performs local learning during each round by its own incoming new and portion of its historical traffic data and does FedAvg after it collects all the models from other detectors. A device predicts the traffic volume for the next 5-min interval using the global model produced from FedAvg at the end of this round. See Algorithm 2 and 4 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing) for details. To compare with the performance of the federated models, the simulation also lets each device perform centralized learning and records the prediction by the centralized (baseline) models. See Algorithm 3 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing).

### (1) Prepare the dataset
By default, main.py will look for the traffic data (csv) files inside of a folder named `traffic_data` residing at the root of your Google Drive. If your data files are in a different folder, please provide the absolute path to that folder to the  `-dp` argument (introduced later). Each csv file should have itself named to a detector id, or a detector name, and it should contain at least a "volume" column indicating the traffic volume as float values. For the dataset format, you may refer to the dataset we used to conduct all the experiments in this paper, which is included in [traffic_data.zip](https://github.com/hanglearning/BFRT/blob/main/CCGrid_experimental_results/traffic_data.zip). If you don't have your own datasets, you could use our dataset to reproduce the experimental results.

### (2) Run a new simulation
Sample running command

    $ python /content/traffic_fedavg/main.py -m "gru" -hn 50 -dp "/content/drive/MyDrive/traffic_data" -ml 60 -e 5
Arguments explanation:

1. `-m "gru"` let each detector creates a GRU model 
2. `-hn 50` all GRU models will have 50 hidden neurons in each of the two hidden layers. The GRU model structure is defined in [build_gru.py](https://github.com/hanglearning/BFRT/blob/main/build_gru.py) and LSTM is defined in [build_lstm.py](https://github.com/hanglearning/BFRT/blob/main/build_lstm.py). You may change the model architecture or create your own models following a similar code structure. If you do, please remember to import the new models in main.py.
3. `-dp "/content/drive/MyDrive/traffic_data"`  "dp" denotes "data path". As stated above, the code will look for traffic data files inside of the folder named `traffic_data` residing at the root of your Google Drive by default. If your data files are in a different folder, please provide the absolute path to that folder. You may get this path by right-clicking on the folder inside the File Panel of Google Colab.
4. `-ml`  "ml" denotes "max data length", which refers to the `MaxDataSize` in Algorithm 2 [of the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing).
5. `-e` "e" denotes "epoch". This is the local epoch number a device performs local model update for both federated and central learning.

In a nutshell, this command (also provided in the [Colab file](https://colab.research.google.com/drive/1oW4hnAT8cYOFc11pQilLIkGFplqm0zca#scrollTo=G6BMOEhVnPed)) lets each device (recognized by an individual csv data file in the `-dp`) creates a GRU model with 50 hidden nuerons in its 2 hidden layers, and performs 5 local epochs of learning on this GRU model for both federated and central learning. For FL, the max data length is set to 60.

Each time a new simulation is executed, a log folder containing a file recording the command line arguments (`config_vars.pkl`), some intermediate models (used for resuming the simulation), and the prediction records (`realtime_predicts.pkl`) is created and named by the execution date and time as prefix. For example, the log folder starts with "02212022_160019" means that your execution was run at 16:00:19 Feb 21 2022. This folder, by default, can be found in a folder named "BFRT_logs" residing in your Drive root directory. If you wish to change in which folder it contains the log folders, please provide the folder path to `-lb`. For instance, `-dp "/content/drive/MyDrive/a/"` will then save the logs into a folder named "a" in your Drive's root directory.

### (3) Resume a simulation
Whether using the paid or free version of Google Colab, it usually cannot go through the entire simulation process, unless you have provided very few data points. We have built a function to resume the simulation process in case the simulation is killed. As mentioned above, each new simulation will create a log folder. To resume a simulation, the **only two** arguments you need is `-dp` and `-rp`. A sample command would be

    ! python /content/traffic_fedavg/main.py -dp "/content/drive/MyDrive/traffic_data" -rp "/content/drive/MyDrive/BFRT_logs/02212022_160019"

`rp` stands for "resume path". This is the path to the log folder of the killed simulation. The program will pick up the latest models in the log folder, and continue with the rest of the communication rounds. You may reuse the same command to resume a simulation until all the communication rounds are exhausted.

### All available arguments to [main.py](https://github.com/hanglearning/BFRT/blob/main/main.py)

#### (a) Arguments for System Variables
| Argument  | Type | Default Value | Description |
| ------------- | ------------- |  ------------- | ------------- |
| -dp | str |  '/content/drive/MyDrive/traffic_data/'| Dataset path |
| -lb | str |  "/content/drive/MyDrive/BFRT_logs" |Base folder path to store running logs and h5 model files|
| -pm | int |  0 | This is a boolean value indicating whether to reserve h5 model files from old communication rounds. By default the program only preserves the latest 2 models for each detector for resume purpose. If set to 1, the program will save all the intermediate models, but it would quickly occupy your Google Drive space. If you are using the free 15GB Google Drive, we recommand keep this argument as 0.
| -rp| str| None| Provide the leftover log folder path to continue FL

#### (b) Arguments for Learning
| Argument  | Type | Default Value | Description |
| ------------- | ------------- |  ------------- | ------------- |
| -m | str |  'lstm' | Model to choose - 'lstm' or 'gru' |
| -il | int |  12| Input length for the LSTM/GRU network. See `input_shape` in Algorithm 4 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing). |
| -hm | int |  128| number of neurons in one of each 2 layers |
| -b | int |  1| batch number for training |
| -e | int |  5| local epoch number in per comm round|
| -dp | str |  '/content/drive/MyDrive/traffic_data/'| Dataset path |
| -tp | float |  1.0 | percentage of the data used for learning (we used 0.8 in our experiments, see footnote in Page 5 of [the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing)) |

#### \(c\) Arguments for Federated Learning
| Argument  | Type | Default Value | Description |
| ------------- | ------------- |  ------------- | ------------- |
| -c | int |  None | Specify the number of communication rounds. By default, the program aims to run until all the data is exhausted. |
| -ml | int |  24| Maximum data length for training in each communication round, simulating the memory space a sensor has. See `MaxDataSize` in Algorithm 2 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing). |

## Evaluate Performance by Plots

#### Plotting Experimental Results

The code for plotting the experimental results are provided in the <i>plotting_code</i> folder. [ccgrid_f3_plot_realtime_prediction_all_sensors.py](https://github.com/hanglearning/BFRT/blob/main/plotting_code/ccgrid_f3_plot_realtime_prediction_all_sensors.py) was used to plot Figure 3 and [ccgrid_f4_plot_realtime_errors_interval_with_table.py](https://github.com/hanglearning/BFRT/blob/main/plotting_code/ccgrid_f4_plot_realtime_errors_interval_with_table.py) was used to plot Figure 4 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing). 

#### (1) Plot the real-time prediction curves
Sample command:

    $ python /content/traffic_fedavg/plotting_code/ccgrid_f3_plot_realtime_prediction_all_sensors.py -lp "/content/drive/MyDrive/BFRT_logs/02212022_160019" -pl 24 -r '19912_NB' -row 2 -col 3

Arguments explanation:
`-lp` denotes "log path". Provide the path of the desired log folder  for plotting.
`-pl` denotes "plot last". Provide a number x to plot the last x communication rounds prediction curves. By default, x is set to 24.
`-r` denotes "representative". Specify which detector to be the big representative figure.
`-row` specifies how many rows to arrange the plots of the rest of the detectors
`-col` specifies how many columns to arrange the plots of the rest of the detectors

In a nutshell, this sample command will plot the last 24 rounds of real-time prediction learning curves recorded in the 02212022_160019 simulation for all sensors, with the detector 19912_NB as the representative, and render the rest of the sensors to 2*3 subplots, just like Figure 3 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing). If you have different number of sensors or wish to change which sensor to be the representative, please adjust the command to fit your needs. This code will put the resulting figures in `02212022_160019/plots/realtime_learning_curves_all_sensors`, along with an `errors.txt` containing the aggregated errors for each sensor for the last 24 communication rounds, which are the values we reported in Table I [of the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing).

#### (2) Plot the real-time learning error curves

Sample command:

    $ python /content/traffic_fedavg/plotting_code/ccgrid_f4_plot_realtime_errors_interval_with_table.py -lp "/content/drive/MyDrive/BFRT_logs/02212022_160019" -ei 100 -et "MAE" -yt 150 -row 1 -col 7

Arguments explanation:
`-lp` same as in (1) Plot the real-time prediction curves.
`-ei` denotes "error interval". Provide a number x to this argument to plot error values calculated by segments of x. By default, x is set to 100.
`-et` denotes "error type". Available options are "MAE", "MSE", "RMSE" and "MAPE". We used MAE for our paper and it is by default.
`-yt` stands for "y-axis top value". Specify a number to limit the max value on y-axis. We used 100 and 150 for our paper. 150 is default and it worked well for our MAE error.
`-row` specifies how many rows to arrange the plots for all the detectors
`-col` specifies how many columns to arrange the plots for all the detectors

In a nutshell, this sample command will plot the MAE errors for real-time learning in the 02212022_160019 simulation for all sensors in a 1*7 plot, just like Figure 4 [in the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing). If you have a different number of sensors, please adjust the row and column to fit your needs. This code will put the resulting figures in `02212022_160019/plots/realtime_errors_interval`, along with an `errors.txt` containing the aggregated errors for the desired interval, which were used to plot Figure 4 [of the paper](https://drive.google.com/file/d/1A9HeLCPTS6-ZcAYs8PLyJYM9o2bQy4v2/view?usp=sharing).


#### (3) Reproduce our plots and tables

The logs of the four simulations used for the figures and tables inside of our paper can be found in [this folder](https://github.com/hanglearning/BFRT/tree/main/CCGrid_experimental_results/logs). To reproduce our plots and error tables, please point the two plotting code files to the path of these log folders in the related path to your Google Drive or local directory.

Please raise any issues and concerns you found. Thank you!

## Acknowledgments

(1) Our federated learning code is extended from the LSTM and GRU central learning methods found in this [xiaochus's TrafficFlowPrediction repo](https://github.com/xiaochus/TrafficFlowPrediction). Thank you!

(2) This document was composed for free at [StackEdit](https://stackedit.io/).
