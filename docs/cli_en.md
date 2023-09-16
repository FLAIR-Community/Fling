# Cli Usage in Fling

In this doc, we will introduce the usage of cli in Fling.

## version

You can use:

```shell
fling -v
```

or command:

```shell
fling --version
```

To list the current of Fling you have installed. It is also a method to check whether you Fling is successfully installed.

## fling run

You can use `fling run` to start an experiment based on a config file. Here is an example:

```shell
fling run -c argzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3
```

Here are explanations about the meaning of each arguments.

- -c / --config: the configuration file you want to start with.
- -p / --pipeline: the pipeline you want to execute. This pipeline should be found in `fling.pipeline`.
- -s / --seed: the seeds you want to run. It can be either a integer or several integers similar to the example. By default, this seed is set to be 0.

## fling create

In the standard implementation of Fling, we use a config file corresponding to a basic setting for each experiment. However, when we need to make extensive adjustments to one or several parameters in the config, we have to create a large number of config files. In this case, the aforementioned basic mode becomes overly cumbersome.

To address this issue, a common practice is to introduce parameter passing in the CLI (command-line interface), such as: `python main.py --lr 0.01`. However, since our configs often present a complex multi-layered structure, simple parameter passing makes it difficult to map the passed parameters to the values in the complex config.

To solve this problem, we introduced the setting of "predefined commands". When defining these commands, we need to explicitly agree on the correspondence between the names of the passed parameters and the keys in the config. For example:

```shell
fling create -n my_run \
--argument_map learning_rate:learn.optimizer.lr \
--argument_map log_path:other.logging_path
```

Then you can use the predefined command to run your own experiments:

```shell
fling my_run -c argzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3 \
--extra_argument learning_rate:0.1
```

Note:

- If the keys defined in the `--argument_map` does not exist in your call on `my_run`, this argument will use the value in the original config file by default.

## fling list

By typing `fling list` in the command line, you can list all the predefined commands in your current environment.

## fling remove

By typing `fling remove -n CMD_NAME` in the command line, you can remove predefined command CMD_NAME in your current environment.

## fling info

By typing `fling info CMD_NAME` in the command line, you can list the detailed argument map of the command CMD_NAME in your environment.