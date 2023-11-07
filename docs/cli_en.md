# CLI Usage in Fling

In this doc, we will introduce the usage of cli in Fling.

## Check the version of Fling

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
fling run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3
```

Here are explanations about the meaning of each argument.

- -c / --config: the configuration file you want to start with.
- -p / --pipeline: the pipeline you want to execute. This pipeline should be found in `fling.pipeline`.
- -s / --seed: the seeds you want to run. It is used to generate pseudo-random numbers, which is responsible for tasks such as model parameter initialization and sampling data for each client during experiments. It can be either an integer or several integers similar to the example. By default, this seed is set to be 0.
- -e / --extra_argument: the modification you want to do based on the configuration file. For example, ``-e learn.optimizer.lr:0.1`` , means that the learning rate is set to 0.1. Note that the priority of this method is higher than the configuration file, which means that the value of configuration file will be over-written.
- -pc / --print_config: if this flag is included, the exp config will be printed in the command line.

## fling create

When using the `-e` option in the `fling run` method to pass parameters, the keys for the parameters tend to be complex because our configuration files often have a multi-level structure. For example, if we want to modify the learning rate and the log path, the required command is:

```shell
fling run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3 \
-e learn.optimizer.lr:0.01 \
-e other.logging_path:logging/toy_experiment
```

To solve this problem, we allow users to customize the execution command, assigning a shorter and more understandable name to the parameter key they need. For example:

```shell
fling create -n my_run \
--argument_map learning_rate:learn.optimizer.lr \
--argument_map log_path:other.logging_path
```

In this example, we defined a command called `my_run`. In this command, we established a mapping relationship for parameter keys using the `-a/--argument_map` option: `learn.optimizer.lr` is mapped to `learning_rate` and `other.logging_path` is mapped to `log_path`. Next, to execute the defined `my_run` command, you can use the following method:

```shell
fling my_run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3 \
--extra_argument learning_rate:0.1 \
--extra_argument log_path:logging/toy_experiment
```

This clearly shows that by doing so, the key for passing parameters has been greatly simplified.

Note:

- If the keys defined in the `-a/--argument_map` does not exist in your call on `my_run`, this argument will use the value in the original config file by default.

## fling list

By typing `fling list` in the command line, you can list all the predefined commands in your current environment.

## fling remove

By typing `fling remove -n CMD_NAME` in the command line, you can remove predefined command `CMD_NAME` in your current environment.

## fling info

By typing `fling info CMD_NAME` in the command line, you can list the detailed argument map of the command `CMD_NAME` in your environment.