# Fling 的 CLI 使用

在这份文档中，我们将介绍 Fling 中命令行界面（CLI）的使用方式。

## 查看Fling的版本

您可以使用如下命令：

```shell
fling -v
```

或者：

```shell
fling --version
```

以查看您当前已安装的 Fling 版本。同时也可以检查 Fling 是否成功安装。

## fling run 命令

您可以使用 `fling run` 来根据配置文件启动一次实验。以下是一个用法示例：

```shell
fling run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3
```

以下是关于每个参数的含义解释。

- -c / --config：您想要使用的配置文件。
- -p / --pipeline：您想要执行的 pipeline。此 pipeline 应包含在 `fling.pipeline` 中。
- -s / --seed：您想要设置的 seeds 。它用以生成伪随机数，在实验中负责模型参数初始化、各客户端的数据采样等。它可以是一个整数，也可以是类似示例中的多个整数。默认情况下，该 seed 被设定为0。
- -e / --extra_argument：您想要在配置文件的基础上进行的修改。例如 ``-e learn.optimizer.lr:0.1`` ，意味着学习率被设置为0.1。需注意的是：此方法的优先级高于配置文件，意味着配置文件的值将被覆盖。
- -pc / --print_config：如果命令中包含此参数，则将会在命令行中把实验配置打印出来。

## fling create 命令

当在 `fling run` 命令中使用 `-e` 选项传递参数时，由于我们的配置文件通常具有多层结构，参数的键名往往会很复杂。例如，如果我们要修改学习率和训练记录的路径，所需的命令是：

```shell
fling run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3 \
-e learn.optimizer.lr:0.01 \
-e other.logging_path:logging/toy_experiment
```

为了解决这个问题，我们允许用户自定义执行命令，为他们需要的参数键分配更短且更易理解的名称。例如：

```shell
fling create -n my_run \
--argument_map learning_rate:learn.optimizer.lr \
--argument_map log_path:other.logging_path
```

在上面示例中，我们定义了一个名为 `my_run` 的命令。在此命令中，我们使用 `-a/--argument_map` 选项建立了参数键名的映射关系：`learn.optimizer.lr` 映射到 `learning_rate`，而 `other.logging_path` 映射到 `log_path`。接下来，要执行已定义的 `my_run` 命令，您可以使用如下方法：

```shell
fling my_run -c flzoo/mnist/mnist_fedper_cnn_toy_config.py -p personalized_model_pipeline -s 1,2,3 \
--extra_argument learning_rate:0.1 \
--extra_argument log_path:logging/toy_experiment
```

通过这种方法我们可以明显地看出，所传递参数的键名被大大简化了。

注意：

- 对于您自定义在 `-a/--argument_map` 中，但未在 `my_run` 的调用中使用的键，该参数将默认使用原始配置文件中的值。

## fling list 命令

通过在命令行中输入并执行 `fling list` 命令，您可以列出当前环境中所有预定义的命令。

## fling remove 命令

通过在命令行中输入并执行 `fling remove -n CMD_NAME` 命令，您可以在当前环境中移除预定义的命令 `CMD_NAME`。

## fling info 命令

通过在命令行中输入并执行 `fling info CMD_NAME` 命令，您可以列出当前环境中预定义命令 `CMD_NAME` 的详细参数映射信息。
