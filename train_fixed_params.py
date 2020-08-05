import argparse
import datetime as dte
import os

import data_formatters.base
import configs
import libs.hyper_opt_manager
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = configs.ExperimentConfig
HyperparamOptManager = libs.hyper_opt_manager.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):

    num_repeats = 1
    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    # Tensorflow setup
    default_keras_session = tf.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Training from defined parameters for {} ***".format(expt_name))
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params['model_folder'] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params['num_epochs'] = 1
        params['hidden_layer_size'] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print('*** Loading hyperparam manager ***')
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.Inf
    for _ in range(num_repeats):
        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
            tf.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            model.fit()
            attention_weights = model.get_attention(train)
            for k in attention_weights:
                print(k)
                print(attention_weights[k].shape)
            self_att = attention_weights['decoder_self_attn'][0, Ellipsis]
            self_att = self_att.mean(axis=0)  # (114,114). 아래 삼각형 절반만 차 있는.
            static_weights = attention_weights['static_flags'].mean(axis=0)
            # 언제 변하나를 그래프로?
            historical_flags = attention_weights['historical_flags'].mean(axis=0).mean(axis=0)
            future_flags = attention_weights['future_flags'].mean(axis=0).mean(axis=0)

            val_loss = model.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p10_forecast = data_formatter.format_predictions(output_map["p10"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        targets.iloc[:, 2:] = np.exp(targets.iloc[:, 2:])
        p10_forecast.iloc[:, 2:] = np.exp(p10_forecast.iloc[:, 2:])
        p50_forecast.iloc[:, 2:] = np.exp(p50_forecast.iloc[:, 2:])
        p90_forecast.iloc[:, 2:] = np.exp(p90_forecast.iloc[:, 2:])

        targets.to_csv(os.path.join(opt_manager.hyperparam_folder, 'targets.csv'))
        p10_forecast.to_csv(os.path.join(opt_manager.hyperparam_folder, 'p10.csv'))
        p50_forecast.to_csv(os.path.join(opt_manager.hyperparam_folder, 'p50.csv'))
        p90_forecast.to_csv(os.path.join(opt_manager.hyperparam_folder, 'p90.csv'))

        def extract_numerical_data(data):
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.9)

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, "=", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))

if __name__=="__main__":
    def get_args():
        experiment_names = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="nike_tran",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes"

    name, output_folder, use_tensorflow_with_gpu = get_args()

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False
    )