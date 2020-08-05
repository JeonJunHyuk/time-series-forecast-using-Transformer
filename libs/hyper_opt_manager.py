from __future__ import absolute_import, division, print_function

import collections
import os
import shutil
import libs.utils as utils
import numpy as np
import pandas as pd

Deque = collections.deque


class HyperparamOptManager:
    """ optimization + random search + single GPU"""
    def __init__(self,
                 param_ranges,
                 fixed_params,
                 model_folder,
                 override_w_fixed_params=True):

        self.param_ranges = param_ranges

        self._max_tries = 1000
        self.results = pd.DataFrame()
        self.fixed_params = fixed_params
        self.saved_params = pd.DataFrame()

        self.best_score = np.Inf
        self.optimal_name = ""

        # Setup
        # Create folder for saving if it's not there
        self.hyperparam_folder = model_folder
        utils.create_folder_if_not_exists(self.hyperparam_folder)

        self._override_w_fixed_params = override_w_fixed_params

    def load_results(self):
        print("Loading results from", self.hyperparam_folder)

        results_file = os.path.join(self.hyperparam_folder, "results.csv")
        params_file = os.path.join(self.hyperparam_folder, "params.csv")

        if os.path.exists(results_file) and os.path.exists(params_file):
            self.results = pd.read_csv(results_file, index_col=0)
            self.saved_params = pd.read_csv(params_file, index_col=0)

            if not self.results.empty:
                self.results.at["loss"] = self.results.loc["loss"].apply(float)
                self.best_score = self.results.loc["loss"].min()

                is_optimal = self.results.loc["loss"] == self.best_score
                self.optimal_name = self.results.T[is_optimal].index[0]

                return True

        return False

    def _get_params_from_name(self, name):
        params = self.saved_params
        selected_params = dict(params[name])
        if self._override_w_fixed_params:
            for k in self.fixed_params:
                selected_params[k] = self.fixed_params[k]

        return selected_params

    def get_best_params(self):
        optimal_name = self.optimal_name
        return self._get_params_from_name(optimal_name)

    def clear(self):
        shutil.rmtree(self.hyperparam_folder)
        os.makedirs(self.hyperparam_folder)
        self.results = pd.DataFrame()
        self.saved_params = pd.DataFrame()

    def _check_params(self, params):
        valid_fields = list(self.param_ranges.keys()) + list(self.fixed_params.keys())
        invalid_fields = [k for k in params if k not in valid_fields]
        missing_fields = [k for k in valid_fields if k not in params]

        if invalid_fields:
            raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(invalid_fields, valid_fields))
        if missing_fields:
            raise ValueError("Missing Fields Found {} - Valid ones are {}".format(missing_fields, valid_fields))

    def _get_name(self, params):
        self._check_params(params)
        fields = list(params.keys())
        fields.sort()
        return "_".join([str(params[k]) for k in fields])

    def get_next_parameters(self, ranges_to_skip=None):
        if ranges_to_skip is None:
            ranges_to_skip = set(self.results.index)

        if not isinstance(self.param_ranges, dict):
            raise ValueError("Only works for random search!")

        param_range_keys = list(self.param_ranges.keys())
        param_range_keys.sort()

        def _get_next():
            parameters = {
                k: np.random.choice(self.param_ranges[k]) for k in param_range_keys
            }

            # Adds fixed params
            for k in self.fixed_params:
                parameters[k] = self.fixed_params[k]

            return parameters

        for _ in range(self._max_tries):
            parameters = _get_next()
            name = self._get_name(parameters)

            if name not in ranges_to_skip:
                return parameters

        raise ValueError("Exceeded max number of hyperparameter searches!!")

    def update_score(self, parameters, loss, model, info=""):
        if np.isnan(loss):
            loss = np.Inf

        if not os.path.isdir(self.hyperparam_folder):
            os.makedirs(self.hyperparam_folder)

        name = self._get_name(parameters)

        is_optimal = self.results.empty or loss < self.best_score

        if is_optimal:
            if model is not None:
                print("Optimal model found, updating")
                model.save(self.hyperparam_folder)
            self.best_score = loss
            self.optimal_name = name

        self.results[name] = pd.Series({"loss": loss, "info": info})
        self.saved_params[name] = pd.Series(parameters)

        self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
        self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

        return is_optimal


"""gpu 많을 때?"""
class DistributedHyperparamOptManager(HyperparamOptManager):
    def __init__(self,
                 param_ranges,
                 fixed_params,
                 root_model_folder,
                 worker_number,
                 search_iterations=1000,
                 num_iterations_per_worker=5,
                 clear_serialised_params=False):

        max_workers = int(np.ceil(search_iterations / num_iterations_per_worker))

        # Sanity checks
        if worker_number > max_workers:
            raise ValueError(
                "Worker number ({}) cannot be larger than the total number of workers!"
                .format(max_workers))
        if worker_number > search_iterations:
            raise ValueError(
                "Worker number ({}) cannot be larger than the max search iterations ({})!"
                .format(worker_number, search_iterations))

        print("*** Creating hyperparameter manager for worker {}***".format(worker_number))

        hyperparam_folder = os.path.join(root_model_folder, str(worker_number))
        super().__init__(
            param_ranges,
            fixed_params,
            hyperparam_folder,
            override_w_fixed_params=True)

        serialised_ranges_folder = os.path.join(root_model_folder, "hyperparams")
        if clear_serialised_params:
            print("Regenerating hyperparameter list")
            if os.path.exists(serialised_ranges_folder):
                shutil.rmtree(serialised_ranges_folder)

        utils.create_folder_if_not_exists(serialised_ranges_folder)

        self.serialised_ranges_path = os.path.join(
            serialised_ranges_folder, "ranges_{}.csv".format(search_iterations))
        self.hyperparam_folder = hyperparam_folder
        self.worker_num = worker_number
        self.total_search_iterations = search_iterations
        self.num_iterations_per_worker = num_iterations_per_worker
        self.global_hyperparam_df = self.load_serialised_hyperparam_df()
        self.worker_search_queue = self._get_worker_search_queue()

    @property
    def optimisation_completed(self):
        return False if self.worker_search_queue else True

    def get_next_parameters(self):
        param_name = self.worker_search_queue.pop()
        params = self.global_hyperparam_df.loc[param_name, :].to_dict()

        # override가 뭐시여
        for k in self.fixed_params:
            print("Overriding saved {}: {}".format(k, self.fixed_params[k]))
            params[k] = self.fixed_params[k]

        return params

    def load_serialised_hyperparam_df(self):
        print("Loading params for {} search iterations form {}".format(
            self.total_search_iterations, self.serialised_ranges_path))
        if os.path.exists(self.serialised_ranges_folder):
            df = pd.read_csv(self.serialised_ranges_path, index_col=0)
        else:
            print("Unable to load - regenerating search ranges instead")
            df = self.update_serialised_hyperparam_df()

        return df

    def update_serialised_hyperaram_df(self):
        search_df = self._generate_full_hyperparam_df()
        print("Serialising params for {} search iterations to {}".format(
            self.total_search_iterations, self.serialised_ranges_path))
        return search_df

    def _generate_full_hyperparam_df(self):
        np.random.seed(1)
        name_list = []
        param_list = []
        for _ in range(self.total_search_iterations):
            params = super().get_next_parameters(name_list)
            name = self._get_name(params)
            name_list.append(name)
            param_list.append(params)

        full_search_df = pd.DataFrame(param_list, index=name_list)
        return full_search_df

    def clear(self):
        super().clear()
        self.worker_search_queue = self._get_worker_search_queue()

    def load_results(self):
        success = super().load_results()

        if success:
            self.worker_search_queue = self._get_worker_search_queue()

        return success

    def _get_worker_search_queue(self):
        global_df = self.assign_worker_numbers(self.global_hyperparam_df)
        worker_df = global_df[global_df["worker"] == self.worker_num]

        left_overs = [s for s in worker_df.index if s not in self.results.columns]

        return Deque(left_overs)

    def assign_worker_numbers(self, df):
        output = df.copy()
        n = self.total_search_iterations
        batch_size = self.num_iterations_per_worker
        max_worker_num = int(np.ceil(n/batch_size))
        worker_idx = np.concatenate([
            np.tile(i+1, self.num_iterations_per_worker)
            for i in range(max_worker_num)
        ])
        output["worker"] = worker_idx[:len(output)]
        return output
