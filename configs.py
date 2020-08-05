import os

import data_formatters.favorita
import data_formatters.volatility
import data_formatters.nike
import data_formatters.nike_tran


class ExperimentConfig(object):
    default_experiments = ['favorita', 'volatility', 'nike', 'nike_tran', 'nike_traffic_0706']

    def __init__(self, experiment='favorita', root_folder=None):
        if experiment not in self.default_experiments:
            raise ValueError('unrecognized experiment={}'.format(experiment))

        if root_folder is None:
            root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
            print('Using root folder {}'.format(root_folder))

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, 'data', experiment)
        self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
        self.results_folder = os.path.join(root_folder, 'results', experiment)

        for relevant_directory in [
            self.root_folder, self.data_folder, self.model_folder, self.results_folder
        ]:
            if not os.path.exists(relevant_directory):
                os.makedirs(relevant_directory)

    @property
    def data_csv_path(self):
        csv_map = {
            'favorita': 'favorita_consolidated.csv',
            'volatility': 'formatted_omi_vol.csv',
            'nike': 'nike_first.csv',
            'nike_traffic_0706': 'nike_traffic_0706_processed.csv',
            'nike_tran': 'nike_tran_processed.csv'
        }
        return os.path.join(self.data_folder, csv_map[self.experiment])

    @property
    def hyperparam_iterations(self):
        # if self.experiment == 'nike':
        #     return 1000
        # else:
        return 240 if self.experiment == 'nike' else 60

    def make_data_formatter(self):
        data_formatters_class = {
            'favorita': data_formatters.favorita.FavoritaFormatter,
            'volatility': data_formatters.volatility.VolatilityFormatter,
            'nike': data_formatters.nike.NikeFormatter,
            'nike_tran': data_formatters.nike_tran.NikeTranFormatter,
            'nike_traffic_0706': data_formatters.nike_traffic_0706.NikeTraffic0706Formatter
        }

        return data_formatters_class[self.experiment]()