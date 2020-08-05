import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import pandas as pd

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class NikeTranFormatter(GenericDataFormatter):
    _column_definition = [
        ('ID', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('Transaction', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('Quantity', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Amount Krw', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('FCST_QTY', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('FCST_PER', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('launch', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('event', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('Order Fg', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('Lob Cd', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]

    def __init__(self):
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, vp=None, tp=None):
        print('Formatting train-valid-test splits')
        fixed_params = self.get_fixed_params()
        # extra = fixed_params['num_encoder_steps']
        time_steps = fixed_params['total_time_steps']
        lookback = fixed_params['num_encoder_steps']
        forecast_horizon = time_steps - lookback

        if vp is None:
            vp = 30
        if tp is None:
            tp = 30

        df['date'] = pd.to_datetime(df['date'])

        test_start = df['date'].unique()[-tp]
        test_extra = df['date'].unique()[-(tp+lookback)]
        valid_start = df['date'].unique()[-(tp+vp)]
        valid_extra = df['date'].unique()[-(tp+vp+lookback)]

        df_lists = {'train':[], 'valid':[], 'test':[]}
        for _, sliced in df.groupby('ID'):
            index = sliced['date']
            train = sliced.loc[index < valid_start]
            valid = sliced.iloc[-(tp+vp+lookback):-tp, :]
            test = sliced.iloc[-(tp+lookback):, :]

            sliced_map = {'train': train, 'valid': valid, 'test': test}

            for k in sliced_map:
                tvt = sliced_map[k]
                if len(tvt) >= time_steps:
                    df_lists[k].append(tvt)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}
        print(dfs)

        train = dfs['train']
        self.set_scalers(train, set_real=True)
        self.set_scalers(df, set_real=False)

        def filter_ids(frame):
            identifiers = set(self.identifiers)
            index = frame['ID']
            return frame.loc[index.apply(lambda x: x in identifiers)]

        valid = filter_ids(dfs['valid'])
        test = filter_ids(dfs['test'])
        # train = df.loc[df.index < valid_start]
        # valid = df.loc[(df.index >= valid_extra) & (df.index < test_start)]
        # test = df.loc[df.index > test_extra]

        return (self.transform_inputs(data) for data in [train,valid,test])

    def set_scalers(self, df, set_real=True):
        print('Setting scalers with training data...')
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        if set_real:

            # Extract identifiers in case required
            self.identifiers = list(df[id_column].unique())

            real_inputs = utils.extract_cols_from_data_type(
                DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})

            # Format real scalers
            self._real_scalers = {}
            # for col in real_inputs:
            #     self._real_scalers[col] = (df[col].mean(), df[col].std())
            # self._target_scaler = (df[target_column].mean(), df[target_column].std())
            data = df[real_inputs].values
            self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
            self._target_scaler = sklearn.preprocessing.StandardScaler().fit(df[[target_column]].values)


        else:
            # Format categorical scalers
            categorical_inputs = utils.extract_cols_from_data_type(
                DataTypes.CATEGORICAL, column_definitions,
                {InputTypes.ID, InputTypes.TIME})

            categorical_scalers = {}
            num_classes = []
            if self.identifiers is None:
                raise ValueError('Scale real-valued inputs first!')
            id_set = set(self.identifiers)
            valid_idx = df['ID'].apply(lambda x: x in id_set)
            for col in categorical_inputs:
                # Set all to str so that we don't have mixed integer/string columns
                srs = df[col].apply(str).loc[valid_idx]
                categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                    srs.values)

                num_classes.append(srs.nunique())

            # Set categorical scaler outputs
            self._cat_scalers = categorical_scalers
            self._num_classes_per_cat_input = num_classes

        # self.identifiers = list(df[id_column].unique())
        #
        #
        #
        # categorical_inputs = utils.extract_cols_from_data_type(
        #     DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})
        #
        # categorical_scalers = {}
        # num_classes = []
        # for col in categorical_inputs:
        #     srs = df[col].apply(str)
        #     categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
        #     num_classes.append(srs.nunique())
        #
        # self._cat_scalers = categorical_scalers
        # self._num_classes_per_cat_input = num_classes


    def transform_inputs(self, df):
        output = df.copy()
        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})

        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)


        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        output = output.fillna(0)
        print(output)

        return output

    def format_predictions(self, predictions):
        output = predictions.copy()
        column_names = predictions.columns
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    def get_fixed_params(self):
        fixed_params = {
            'total_time_steps': 7*12+30,
            'num_encoder_steps': 7*12,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }
        return fixed_params

    def get_default_model_params(self):
        model_params = {
            'dropout_rate': 0.5,
            'hidden_layer_size': 80,
            'learning_rate': 0.0001,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 1,
            'stack_size': 1
        }
        return model_params