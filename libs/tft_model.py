from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import json
import os
import shutil

import data_formatters.base
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

InputTypes = data_formatters.base.InputTypes

def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def apply_mlp(inputs,
              hidden_size,
              output_size,
              output_activation=None,
              hidden_activation='tanh',
              use_time_distributed=False):
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(inputs)
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size, activation=output_activation))(hidden)
    else:
        hidden = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)(inputs)
        return tf.keras.layers.Dense(output_size, activation=output_activation)(hidden)

# dropout ->
def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation=activation))(x)
        gated_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(x)
    else:
        activation_layer = tf.keras.layers.Dense(hidden_layer_size,activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')(x)

    return tf.keras.layers.Multiply()([activation_layer,
                                       gated_layer]), gated_layer


def add_and_norm(x_list):
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(x)

    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(additional_context)

    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


def get_decoder_mask(self_attn_inputs):
    len_s = tf.shape(self_attn_inputs)[1]
    bs = tf.shape(self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs),1)
    return mask


class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')

    def __call__(self, q, k, v, mask):
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]) / temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+9)*(1. - K.cast(x,'float32')))(mask)
            attn = Add()([attn,mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0],x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention():
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        vs_layer = Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)

        self.attention = ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)


    def __call__(self, q, k, v, mask=None):
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        outputs = K.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = Dropout(self.dropout)(outputs)

        return outputs, attn


class TFTDataCache(object):
    _data_cache = {}

    @classmethod
    def update(cls, data, key):   # data를 data cache의 key 에 저장
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        return key in cls._data_cache


class TemporalFusionTransformer(object):
    def __init__(self, raw_params, use_cudnn=False):
        self.name = self.__class__.__name__
        params = dict(raw_params)

        # Data parameters
        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(str(params['known_categorical_inputs']))

        self.column_definition = params['column_definition']

        # Network params
        self.quantiles = [0.1, 0.5, 0.9]
        self.use_cudnn = use_cudnn # Whether to use GPU optimised LSTM
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])

        # Serialisation options
        self._temp_folder = os.path.join(params['model_folder'],'tmp')
        self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print('*** {} params ***'.format(self.name))
        for k in params:
            print('# {} = {}'.format(k, params[k]))

        # Build model
        self.model = self.build_model()


    def get_tft_embeddings(self, all_inputs):
        time_steps = self.time_steps

        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError('Observation cannot be known a priori!')
        for i in self._input_obs_loc:
            if i in self._static_input_loc:
                raise ValueError('Observation cannot be static!')

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    all_inputs.get_shape().as_list()[-1], self.input_size))

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
            self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        embeddings = []
        for i in range(num_categorical_variables):
            embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer([time_steps]),   # 이건 input layer. Input은 tensor
                tf.keras.layers.Embedding(
                    self.category_counts[i],   # input_dim
                    embedding_sizes[i],        # output_dim
                    input_length=time_steps,
                    dtype=tf.float32)
            ])
            embeddings.append(embedding)

        regular_inputs, categorical_inputs = all_inputs[:, :, :num_regular_variables], all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            embeddings[i](categorical_inputs[Ellipsis, i])
            for i in range(num_categorical_variables)
        ]

        # static inputs
        if self._static_input_loc:
            static_inputs = [tf.keras.layers.Dense(self.hidden_layer_size)(
                regular_inputs[:, 0, i:i+1]) for i in range(num_regular_variables)
                            if i in self._static_input_loc] \
                + [embedded_inputs[i][:, 0, :]
                   for i in range(num_categorical_variables)
                   if i + num_regular_variables in self._static_input_loc]
            static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            return tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.hidden_layer_size))(x)

        # Targets
        obs_inputs = tf.keras.backend.stack([convert_real_to_embedding(regular_inputs[Ellipsis, i:i+1])
                                             for i in self._input_obs_loc], axis=-1)

        # Observed (a priori unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i not in self._input_obs_loc:
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i:i+1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = tf.keras.backend.stack(unknown_inputs+ wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # A priori kwnon inputs
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis,i:i+1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i] for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]
        known_combined_layer = tf.keras.backend.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1
        )
        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    """ input_type 에 대한 column 이름 return"""
    """ 0-5 숫자가 들어가면 column 이름이 나온드아"""
    def _get_single_col_by_type(self, input_type):
        return utils.get_single_col_by_input_type(input_type, self.column_definition)

    def training_data_cached(self):
        return TFTDataCache.contains('train') and TFTDataCache.contains('valid')

    def cache_batched_data(self, data, cache_key, num_samples=-1):
        # data가 raw data 중 train, cache_key가 'train'
        if num_samples > 0:
            TFTDataCache.update(self._batch_sampled_data(data, max_samples=num_samples), cache_key)
        else:
            TFTDataCache.update(self._batch_data(data), cache_key)
        print('Cached data "{}" updated'.format(cache_key))

    def _batch_sampled_data(self, data, max_samples):
        if max_samples < 1:
            raise ValueError('illegal number of samples specified! samples={}'.format(max_samples))

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)

        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [(identifier, self.time_steps+i)
                                            for i in range(num_entries - self.time_steps+1)]
            split_data_map[identifier] = df

        inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if (i + 1 % 1000) == 0:
                print(i+1, 'of', max_samples, 'sample done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.time_steps:start_idx]
            inputs[i, :, :] = sliced[input_cols]
            outputs[i, :, :] = sliced[[target_col]]
            time[i, :, 0] = sliced[time_col]
            identifiers[i, :, 0] = sliced[id_col]

        sampled_data = {
            'inputs': inputs,
            'outputs': outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(outputs[:, self.num_encoder_steps:, :]),
            'time': time,
            'identifier': identifiers
        }

        return sampled_data

    def _batch_data(self, data):
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps   # lookback + forecast
            x = input_data.values
            if time_steps >= lags:
                return np.stack([x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)
            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data_map = {}
        for _, sliced in data.groupby(id_col):
            col_mappings = {
                'identifier': [id_col],
                'time': [time_col],
                'outputs': [target_col],
                'inputs': input_cols
            }

            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())
                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        data_map['outputs'] = data_map['outputs'][:, self.num_encoder_steps:, :]

        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)

        return data_map

    def _get_active_locations(self, x):
        return (np.sum(x, axis=-1) > 0.0) * 1.0

    def _build_base_graph(self):
        # Size definitions
        time_steps = self.time_steps            # 114
        combined_input_size = self.input_size    # 15
        encoder_steps = self.num_encoder_steps   # 84

        # inputs
        all_inputs = tf.keras.layers.Input(
            shape=(time_steps,combined_input_size,))

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(all_inputs)

        # isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = concat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], axis=-1)
        else:
            historical_inputs = concat([
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], axis=-1)

        # isolate only known future inputs
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        """ applies variable selection network to static inputs"""
        def static_combine_and_mask(embedding):
            _, num_static, _ = embedding.get_shape().as_list() # Add temporal features
            flatten = tf.keras.layers.Flatten()(embedding)
            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None)

            sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = K.expand_dims(sparse_weights, axis=-1)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(
                    embedding[:, i:i+1, :], #(,1,80)
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False)
                trans_emb_list.append(e)

            transformed_embedding = concat(trans_emb_list, axis=1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])

            static_vec = K.sum(combined, axis=1)

            return static_vec, sparse_weights

        static_encoder, static_weights = static_combine_and_mask(static_inputs)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        def lstm_combine_and_mask(embedding):
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()
            flatten = K.reshape(embedding, [-1, time_steps, embedding_dim * num_inputs])
            expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)

            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True)

            sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2)

            # Non-linear Processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[Ellipsis, i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True)
                trans_emb_list.append(grn_output)

            transformed_embedding = stack(trans_emb_list, axis=-1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])
            temporal_ctx = K.sum(combined, axis=-1)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_flags, _ = lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = lstm_combine_and_mask(future_inputs)

        # LSTM layer. returns lstm cell initialized with default parameters.
        def get_lstm(return_state):
            if self.use_cudnn:
                lstm = tf.keras.layers.CuDNNLSTM(
                    self.hidden_layer_size,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                )
            else:
                lstm = tf.keras.layers.LSTM(
                    self.hidden_layer_size,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True
                )
            return lstm

        history_lstm, state_h, state_c = get_lstm(return_state=True)(
            historical_features, initial_state=[static_context_state_h,
                                                static_context_state_c]
        )

        future_lstm = get_lstm(return_state=False)(
            future_features, initial_state=[state_h, state_c])

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection
        input_embeddings = concat([historical_features, future_features], axis=1)

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        # Static enrichment layres
        expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True)

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate
        )

        mask = get_decoder_mask(enriched)
        x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)  # q, k, v 들어가네.

        x, _ = apply_gating_layer(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            activation=None)

        x = add_and_norm([x, enriched])

        #Nonlinear processing on outputs
        decoder = gated_residual_network(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True)

        # Final skip connection
        decoder, _ = apply_gating_layer(
            decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # Attention components for explainability
        attention_components = {
            'decoder_self_attn': self_att,   # Temporal attention weights
            'static_flags': static_weights[Ellipsis, 0],   # Static variale selection weights
            'historical_flags': historical_flags[Ellipsis, 0, :],   # Variable selection weights of past inputs
            'future_flags': future_flags[Ellipsis, 0, :]    # Variable selection weights of future inputs
        }

        return transformer_layer, all_inputs, attention_components

    def build_model(self):
        with tf.variable_scope(self.name):
            transformer_layer, all_inputs, attention_components = self._build_base_graph()

            outputs = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.output_size * len(self.quantiles)))(transformer_layer[Ellipsis, self.num_encoder_steps:, :])

            self._attention_components = attention_components

            adam = tf.keras.optimizers.Adam(
                lr=self.learning_rate, clipnorm=self.max_gradient_norm)

            model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

            print(model.summary())

            valid_quantiles = self.quantiles
            output_size = self.output_size

            class QuantileLossCalculator(object):
                def __init__(self, quantiles):
                    self.quantiles = quantiles

                def quantile_loss(self, a, b):  # a가 target, b가 prediction
                    quantiles_used = set(self.quantiles)

                    loss = 0.
                    for i, quantile in enumerate(valid_quantiles):
                        if quantile in quantiles_used:
                            loss += utils.tensorflow_quantile_loss(
                                a[Ellipsis, output_size * i:output_size*(i+1)],
                                b[Ellipsis, output_size * i:output_size*(i+1)], quantile)
                    return loss

            quantile_loss = QuantileLossCalculator(valid_quantiles).quantile_loss

            model.compile(loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal')

            self._input_placeholder = all_inputs

        return model

    def fit(self, train_df=None, valid_df=None):
        print('*** Fitting {} ***'.format(self.name))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_keras_saved_path(self._temp_folder),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        print('Getting batched_data')
        if train_df is None:
            print('Using cached training data')
            train_data = TFTDataCache.get('train')
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            print('Using cached validation data')
            valid_data = TFTDataCache.get('valid')
        else:
            valid_data = self._batch_data(valid_df)

        print('Using keras standard fit')

        def _unpack(data):
            return data['inputs'], data['outputs'], self._get_active_locations(data['active_entries'])

        # Unpack without sample weights
        data, labels, active_flags = _unpack(train_data)
        val_data, val_labels, val_flags = _unpack(valid_data)

        all_callbacks = callbacks

        self.model.fit(
            x=data,
            y=np.concatenate([labels, labels, labels], axis=-1),
            sample_weight=active_flags,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=(val_data, np.concatenate([val_labels, val_labels, val_labels], axis=-1), val_flags),
            callbacks=all_callbacks,
            shuffle=True,  # data가 아니라 batch 를 섞는 거?
            use_multiprocessing=True,
            workers=self.n_multiprocessing_workers)

        tmp_checkpont = self.get_keras_saved_path(self._temp_folder)
        if os.path.exists(tmp_checkpont):
            self.load(
                self._temp_folder,
                use_keras_loadings=True)
        else:
            print('Cannot load from {}, skipping ...'.format(self._temp_folder))

    def evaluate(self, data=None, eval_metric='loss'):
        if data is None:
            print('Using cached validation data')
            raw_data = TFTDataCache.get('valid')
        else:
            raw_data = self._batch_data(data)

        inputs = raw_data['inputs']
        outputs = raw_data['outputs']
        active_entries = self._get_active_locations(raw_data['active_entries'])

        metric_values = self.model.evaluate(
            x=inputs,
            y=np.concatenate([outputs, outputs, outputs], axis=-1),
            sample_weight=active_entries,
            workers=16,
            use_multiprocessing=True)

        metrics = pd.Series(metric_values, self.model.metrics_names)

        return metrics[eval_metric]

    def predict(self, df, return_targets=False):
        data = self._batch_data(df)

        inputs = data['inputs']
        time = data['time']
        identifier = data['identifier']
        outputs = data['outputs']

        combined = self.model.predict(
            inputs,
            workers=16,
            use_multiprocessing=True,
            batch_size=self.minibatch_size)

        # Format output_csv
        if self.output_size != 1:
            raise NotImplementedError('Current version only supports 1D targets!')

        def format_outputs(prediction):
            flat_prediction = pd.DataFrame(
                prediction[:, :, 0],
                columns=['t+{}'.format(i) for i in range(self.time_steps - self.num_encoder_steps)]
            )
            cols = list(flat_prediction.columns)
            flat_prediction['forecast_time'] = time[:, self.num_encoder_steps-1, 0]
            flat_prediction['identifier'] = identifier[:, 0, 0]

            # Arrange in order
            return flat_prediction[['forecast_time', 'identifier'] + cols]

        # Extract predictions for each quantile into different entries
        process_map = {
            'p{}'.format(int(q * 100)):
            combined[Ellipsis, i * self.output_size:(i+1) * self.output_size]
            for i, q in enumerate(self.quantiles)
        }

        if return_targets:
            # Add targets if relevant
            process_map['targets'] = outputs
        return {k: format_outputs(process_map[k]) for k in process_map}

    def get_attention(self, df):
        data = self._batch_data(df)
        inputs = data['inputs']
        identifiers = data['identifier']
        time = data['time']

        def get_batch_attention_weights(input_batch):
            input_placeholder = self._input_placeholder
            attention_weights = {}
            for k in self._attention_components:
                attention_weight = tf.keras.backend.get_session().run(
                    self._attention_components[k],
                    {input_placeholder: input_batch.astype(np.float32)}
                )
                attention_weights[k] = attention_weight
            return attention_weights

        # Compute number of batches
        batch_size = self.minibatch_size
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches += 1

        # Split up inputs into batches
        batched_inputs = [
            inputs[i * batch_size:(i+1) * batch_size, Ellipsis]
            for i in range(num_batches)
        ]

        # Get attention weights, while avoiding large memory increases
        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in batched_inputs
        ]
        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])

            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:, 0, 0]
        attention_weights['time'] = time[:, :, 0]

        return attention_weights

    # Serialisation.
    def reset_temp_folder(self):
        print('Resetting temp folder...')
        utils.create_folder_if_not_exists(self._temp_folder)
        shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)

    def get_keras_saved_path(self, model_folder):
        return os.path.join(model_folder, '{}.check'.format(self.name))

    def save(self, model_folder):
        utils.save(
            tf.keras.backend.get_session(),
            model_folder,
            cp_name=self.name,
            scope=self.name)

    def load(self, model_folder, use_keras_loadings=False):
        if use_keras_loadings:
            serialisation_path = self.get_keras_saved_path(model_folder)
            print('Loading model from {}'.format(serialisation_path))
            self.model.load_weights(serialisation_path)
        else:
            utils.load(
                tf.keras.backend.get_session(),
                model_folder,
                cp_name=self.name,
                scope=self.name)

    @classmethod
    def get_hyperparm_choices(cls):
        return {
            'dropout_rate': [0.1, 0.2, 0.3],
            'hidden_layer_size': [10, 20, 40, 80],
            'minibatch_size': [64, 128],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'max_gradient_norm': [0.01, 1.0, 100.0],
            'num_heads': [1,4],
            'stack_size': [1],
        }