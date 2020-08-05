import tensorflow as tf
import pandas as pd
import os
import json
raw_data = pd.read_csv('./outputs/data/nike_tran/nike_tran_processed.csv', index_col=0)

import data_formatters.nike_tran

formatter = data_formatters.nike_tran.NikeTranFormatter()
train, valid, test = formatter.split_data(raw_data)

fixed_params = formatter.get_experiment_params()
params = formatter.get_default_model_params()
params['model_folder'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),'heart_test')

time_steps = int(fixed_params['total_time_steps'])
input_size = int(fixed_params['input_size'])
output_size = int(fixed_params['output_size'])
category_counts = json.loads(str(fixed_params['category_counts']))
n_multiprocessing_workers = int(fixed_params['multiprocessing_workers'])
# Relevant indices for TFT
_input_obs_loc = json.loads(str(fixed_params['input_obs_loc']))
_static_input_loc = json.loads(str(fixed_params['static_input_loc']))
_known_regular_input_idx = json.loads(str(fixed_params['known_regular_inputs']))
_known_categorical_input_idx = json.loads(str(fixed_params['known_categorical_inputs']))
column_definition = fixed_params['column_definition']
# Network params
quantiles = [0.1, 0.5, 0.9]
use_cudnn = False # Whether to use GPU optimised LSTM
hidden_layer_size = int(params['hidden_layer_size'])
dropout_rate = float(params['dropout_rate'])
max_gradient_norm = float(params['max_gradient_norm'])
learning_rate = float(params['learning_rate'])
minibatch_size = int(params['minibatch_size'])
num_epochs = int(fixed_params['num_epochs'])
early_stopping_patience = int(fixed_params['early_stopping_patience'])
num_encoder_steps = int(fixed_params['num_encoder_steps'])
num_stacks = int(params['stack_size'])
num_heads = int(params['num_heads'])

import numpy as np
import data_formatters.base

InputTypes = data_formatters.base.InputTypes

def _batch_data(data):
    def _batch_single_entity(input_data):
        time_steps_tmp = len(input_data)
        lags = time_steps  # lookback + forecast
        x = input_data.values
        if time_steps_tmp >= lags:
            return np.stack([x[i:time_steps_tmp - (lags - 1) + i, :] for i in range(lags)], axis=1)
        else:
            return None

    def _get_single_col_by_type(input_type, column_definition):
        l = [tup[0] for tup in column_definition if tup[2] == input_type]
        return l[0]

    id_col = _get_single_col_by_type(InputTypes.ID, column_definition)
    time_col = _get_single_col_by_type(InputTypes.TIME, column_definition)
    target_col = _get_single_col_by_type(InputTypes.TARGET, column_definition)
    input_cols = [
        tup[0]
        for tup in column_definition
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

    data_map['outputs'] = data_map['outputs'][:, num_encoder_steps:, :]

    active_entries = np.ones_like(data_map['outputs'])
    if 'active_entries' not in data_map:
        data_map['active_entries'] = active_entries
    else:
        data_map['active_entries'].append(active_entries)

    return data_map

train = _batch_data(train)
valid = _batch_data(valid)


_data_cache = {}
_data_cache['train'] = train
_data_cache['valid'] = valid

train_data = _data_cache['train'].copy()
valid_data = _data_cache['valid'].copy()

data = train_data['inputs']
labels = train_data['outputs']
active_flags = (np.sum(train_data['active_entries'], axis=-1) >0.0)*1.0

val_data, val_labels, val_flags = valid_data['inputs'], valid_data['outputs'], (np.sum(valid_data['active_entries'],axis=-1) > 0.0)*1.0

all_inputs = tf.keras.layers.Input(shape=(time_steps, input_size,))
# shape=(,114,15) tensor

num_categorical_variables = len(category_counts)
num_regular_variables = input_size - num_categorical_variables

embedding_sizes = [hidden_layer_size for i, size in enumerate(category_counts)]
embedding_sizes   # [80, 80, 80, 80, 80, 80, 80, 80, 80]

embeddings = []
for i in range(num_categorical_variables):
    embedding = tf.keras.Sequential([   # output=(,114,80), param=category_counts[i]*80
        tf.keras.layers.InputLayer([time_steps]),
        tf.keras.layers.Embedding(
            category_counts[i],   # input_dim = [7, 31, 52, 12, 4, 1, 1, 2, 3]
            embedding_sizes[i],   # output_dim = 80
            input_length=time_steps,
            dtype=tf.float32
        )
    ])
    embeddings.append(embedding)

regular_inputs = all_inputs[:, :, :num_regular_variables]  # (,114,6)
categorical_inputs = all_inputs[:, :, num_regular_variables:]  # (,114,9)

# categorical_inputs의 column들 각각 전부 (,114,80)으로 변환
embedded_inputs = [
    embeddings[i](categorical_inputs[Ellipsis, i])
    for i in range(num_categorical_variables)
]

# static은 어떻게 embedding?
# regular, categorical 나눠서. tensor로 처리해서 stack.
# regular는 Dense(80)으로, categorical이랑 + 하면 list에 append.
# regular_inputs[:,0,0:1]은 (,1).
# embedded_inputs[0][:,0,:]은 (,80).
# static_inputs list 안에 (,80)이 나열됨.
# 그걸 axis=1로 stack 하면 (,n,80)
# 지금은 static인 게 embedded_inputs 중에서만 두 개라 (,2,80)
static_inputs = [tf.keras.layers.Dense(hidden_layer_size)(regular_inputs[:,0,i:i+1])
                 for i in range(num_regular_variables) if i in _static_input_loc] \
    + [embedded_inputs[i][:,0,:] for i in range(num_categorical_variables)
       if i + num_regular_variables in _static_input_loc]
static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

# target embedding
# (114,1)
regular_inputs[Ellipsis, 0:1]
# (114,80)
tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size))(regular_inputs[Ellipsis, 2:3])
# (114,80,1)
obs_inputs = tf.keras.backend.stack([tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size))\
     (regular_inputs[Ellipsis, i:i+1]) for i in _input_obs_loc], axis=-1)

# observed inputs embedding
# 이것도 categorical, regular 나눠서
# categorical 먼저
wired_embeddings = []
for i in range(num_categorical_variables):
    # categorical_inputs 중에 known, target 뺀 거.
    if i not in _known_categorical_input_idx and i not in _input_obs_loc:
        e = embeddings[i](categorical_inputs[:, :, i])
        # (,114) -> InputLayer([time_steps]) -> Embedding(cat_counts,80,...)
        # -> (114, 80)
        wired_embeddings.append(e)
# 없어

# 밑에가 regular
unknown_inputs = []
for i  in range(regular_inputs.shape[-1]):
    # regular_inputs 중에 known, target 뺀 거.
    if i not in _known_regular_input_idx and i not in _input_obs_loc:
        e = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size))\
            (regular_inputs[Ellipsis, i:i+1])
        # (,114,1) -> Dense(80) -> (,114,80) -> TD 거쳐도 (,114,80)
        unknown_inputs.append(e)
# unknown_inputs 엔  (,114,80) 이 두 개

unknown_inputs = tf.keras.backend.stack(unknown_inputs + wired_embeddings, axis=-1)
# (,114,80) 두 개가 stack axis=-1 돼서 (,114,80,2)

# Priori known inputs embedding
# (,114,1) -> (,114,80)
known_regular_inputs = [
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size))(regular_inputs[Ellipsis,i:i+1])
    for i in _known_regular_input_idx
    if i not in _static_input_loc
]
# (,114,80) 3개
known_categorical_inputs = [
    embedded_inputs[i] for i in _known_categorical_input_idx
    if i + num_regular_variables not in _static_input_loc
]
# (,114,80) 7개

known_combined_layer = tf.keras.backend.stack(
    known_regular_inputs + known_categorical_inputs, axis=-1
)
# (,114,80,10)

# 이제 embedding 끝
# unknown_inputs = (,114,80,2) tensor
# known_combined_layer = (,114,80,10) tensor
# obs_inputs = (,114,80,1) tensor
# static_inputs = (,2,80) tensor

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

# historical & future 나누기
# indexing 이 왜 이렇게 되냐... 앞에서부터 세고 뒤에껀 생략 가능인가봄.
historical_inputs = concat([
    unknown_inputs[:, :num_encoder_steps, :],   # (,84,80,2)
    known_combined_layer[:, :num_encoder_steps, :],   # (,84,80,10)
    obs_inputs[:, :num_encoder_steps, :],   # (,84,80,1)
], axis=-1)
# historical_inputs=(,84,80,13).

future_inputs = known_combined_layer[:, num_encoder_steps:, :]
# future_inputs = (,30,80,10)

# static variable selection network

_, num_static, _ = static_inputs.get_shape().as_list()  # [None, 2, 80]
# 그래서 num_static이 2
flatten = tf.keras.layers.Flatten()(static_inputs)   # (, 160)

# 여기서부터 잠깐 grn
linear = Dense(num_static) # output_dim = 2
skip = linear(flatten)  # (,160) -> (,2)

hidden = tf.keras.layers.Dense(hidden_layer_size)(flatten)  # (,160) -> (,80)
# 여기선 additional_context가 없네. 있으면 + 해주는데? +하면 어떻게 되나.
hidden = tf.keras.layers.Activation('elu')(hidden)
hidden = tf.keras.layers.Dense(hidden_layer_size)(hidden)  # (,80) -> (,80) 이거 왜 하는거지.

# grn 안에 apply_gating_layer
hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
activation_layer = tf.keras.layers.Dense(num_static)(hidden)   # (,2)
gated_layer = tf.keras.layers.Dense(num_static,activation='sigmoid')(hidden)  # (,2)
gating_layer = tf.keras.layers.Multiply()([activation_layer, gated_layer])
gate = gated_layer

mlp_outputs = LayerNorm()(Add()([skip, gating_layer])) #(,2)
# 여기까지 grn. output이 mlp_outputs
# 아직 static variable selection network

sparse_weights = tf.keras.layers.Activation('softmax')(mlp_outputs) #(,2)
sparse_weights = K.expand_dims(sparse_weights, axis=-1) #(,2,1)

def linear_layer(size,
                 activation=None,
                 use_bias=True,
                 use_time_distributed=False):
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear

def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size,activation=activation))(x)
        gated_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(x)
    else:
        activation_layer = tf.keras.layers.Dense(hidden_layer_size,activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid')(x)

    return tf.keras.layers.Multiply()([activation_layer, gated_layer]), gated_layer


def add_and_norm(x_list):
    return LayerNorm()(Add()(x_list))

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
        return add_and_norm([skip,gating_layer]), gate
    else:
        return add_and_norm([skip,gating_layer])

trans_emb_list = []  # (,1,80) 두 개
for i in range(num_static):
    e = gated_residual_network(   # e=(,1,80)
        static_inputs[:,i:i+1,:], #(,1,80)
        hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=False)
    trans_emb_list.append(e)

transformed_embedding = concat(trans_emb_list, axis=1)  #(,2,80)
combined = tf.keras.layers.Multiply()([sparse_weights,transformed_embedding])
# (,2,1) 랑 (,2,80)을 element-wise multiply
static_encoder = K.sum(combined, axis=1)   #(,80)
static_weights = sparse_weights    #(,2,1)

static_context_variable_selection = gated_residual_network(   #(,80)
    static_encoder,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=False)
static_context_enrichment = gated_residual_network(   #(,80)
    static_encoder,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=False)
static_context_state_h = gated_residual_network(   #(,80)
    static_encoder,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=False)
static_context_state_c = gated_residual_network(   #(,80)
    static_encoder,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=False)

# historical, future_inputs 를 lstm_combine_and_mask에 넣는다.
_, time_steps, embedding_dim, num_inputs = historical_inputs.get_shape().as_list()
# 84, 80, 13

flatten = K.reshape(historical_inputs, [-1, time_steps, embedding_dim * num_inputs])
# (,84, 1040)

expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)
# (,1,80)

# (,84,13), (,84,13)
mlp_outputs, static_gate = gated_residual_network(
    flatten,  # (84,1040)
    hidden_layer_size, # 80
    output_size=num_inputs, # 13
    dropout_rate=dropout_rate,
    use_time_distributed=True,
    additional_context=expanded_static_context, #(1,80)
    return_gate=True)

sparse_weights_lstm = tf.keras.layers.Activation('softmax')(mlp_outputs)
sparse_weights_lstm = tf.expand_dims(sparse_weights_lstm, axis=2)
#(,84,1,13)

trans_emb_list = []
for i in range(num_inputs):
    grn_output = gated_residual_network(   # (,84,80)
        historical_inputs[Ellipsis,i], #(,84,80,13) 중에 (84,80)
        hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=True)
    trans_emb_list.append(grn_output)

transformed_embedding = stack(trans_emb_list, axis=-1)  # (,84,80,13)

combined = tf.keras.layers.Multiply()([sparse_weights_lstm, transformed_embedding])
# (,84,80,13)

temporal_ctx = K.sum(combined, axis=-1) #(,84,80)

historical_features = temporal_ctx  #(,84,80)
historical_flags = sparse_weights_lstm   #(, 30, 1, 10)


# 이번엔 future_inputs 차례
_, time_steps, embedding_dim, num_inputs = future_inputs.get_shape().as_list()
# , 30, 80, 10

flatten = K.reshape(future_inputs, [-1, time_steps, embedding_dim * num_inputs])
# (,30, 800)

expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)
# (,1,80)

# (,30,10), (,30,10)
mlp_outputs, static_gate = gated_residual_network(
    flatten,  # (,30,800)
    hidden_layer_size, # 80
    output_size=num_inputs, # 10
    dropout_rate=dropout_rate,
    use_time_distributed=True,
    additional_context=expanded_static_context, #(1,80)
    return_gate=True)

sparse_weights_lstm = tf.keras.layers.Activation('softmax')(mlp_outputs)
sparse_weights_lstm = tf.expand_dims(sparse_weights_lstm, axis=2)
#(,30,1,10)

trans_emb_list = []
for i in range(num_inputs):
    grn_output = gated_residual_network(   # (,30,80)
        future_inputs[Ellipsis,i], #(,30,80,10) 중에 (30,80)
        hidden_layer_size,
        dropout_rate=dropout_rate,
        use_time_distributed=True)
    trans_emb_list.append(grn_output)

transformed_embedding = stack(trans_emb_list, axis=-1)  # (,30,80,10)

combined = tf.keras.layers.Multiply()([sparse_weights_lstm, transformed_embedding])
# (,30,80,10)

temporal_ctx = K.sum(combined, axis=-1) #(,30,80)

future_features = temporal_ctx   #(,30,80)
future_flags = sparse_weights_lstm   #(30,1,10)

# (,84,80), (,80), (,80) -> (,84,80), (,80), (,80). 들어간 모양 그대로 나옴
history_lstm, state_h, state_c = tf.keras.layers.LSTM(
    hidden_layer_size,
    return_sequences=True,
    return_state=True,
    stateful=False,
    activation='tanh',
    recurrent_activation='sigmoid',
    recurrent_dropout=0,
    unroll=False,
    use_bias=True
)(historical_features, initial_state=[static_context_state_h,
                                      static_context_state_c])

# (,30,80),(,80),(,80) -> (,30,80)
future_lstm = tf.keras.layers.LSTM(
    hidden_layer_size,
    return_sequences=True,
    return_state=False,
    stateful=False,
    activation='tanh',
    recurrent_activation='sigmoid',
    recurrent_dropout=0,
    unroll=False,
    use_bias=True
)(future_features, initial_state=[state_h, state_c])

# (,84,80), (30,80) -> (,114,80)
lstm_layer = concat([history_lstm,future_lstm], axis=1)

# gated skip connection
# (,114,80)
input_embeddings = concat([historical_features, future_features], axis=1)

# (,114,80)
lstm_layer, _ = apply_gating_layer(
    lstm_layer, hidden_layer_size, dropout_rate, activation=None)

# (,114,80)
temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

# (,1,80)
expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
enriched, _ = gated_residual_network(   # (,114,80)
    temporal_feature_layer, # (,114,80)
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=True,
    additional_context=expanded_static_context,  #(,1,80)
    return_gate=True)

# Attention 시작
n_head = num_heads
d_model = hidden_layer_size
d_k = d_v = d_model // n_head   # 80 // 1
qs_layers = []
ks_layers = []
vs_layers = []
vs_layer = Dense(d_v, use_bias=False)  # output_dim = 80

# 칸 만들어놓기?
for _ in range(n_head):
    qs_layers.append(Dense(d_k, use_bias=False))
    ks_layers.append(Dense(d_k, use_bias=False))
    vs_layers.append(vs_layer) # 얜 head만큼 만들 필요 없응게.

w_o = Dense(d_model, use_bias=False)

# enriched = (,114,80)
len_s = tf.shape(enriched)[1]
bs = tf.shape(enriched)[:1]
mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)   #(None, None, None)

attn_dropout = 0.0
dropout_attn = Dropout(attn_dropout)
activation_attn = Activation('softmax')

heads=[]
attns=[]

for i in range(n_head):
    qs = qs_layers[i](enriched)  # (,114,80) -> Dense(80) -> (,114,80)
    ks = ks_layers[i](enriched)
    vs = vs_layers[i](enriched)

    temper = tf.sqrt(tf.cast(tf.shape(ks)[-1], dtype='float32'))  # 빈칸?
    attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]) / temper)([qs,ks])
    # attn = (,114,114)
    mmask = Lambda(lambda x:(-1e+9)*(1. - K.cast(x, 'float32')))(mask)
    attn = Add()([attn, mmask])
    attn = activation_attn(attn)
    attn = dropout_attn(attn)
    head = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, vs])
    # head = (,114,80)
    head_dropout = Dropout(dropout_rate)(head)
    heads.append(head_dropout)
    attns.append(attn)

head = K.stack(heads) if n_head > 1 else head
# (,114,80)
attn = K.stack(attns)
# (1,None,114, 114)
head = K.mean(head, axis=0) if n_head > 1 else head
head = w_o(head)
head = Dropout(dropout_rate)(head)

x, self_att = head, attn
# (,114,80), (1,None,114,114)

x, _ = apply_gating_layer(
    x,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    activation=None)
# x = (,114,80)

x = add_and_norm([x, enriched])
# x = (,114,80)

decoder = gated_residual_network(
    x,
    hidden_layer_size,
    dropout_rate=dropout_rate,
    use_time_distributed=True)
# decoder = (,114,80)

decoder, _ = apply_gating_layer(
    decoder, hidden_layer_size, activation=None)
# decoder = (,114,80)

transformer_layer = add_and_norm([decoder, temporal_feature_layer])
# (,114,80)

attention_components = {
    'decoder_self_attn': self_att,     # (1,None,114,114)
    'static_flags': static_weights[Ellipsis, 0],     # (, 2)
    'historical_flags': historical_flags[Ellipsis,0,:],  # (,30,1,10) 중 (,30,10)
    'future_flags': future_flags[Ellipsis,0,:]    # (,30,1,10) 중 (,30,10)
}

# 여기까지 _build_base_graph
# 여기서부터 build_model

# (,30,80) -> (,30,3)
outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(3))(transformer_layer[Ellipsis,num_encoder_steps:, :])

adam = tf.keras.optimizers.Adam(
    lr=learning_rate, clipnorm=max_gradient_norm)

model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

# 다음은 loss 정의


# # 잠시 attention을 볼까
# train, valid, test = formatter.split_data(raw_data)
#
# data = _batch_data(train)
# inputs = data['inputs']
# identifiers = data['identifier']
# time = data['time']
#
# batch_size = minibatch_size  # 64
# n = inputs.shape[0]   # 합(identifier 수 * 일 수) = 5932
# num_batches = n // batch_size   # 93
# if n - (num_batches * batch_size) > 0:   # -20
#     num_batches += 1
#
# batched_inputs = [   # 93, 64, 30, 15? 어떻게 돼있는겨..
#     inputs[i * batch_size:(i+1) * batch_size, Ellipsis]   # (64,30,15)
#     for i in range(num_batches)
# ]
#
# def get_batch_attention_weights(input_batch):   # (64,30,15) 가 들어가..
#     input_placeholder = all_inputs   # (,114,15). None일 수도 있겠는디..
#     attention_weights = {}
#     for k in attention_components:   # k는 key.
#         attention_weight = attention_components[k](input_batch.astype(np.float32))
#         attention_weights[k] = attention_weight
#     return attention_weights
#
#
# # batch 는 (64,30,15). bathed_inputs 는 93,64,30,15
# attention_by_batch = [
#     get_batch_attention_weights(batch) for batch in batched_inputs
# ]
#
# for batch in batched_inputs:
#     print(np.array(batch).shape)
#
#
#
# input_placeholder = None
# for k in attention_components:
#     attention_weight = tf.keras.backend.get_session().run(
#         attention_components[k],
#         {input_placeholder: batched_inputs[0].astype(np.float32)}
#     )
#     attention_weights[k] = attention_weight