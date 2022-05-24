import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import sys
import numpy as np
from hls4ml.converters import convert_from_keras_model
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from callbacks import all_callbacks

os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

DATA_DIR = 'npy'
MODEL_DIR = 'model'

#BOARD_NAME = 'pynq-z1'
#FPGA_PART = 'xc7z020clg400-1'

BOARD_NAME = 'arty-a7-100t'
FPGA_PART = 'xc7a100tcsg324-1'

#BOARD_NAME = 'ultra96v2'
#FPGA_PART = 'xczu3eg-sbva484-1-e'

CLOCK_PERIOD = 10

#
# Load and scale dataset
#
print("Load and scale dataset")
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, 5)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)
np.save(DATA_DIR + '/y_test.npy', y_test)
np.save(DATA_DIR + '/X_test.npy', X_test)
np.save(DATA_DIR + '/classes.npy', le.classes_, allow_pickle=True)

#
# Model
#
print("Create model")
model = Sequential()
model.add(QDense(64, input_shape=(16,), name='fc1',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu1'))
model.add(QDense(32, input_shape=(16,), name='fc2',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu2'))
model.add(QDense(32, input_shape=(16,), name='fc3',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(QActivation(activation=quantized_relu(6), name='relu3'))
model.add(QDense(5, name='output',
                 kernel_quantizer=quantized_bits(6,0,alpha=1), bias_quantizer=quantized_bits(6,0,alpha=1),
                 kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
model.add(Activation(activation='softmax', name='softmax'))

#
# Training (or loading model)
#
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
pruning_params = {'pruning_schedule' : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
model = prune.prune_low_magnitude(model, **pruning_params)

train = not os.path.exists(MODEL_DIR + '/model.h5')
if train:
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    callbacks = all_callbacks(stop_patience = 1000,
                              lr_factor = 0.5,
                              lr_patience = 10,
                              lr_epsilon = 0.000001,
                              lr_cooldown = 2,
                              lr_minimum = 0.0000001,
                              outputDir = 'training_dir')
    callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=30, validation_split=0.25, shuffle=True,
              callbacks = callbacks.callbacks)
    print("Save model in 'model/model.h5'")
    # Save the model again but with the pruning 'stripped' to use the regular layer types
    model = strip_pruning(model)
    if not os.path.isdir(MODEL_DIR): 
        os.mkdir(MODEL_DIR)
    model.save(MODEL_DIR + '/model.h5')
else:
    from tensorflow.keras.models import load_model
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)
    print("Load model from " + MODEL_DIR + '/model.h5')
    model = load_model(MODEL_DIR + '/model.h5', custom_objects=co)

#
# TF prediction
#
print("Run prediction")
y_keras = model.predict(X_test)
np.save(DATA_DIR + '/y_qkeras.npy', y_keras)

#
# hls4ml
#
import hls4ml
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

hls_config['Model'] = {}
hls_config['Model']['ReuseFactor'] = 64
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
hls_config['LayerName']['fc1']['ReuseFactor'] = 64
hls_config['LayerName']['fc2']['ReuseFactor'] = 64
hls_config['LayerName']['fc3']['ReuseFactor'] = 64
input_data = os.path.join(os.getcwd(), DATA_DIR + '/X_test.npy')
output_predictions = os.path.join(os.getcwd(), DATA_DIR + '/y_qkeras.npy')

hls_config['SkipOptimizers'] = ['relu_merge']

hls_model = convert_from_keras_model(model=model,
                                     clock_period=CLOCK_PERIOD,
                                     backend='VivadoAccelerator',
                                     board=BOARD_NAME,
                                     part=FPGA_PART,
                                     io_type='io_stream',
                                     interface='axi_master',
                                     driver='c',
                                     input_data_tb=DATA_DIR+'/X_test.npy',
                                     output_data_tb=DATA_DIR+'/y_qkeras.npy',
                                     hls_config=hls_config,
                                     output_dir=BOARD_NAME+'_axi_m_backend')

_ = hls_model.compile()

y_hls = hls_model.predict(np.ascontiguousarray(X_test))

if len(sys.argv) == 2 and sys.argv[1] == 'profile':
    print('Number of arguments:', len(sys.argv), 'arguments.')

    from sklearn.metrics import accuracy_score
    print('-----------------------------------')
    print('Keras  Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
    print('hls4ml Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
    print('-----------------------------------')
else:
    # When building please remember to package and export the IP
    hls_model.build(csim=False, synth=True, export=True)

    # Write header files with hardcoded data set
    hls4ml.writer.vivado_accelerator_writer.VivadoAcceleratorWriter.write_header_file(hls_model, X_test, y_test, y_keras, y_hls, 64, BOARD_NAME + '_axi_m_backend/sdk/common/data.h')

    #
    hls4ml.report.read_vivado_report(BOARD_NAME + '_axi_m_backend/')

    # Generate bitstream and HDF file
    hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
