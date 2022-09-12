import os
import cv2
import sys
import hls4ml
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
gpu_count = len(gpu_devices)
if gpu_count > 0:
    print('Num of available GPUs:', len(gpu_devices))
else:
    print('NO GPUs are available. Using CPUs.')


def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

# Choose a target board/chip
##
# BOARD_NAME = 'pynq-z1'
# PART = 'xc7z020clg400-1'
# CLOCK_PERIOD = 10
##
#BOARD = 'pynq-z2'
#PART = 'xc7z020clg400-1'
#CLOCK_PERIOD = 10
##
BOARD = 'ultra96v2'
PART = 'xczu3eg-sbva484-1-e'
CLOCK_PERIOD = 5
##

# Load dataset
DATA_DIR = 'npy'
X_test = np.load(DATA_DIR + '/X_test.npy')

# Load pre-trained model (.PB)
MODEL_DIR = 'model/best_status'
model = tf.keras.models.load_model(MODEL_DIR)
#model.summary()

# DEBUG: remove later layers (Lambda)
model = Model(model.input, model.layers[-6].output)
model.summary()

# Run model prediction
y_keras = model.predict(X_test)
np.save(DATA_DIR + '/y_keras.npy', y_keras)

## Save input and output images
#IMG_DIR = 'jpg'
#cv2.imwrite(IMG_DIR + '/' + 'input.jpg', X_test[0])
#cv2.imwrite(IMG_DIR + '/' + 'output.jpg', y_keras[0])

# Setup rounding and saturation modes on activation layers
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(
    layers=['Activation'],
    rounding_mode='AP_RND_CONV',
    saturation_mode='AP_SAT')

# Get hls4ml configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Setup hls4ml configuration
DEF_RF = 8
config['Model']['Strategy'] = 'Resource'
config['Model']['ReuseFactor'] = DEF_RF
config['Model']['BramFactor'] = 0
for layer in config['LayerName']:
   config['LayerName'][layer]['ReuseFactor'] = DEF_RF

#print('-----------------------------------')
#print('Configuration')
#print_dict(config)
#print('-----------------------------------')

# Get hls4ml model
INTERFACE = 'axi_master'
OUTPUT_DIR = '{}_qresource_rf{}'.format(INTERFACE, DEF_RF)
hls_model =  hls4ml.converters.convert_from_keras_model(
        model=model,
        clock_period=CLOCK_PERIOD,
        hls_config=config,
        part=PART,
        io_type='io_stream',
        output_dir=OUTPUT_DIR,
        # ---- #
        backend='VivadoAccelerator',
        interface=INTERFACE,
        board=BOARD,
        driver='python',
        input_data_tb=DATA_DIR+'/X_test.npy',
        output_data_tb=DATA_DIR+'/y_test.npy')

hls_model.compile()

## Run hls4ml model prediction
#y_hls = hls_model.predict(np.ascontiguousarray(X_test))
#
## # Print some predictions
## for i in range(0,8):
##    print('[', i, ']')
##    print('   - QKeras   : ', y_qkeras[i])
##    print('   - hls4ml   : ', y_hls[i])
#
#
#results = hls_model.build(csim=False, synth=True, vsynth=False, export=True, bitfile=True)
#
## # Show reports
## hls4ml.report.read_vivado_report(OUTPUT_DIR)
