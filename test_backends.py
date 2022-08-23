import os
import numpy as np

from tensorflow import keras
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PATH'] = '/tools/Xilinx/Vivado/2019.1/bin:' + os.environ['PATH']

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

#BOARD_NAME = 'pynq-z1'
#FPGA_PART = 'xc7z020clg400-1'

#BOARD_NAME = 'arty-a7-100t'
#FPGA_PART = 'xc7a100tcsg324-1'

BOARD_NAME = 'ultra96v2'
FPGA_PART = 'xczu3eg-sbva484-1-e'
CLOCK_PERIOD = 5

# Load and scale dataset
DATA_DIR = 'npy'
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

# Load pre-trained quantized model
MODEL_DIR = 'model'
WEIGHTS = MODEL_DIR + '/qkeras_3layer_weights.h5'
DESC = MODEL_DIR + '/qkeras_3layer.json'
co = {}
from qkeras.utils import _add_supported_quantized_objects
_add_supported_quantized_objects(co)

with open(DESC) as fl:
    desc = "\n".join(fl.readlines())
model = keras.models.model_from_json(desc, custom_objects=co)
model.load_weights(WEIGHTS)
#model.summary()

# Run QKeras model prediction
y_qkeras = model.predict(X_test)
np.save(DATA_DIR + '/y_qkeras.npy', y_qkeras)

# Setup rounding and saturation modes on activation layers
import hls4ml
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(
        layers=['Activation'],
        rounding_mode='AP_RND_CONV',
        saturation_mode='AP_SAT')

# Get hls4ml configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
#print("-----------------------------------")
#print("Configuration")
#print_dict(config)
#print("-----------------------------------")

# Setup hls4ml configuration
DEF_RF = 32
config["Model"]["Strategy"] = "Resource"
config["Model"]["ReuseFactor"] = DEF_RF
config["Model"]["BramFactor"] = 0
for layer in config["LayerName"]:
    config["LayerName"][layer]["ReuseFactor"] = DEF_RF
#print("-----------------------------------")
#print("Configuration")
#print_dict(config)
#print("-----------------------------------")

# Get hls4ml model
ENABLE_WRAPPER=True

if ENABLE_WRAPPER:
    hls_model =  hls4ml.converters.convert_from_keras_model(
            model=model,
            clock_period=CLOCK_PERIOD,
            backend='VivadoAccelerator',
            board=BOARD_NAME,
            part=FPGA_PART,
            io_type='io_stream',
            interface='axi_master',
            driver='c',
            input_data_tb=DATA_DIR+'/X_test.npy',
            output_data_tb=DATA_DIR+'/y_test.npy',
            hls_config=config,
            output_dir='wrapped_qresource32')
else:
    hls_model = hls4ml.converters.convert_from_keras_model(
            model=model,
            clock_period=CLOCK_PERIOD,
            hls_config=config,
            part=FPGA_PART,
            io_type = 'io_stream',
            output_dir = 'qresource32')

hls_model.compile()

# Run hls4ml model prediction
y_hls = hls_model.predict(np.ascontiguousarray(X_test[0]))

# Print some predictions
for i in range(0,1):
    print("[", i, "]")
    print("   - Reference: ", y_test[i])
    print("   - QKeras   : ", y_qkeras[i])
    print("   - hls4ml   : ", y_hls[i])


results = hls_model.build(csim=True, synth=False, vsynth=False)

###
### Work in progress
###
# TF prediction
#print("Run prediction")
#y_keras = model.predict(X_test)
#np.save(DATA_DIR + '/y_qkeras.npy', y_keras)
#
##
## hls4ml
##
#import hls4ml
##hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
##hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
##hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
#hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
#
#hls_config['Model'] = {}
#hls_config['Model']['ReuseFactor'] = 64
#hls_config['Model']['Strategy'] = 'Resource'
#hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
#hls_config['LayerName']['fc1']['ReuseFactor'] = 64
#hls_config['LayerName']['fc2']['ReuseFactor'] = 64
#hls_config['LayerName']['fc3']['ReuseFactor'] = 64
#input_data = os.path.join(os.getcwd(), DATA_DIR + '/X_test.npy')
#output_predictions = os.path.join(os.getcwd(), DATA_DIR + '/y_qkeras.npy')
#
#hls_config['SkipOptimizers'] = ['relu_merge']
#
#hls_model = convert_from_keras_model(model=model,
#                                     clock_period=CLOCK_PERIOD,
#                                     backend='VivadoAccelerator',
#                                     board=BOARD_NAME,
#                                     part=FPGA_PART,
#                                     io_type='io_stream',
#                                     interface='axi_master',
#                                     driver='c',
#                                     input_data_tb=DATA_DIR+'/X_test.npy',
#                                     output_data_tb=DATA_DIR+'/y_qkeras.npy',
#                                     hls_config=hls_config,
#                                     output_dir=BOARD_NAME+'_axi_m_backend')
#
##print(hls_model)
#
#_ = hls_model.compile()
#
#y_hls = hls_model.predict(np.ascontiguousarray(X_test))
#
#if len(sys.argv) == 2 and sys.argv[1] == 'profile':
#    print('Number of arguments:', len(sys.argv), 'arguments.')
#
#    from sklearn.metrics import accuracy_score
#    print('-----------------------------------')
#    print('Keras  Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
#    print('hls4ml Accuracy: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
#    print('-----------------------------------')
#else:
#    # When building please remember to package and export the IP
#    hls_model.build(csim=False, synth=True, export=True)
#
#    # Write header files with hardcoded data set
#    #hls4ml.writer.vivado_accelerator_writer.VivadoAcceleratorWriter.write_header_file(hls_model, X_test, y_test, y_keras, y_hls, 64, BOARD_NAME + '_axi_m_backend/sdk/common/data.h')
#
#    #
#    #hls4ml.report.read_vivado_report(BOARD_NAME + '_axi_m_backend/')
#
#    # Generate bitstream and HDF file
#    #hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
