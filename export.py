import tensorflow as tf
import tensorflow.python.keras.backend as K
from utils.models import P2PNet

import tf2onnx
import onnx
from onnxsim import simplify
import onnx2tf

K.clear_session()
model = P2PNet(
    feature_size=256,
    backbone_name="vgg16",
    preprocessing=False,
    input_shape=(None, None, 3),
)
model.load_weights("weights/ckpt/VGG16_256_860")

RESOLUTION = [
    [None, None],
    [256,256],
    [256,384],
    [256,512],
    [256,640],
    [256,896],
    [256,1024],
    [384,384],
    [384,512],
    [384,640],
    [384,896],
    [384,1024],
    [384,1280],
    [512,512],
    [512,640],
    [512,896],
    [512,1024],
    [512,1280],
    [640,640],
    [640,896],
    [640,1024],
    [640,1280],
    [768,1280],
]


for H, W in RESOLUTION:
    built_graph = model.build_graph(shape=(H, W, 3))
    output_path = f'saved_model_{H}x{W}'
    tf.saved_model.save(built_graph, output_path)

    model_proto, external_tensor_storage = \
        tf2onnx.convert.from_keras(
            model=built_graph,
            opset=11,
            inputs_as_nchw=['input:0'],
            shape_override={'input:0': [1, H, W, 3]},
            output_path=f'./p2pnet_vgg16_{H}x{W}.onnx'
        )

    model_onnx = onnx.load(f'./p2pnet_vgg16_{H}x{W}.onnx')
    model_simp, check = simplify(model_onnx)
    onnx.save(model_simp, f'./p2pnet_vgg16_{H}x{W}.onnx')

for H, W in RESOLUTION:
    if H is not None and W is not None:
        onnx2tf.convert(
            input_onnx_file_path=f'./p2pnet_vgg16_{H}x{W}.onnx',
            output_folder_path=output_path,
            output_signaturedefs=True,
            copy_onnx_input_output_names_to_tflite=True,
            not_use_onnxsim=True,
            check_onnx_tf_outputs_elementwise_close_full=True,
        )
