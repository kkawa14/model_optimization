# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import model_compression_toolkit as mct
import torch
from tests_pytest._test_util.tpc_util import configure_mp_activation_opsets
from model_compression_toolkit.target_platform_capabilities.schema.v2 import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorSetNames, TargetPlatformCapabilities, Fusing, OperatorsSet
from tests.common_tests.helpers.generate_test_tpc import generate_test_attr_configs, generate_test_op_qc
import onnx
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import \
    FakelyQuantONNXPyTorchExporter
from model_compression_toolkit.exporter.model_wrapper import is_pytorch_layer_exportable
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION


def build_tpc():
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=[8],
        enable_activation_quantization=True,
        enable_weights_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO
    )

    opsets, _ = configure_mp_activation_opsets(
        opset_names=[OperatorSetNames.CONV,
                     OperatorSetNames.RELU,
                     OperatorSetNames.SIGMOID,
                     OperatorSetNames.FULLY_CONNECTED,
                     OperatorSetNames.HARDSWISH],
        base_op_config=default_op_cfg,
        a_nbits=[8]
    )
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    test_qc = generate_test_op_qc(**generate_test_attr_configs(), activation_n_bits=16)

    tpc = TargetPlatformCapabilities(
        default_qco=default_cfg,
        operator_set=opsets,
        fusing_patterns=[
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.RELU)), fuse_op_quantization_config=test_qc),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.CONV),
            OperatorsSet(name=OperatorSetNames.SIGMOID))),
        Fusing(operator_groups=(
            OperatorsSet(name=OperatorSetNames.FULLY_CONNECTED),
            OperatorsSet(name=OperatorSetNames.HARDSWISH)), fuse_op_quantization_config=test_qc),
        ]
    )
    return tpc

def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.sigmoid = torch.nn.Sigmoid()
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(in_features=48, out_features=10)
            self.hswish = torch.nn.Hardswish()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.sigmoid(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = self.hswish(x)
            return x
    return BaseModel()

def export_model(model, save_model_path, data_generator, output_names=None):
    exporter = FakelyQuantONNXPyTorchExporter(model,
                                              is_pytorch_layer_exportable,
                                              save_model_path,
                                              data_generator,
                                              onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION,
                                              use_onnx_custom_quantizer_ops=True)
    exporter.export(output_names)
    assert save_model_path.exists(), "ONNX file was not created"
    assert save_model_path.stat().st_size > 0, "ONNX file is empty"
    print("save_model_path:", save_model_path)
    onnx_model = onnx.load(str(save_model_path))
    return onnx_model

def representative_data_gen_2(num_inputs=1):
    batch_size, num_iter, shape = 2, 1, (3, 8, 8)
    def data_gen():
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs
    return data_gen

def validate_outputs(onnx_model, expected_output_names):
    outputs = onnx_model.graph.output
    # Check number of outputs
    assert len(outputs) == len(
        expected_output_names), f"Expected {len(expected_output_names)} output, but found {len(outputs)}"
    found_output_names = [output.name for output in outputs]
    assert found_output_names == expected_output_names, (
        f"Expected output name '{expected_output_names}' found {found_output_names}"
    )

def test_export_fln_quantization_holder(tmp_path):
    """
    Testing the exporter. Make sure that quantization_bypass is converted to False when PytorchFLNActivationQuantizationHolder.
    """
    float_model = get_float_model().to('cuda')

    output_names = ['float_output']
    save_model_path = tmp_path / "float_model.onnx"
    data_generator = representative_data_gen_2(num_inputs=1)
    onnx_float_model = export_model(float_model, save_model_path, data_generator, output_names=output_names)

    tpc = build_tpc()
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc
    )

    output_names = ['quant_output']
    save_model_path = tmp_path / "quantized_model.onnx"
    onnx_model = export_model(quantized_model, save_model_path, data_generator, output_names=output_names)
    expected_output_names = output_names
    validate_outputs(onnx_model, expected_output_names)
