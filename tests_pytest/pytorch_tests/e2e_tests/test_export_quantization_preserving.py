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
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, Signedness
from mct_quantizers import QuantizationMethod
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
import onnx
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import \
    FakelyQuantONNXPyTorchExporter
from model_compression_toolkit.exporter.model_wrapper import is_pytorch_layer_exportable
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION


def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs

def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.dropout = torch.nn.Dropout()
            self.flatten1 = torch.nn.Flatten()
            self.flatten2 = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(108, 216)
            self.fc2 = torch.nn.Linear(216, 432)

        def forward(self, x):
            x = self.conv(x)
            x = self.flatten1(x)
            x = self.fc1(x)
            x = self.flatten2(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return BaseModel()

def get_tpc(insert_preserving):
    base_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    mixed_precision_cfg_list = [base_config]
    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                           base_config=base_config)

    operator_set = []
    preserving_quantization_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False, quantization_preserving=True))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=preserving_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=preserving_quantization_config))
    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)
    operator_set.extend([conv, fc])

    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set),
        insert_preserving_quantizers=insert_preserving)

    return tpc

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

def test_export_quantization_preserving_holder(tmp_path):
    """
    Testing the exporter. Make sure that quantization_bypass is converted to False when PytorchPreservingActivationQuantizationHolder.
    """
    float_model = get_float_model().to('cuda')
    target_platform_cap = get_tpc(insert_preserving=True)
    core_config = CoreConfig()

    output_names = ['float_output']
    save_model_path = tmp_path / "float_model.onnx"
    data_generator = representative_data_gen_2(num_inputs=1)
    onnx_float_model = export_model(float_model, save_model_path, data_generator, output_names=output_names)
    
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=target_platform_cap
    )
    
    output_names = ['quant_output']
    save_model_path = tmp_path / "quantized_model.onnx"
    data_generator = representative_data_gen_2(num_inputs=1)
    onnx_model = export_model(quantized_model, save_model_path, data_generator, output_names=output_names)
    expected_output_names = output_names
    validate_outputs(onnx_model, expected_output_names)
