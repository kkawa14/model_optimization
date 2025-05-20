import pytest

from unittest.mock import Mock

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest._test_util.graph_builder_utils import build_node
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode

from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig, WeightsAttrQuantizationConfig
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import BitWidthConfig, QuantizationConfig

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph

from typing import List

import torch
from torch import nn

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS, POS_ATTR
from model_compression_toolkit.target_platform_capabilities.constants import PYTORCH_KERNEL

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configs_to_node

from model_compression_toolkit.core.common.model_collector import create_stats_collector_for_node

from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector, StatsCollector
from model_compression_toolkit.core.common.framework_info import ChannelAxis

from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS, \
    IMX500_TP_MODEL

import numpy as np

class TestCalculateQuantizationParams:
    def get_op_qco(self):
        # define a default quantization config for all non-specified weights attributes.
        default_weight_attr_config = AttributeQuantizationConfig()

        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True)

        base_cfg = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            signedness=Signedness.AUTO)

        default_config = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            signedness=Signedness.AUTO
        )

        mx_cfg_list = [base_cfg]
        for n in [2, 4, 16]:
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))

        return base_cfg, mx_cfg_list, default_config

    def generate_tpc_local(self, default_config, base_config, mixed_precision_cfg_list):
        default_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple([default_config]))
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple(mixed_precision_cfg_list),
            base_config=base_config)

        operator_set = []

        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
        add = schema.OperatorsSet(name=schema.OperatorSetNames.ADD)
        operator_set.extend([conv, relu, add])

        generated_tpc = schema.TargetPlatformCapabilities(
            default_qco=default_configuration_options,
            operator_set=tuple(operator_set))

        return generated_tpc

    def get_tpc(self):
        base_cfg, mx_cfg_list, default_config = self.get_op_qco()
        tpc = self.generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.conv3(x)
                return x

        return BaseModel()

    def _create_weights_attr_quantization_config(self, weights_n_bits: int) -> AttributeQuantizationConfig:
        """
        Helper method to create a weights attribute quantization configuration.

        Args:
            weights_n_bits (int): Number of bits to use for quantizing weights.

        Returns:
            AttributeQuantizationConfig: Holds the quantization configuration of a weight attribute of a layer.
        """
        weights_attr_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=weights_n_bits,
            weights_per_channel_threshold=False,
            enable_weights_quantization=True,
            lut_values_bitwidth=None)
        return weights_attr_config

    def _create_node_weights_op_cfg(
            self,
            def_weight_attr_config: AttributeQuantizationConfig) -> OpQuantizationConfig:


        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.SYMMETRIC,
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True,
            lut_values_bitwidth=None)

        # define a quantization config to quantize the bias (for layers where there is a bias attribute).
        bias_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
            weights_n_bits=FLOAT_BITWIDTH,
            weights_per_channel_threshold=False,
            enable_weights_quantization=False,
            lut_values_bitwidth=None)

        attr_weights_configs_mapping = {'weight': kernel_base_config, 'bias': bias_config}
        print('attr_weights_configs_mapping', attr_weights_configs_mapping)
        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=def_weight_attr_config,
            attr_weights_configs_mapping=attr_weights_configs_mapping,
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=None,
            fixed_zero_point=None,
            simd_size=None,
            signedness=Signedness.AUTO
        )

        return op_cfg


    def get_test_graph(self, qc):
        float_model = self.get_float_model()
        fw_info = DEFAULT_PYTORCH_INFO

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)
        graph.set_fw_info(fw_info)

        tpc = self.get_tpc()
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, qc.custom_tpc_opset_to_layer)
        graph.set_fqc(fqc)

        def_weight_attr_config = self._create_weights_attr_quantization_config(8)
        op_cfg = self._create_node_weights_op_cfg(def_weight_attr_config=def_weight_attr_config)

        quantization_config = QuantizationConfig()
        print()
        graph.node_to_out_stats_collector = dict()
        for id, n in enumerate(graph.nodes):
            print('n', id, n)
            n.prior_info = fw_impl.get_node_prior_info(n, fw_info, graph)

            #sc = BaseStatsCollector()
            #graph.set_out_stats_collector_to_node(n, sc)

            #from model_compression_toolkit.core.common.model_collector import create_stats_collector_for_node
            #create_stats_collector_for_node(n, fw_info)

            #node_qc_options = n.get_qco(fqc)
            #print('zzz000', node_qc_options)
            if False:
                set_quantization_configs_to_node(n, graph, quantization_config, fw_info, fqc)
            else:
                #"""
                n.candidates_quantization_cfg = []
                candidate_qc_a = CandidateNodeQuantizationConfig(
                    activation_quantization_cfg=NodeActivationQuantizationConfig(qc=quantization_config, op_cfg=op_cfg,
                                                                                 activation_quantization_fn=None,
                                                                                 activation_quantization_params_fn=None),
                    weights_quantization_cfg=NodeWeightsQuantizationConfig(qc=quantization_config, op_cfg=op_cfg, weights_channels_axis=[0, 1], node_attrs_list=['weight', 'bias'])

                )
                if n.name in ['conv3', 'relu']:
                    candidate_qc_a.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.FLN_QUANT
                else:
                    candidate_qc_a.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.QUANT
                n.candidates_quantization_cfg.append(candidate_qc_a)
                #"""

                graph.node_to_out_stats_collector[n] = StatsCollector(init_min_value=-1.234+id, init_max_value=5.678, out_channel_axis=fw_info.out_channel_axis_mapping.get(n.type))
                graph.node_to_out_stats_collector[n].hc._n_bins = 3
                graph.node_to_out_stats_collector[n].hc._bins = np.array([1, 2, 3, 4*(id+1)])
                graph.node_to_out_stats_collector[n].hc._counts= np.array([4, 5, 6])
                print("dbg051900 n_bins", graph.node_to_out_stats_collector[n].hc._n_bins)
                print("dbg051900 bits", graph.node_to_out_stats_collector[n].hc._bins)
                print("dbg051900 counts", graph.node_to_out_stats_collector[n].hc._counts)



            for candidate_qc in n.candidates_quantization_cfg:
                print('################')
                print('candidate_qc', type(candidate_qc))
                print('--weights_quantization_cfg-----------------------------------------')
                print(candidate_qc.weights_quantization_cfg)
                print('-------------------------------------------')
                print('--activation_quantization_cfg-----------------------------------------')
                print(candidate_qc.activation_quantization_cfg)
                print('-------------------------------------------')


        return graph, fw_impl

    # test case for test_calculate_quantization_params
    test_input_0 = (None, None)
    test_expected_0 = ("The filters cannot be None.", None)

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
    ])
    def test_calculate_quantization_params(self, inputs, expected):
        quantization_config = QuantizationConfig()
        graph, fw_impl = self.get_test_graph(quantization_config)
        print()
        print(graph)
        print(graph.nodes)

        calculate_quantization_params(graph, fw_impl, self.representative_data_gen)

        for node in graph.nodes:
            print('node', node)
            for candidate_qc in node.candidates_quantization_cfg:
                print('###############')
                print('candidate_qc.quant_mode', candidate_qc.activation_quantization_cfg.quant_mode)
                print('candidate_qc.actquaparams', candidate_qc.activation_quantization_cfg.activation_quantization_params)
        pass

