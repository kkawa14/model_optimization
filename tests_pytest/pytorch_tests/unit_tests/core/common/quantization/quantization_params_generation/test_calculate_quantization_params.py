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
import pytest
import numpy as np

from unittest.mock import Mock
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    ActivationQuantizationMode, NodeActivationQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from model_compression_toolkit.core.common.collectors.statistics_collector import StatsCollector
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo


class TestCalculateQuantizationParams:
    def build_node(self, name='node', framework_attr={}, layer_class="Dummy",
                   q_mode=ActivationQuantizationMode.QUANT):

        node = Mock(spec=BaseNode)
        node.name = name
        node.layer_class = layer_class
        node.prior_info = Mock(min_output=None, max_output=None)
        node.minmax = (None, None)
        node.get_weights_by_keys.return_value = None
        node.get_node_weights_attributes.return_value = []

        if q_mode == ActivationQuantizationMode.QUANT:
            node.is_activation_quantization_enabled.return_value = True
        else:
            node.is_activation_quantization_enabled.return_value = False

        if q_mode == ActivationQuantizationMode.FLN_QUANT:
            node.is_fln_quantization.return_value = True
        else:
            node.is_fln_quantization.return_value = False


        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=AttributeQuantizationConfig(),
            attr_weights_configs_mapping={},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=16,
            enable_activation_quantization=True,
            quantization_preserving=False,
            supported_input_activation_n_bits=16,
            signedness=Signedness.AUTO
        )
        activation_quantization_cfg = NodeActivationQuantizationConfig(op_cfg=op_cfg)
        activation_quantization_cfg.set_qc(QuantizationConfig())
        activation_quantization_cfg.quant_mode = q_mode

        candidate_quantization_config = Mock(spec=CandidateNodeQuantizationConfig)
        candidate_quantization_config.activation_quantization_cfg = activation_quantization_cfg
        candidate_quantization_config.weights_quantization_cfg = Mock()

        node.candidates_quantization_cfg = [candidate_quantization_config]

        return node

    def get_test_graph(self, node_name, q_mode, data):
        node = self.build_node(node_name, q_mode=q_mode)
        graph = Graph('graph_name', input_nodes=[node], nodes=[node], output_nodes=[node], edge_list=[])

        graph.node_to_out_stats_collector = dict()
        for n in graph.nodes():
            n.prior_info = NodePriorInfo()

            graph.node_to_out_stats_collector[n] = StatsCollector(init_min_value=0.0, init_max_value=1.0, out_channel_axis=0)
            graph.node_to_out_stats_collector[n].hc._n_bins = 3
            graph.node_to_out_stats_collector[n].hc._bins = np.array(data)
            graph.node_to_out_stats_collector[n].hc._counts = np.array([1, 1])

        return graph

    ### test pattern for ActivationQuantizationMode
    @pytest.mark.parametrize(["node_name", "q_mode", "input_data", "expects"], [
        # node_name,             q_mode,                                     input data,      expected value
        ['node_quant',           ActivationQuantizationMode.QUANT,           [0.4, 0.8, 1.2], [1.0, False]],
        ['node_fln_quant',       ActivationQuantizationMode.FLN_QUANT,       [0.7, 1.4, 2.1], [2.0, False]],
        ['node_fln_no_quant',    ActivationQuantizationMode.FLN_NO_QUANT,    [0.7, 1.4, 2.1], [None, None]],
        ['node_no_quant',        ActivationQuantizationMode.NO_QUANT,        [0.7, 1.4, 2.1], [None, None]],
        ['node_preserve_quant',  ActivationQuantizationMode.PRESERVE_QUANT,  [0.7, 1.4, 2.1], [None, None]],
    ])
    def test_calculate_quantization_params(self, node_name, q_mode, input_data, expects, mocker):
        graph = self.get_test_graph(node_name, q_mode, input_data)

        mocker.patch('model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation._collect_nodes_for_hmse', return_value=[])
        calculate_quantization_params(graph, Mock(), Mock())

        for candidate_qc in list(graph.nodes)[0].candidates_quantization_cfg:
            assert type(candidate_qc.activation_quantization_cfg.activation_quantization_params) == dict
            if expects[0] is not None:
                ### QUANT or FLN_QUANT
                assert 'threshold' in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()
                assert 'is_signed' in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()

                threshold = candidate_qc.activation_quantization_cfg.activation_quantization_params['threshold']
                is_signed = candidate_qc.activation_quantization_cfg.activation_quantization_params['is_signed']
                assert threshold == expects[0]
                assert is_signed == expects[1]
            else:
                assert 'threshold' not in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()
                assert 'is_signed' not in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()
