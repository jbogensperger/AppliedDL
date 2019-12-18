from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.classification.base_model import BaseClassificationModel
from kashgari.layers import L

import logging
logging.basicConfig(level='DEBUG')


class DoubleBLSTMModel(BaseClassificationModel):

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_blstm1': {
                'units': 64,
                'return_sequences': True
            },
            'layer_blstm2': {
                'units': 64,
                'return_sequences': False
            },
            'layer_blstm3': {
                'units': 32,
                'return_sequences': False
            },
            'layer_dropout1': {
                'rate': 0.3
            },
            'layer_dropout2': {
                'rate': 0.2
            },
            'layer_dropout3': {
                'rate': 0.2
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            },
            'layer_dense': {
                'activation': 'softmax'
            },
            'layer_flatten': {},
        }

    def build_model_arc(self):
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define your layers
        layer_blstm1 = L.Bidirectional(L.LSTM(**config['layer_blstm1']),
                                       name='layer_blstm1')
        layer_blstm2 = L.Bidirectional(L.LSTM(**config['layer_blstm2']),
                                       name='layer_blstm2')

        layer_blstm3 = L.Bidirectional(L.LSTM(**config['layer_blstm3']),
                                       name='layer_blstm3')

        layer_dropout1 = L.Dropout(**config['layer_dropout1'],
                                  name='layer_dropout1')

        layer_dropout2 = L.Dropout(**config['layer_dropout2'],
                                  name='layer_dropout2')

        layer_dropout3 = L.Dropout(**config['layer_dropout3'],
                                   name='layer_dropout3')

        #layer_flatten = L.Flatten(**config['layer_flatten'])
        #layer_activation = L.Activation(**config['layer_activation'])
        layer_dense = L.Dense(output_dim, **config['layer_dense'])

        # Define tensor flow
        tensor = layer_dropout1(embed_model.output)
        tensor = layer_blstm1(tensor)
        tensor = layer_dropout2(tensor)
        tensor = layer_blstm2(tensor)
        #tensor = layer_dropout3(tensor)
        #tensor = layer_blstm3(tensor)
        output_tensor = layer_dense(tensor)

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)