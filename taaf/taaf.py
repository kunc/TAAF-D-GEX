from keras.engine.base_layer import Layer
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Activation, Dense
import tensorflow as tf

class ATU(Layer):
    """Adaptive Transformation Unit.

    It follows:
    `f(x) = alpha * x + beta`,
    where `alpha` and `beta` are learned arrays with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights of alpha.
        alpha_regularizer: regularizer for the weights of alpha.
        alpha_constraint: constraint for the weights of alpha.
        beta_initializer: initializer function for the weights of beta.
        beta_regularizer: regularizer for the weights of beta.
        beta_constraint: constraint for the weights of beta.

    # References
        - TODO: fill the reference
    """

    def __init__(self,
                 alpha_initializer='ones',
                 beta_initializer='zeros',
                 alpha_regularizer=None,
                 beta_regularizer=None,
                 alpha_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.supports_masking = False  # just to be sure, not tested
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        # TODO: properly test regularizers and constrainsts
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        if K.backend() != 'tensorflow':
            raise ValueError('The only supported backend is tensorflow! Sorry.')
        super(ATU, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 1
        input_dim = input_shape[-1]
        self.alpha = self.add_weight(shape=(input_dim,),
                                     initializer=self.alpha_initializer,
                                     name='alpha',
                                     trainable=True,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint,
                                     )
        self.beta = self.add_weight(shape=(input_dim,),
                                    initializer=self.beta_initializer,
                                    name='beta',
                                    trainable=True,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    )
        super(ATU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        #outputs = tf.multiply(self.alpha, inputs)  # multiply not in Keras backend :(
        #outputs = tf.add(outputs, self.beta)
        outputs = self.alpha * inputs + self.beta
        return outputs

    def get_config(self):
        config = {'alpha_initializer': initializers.serialize(self.alpha_initializer),
                  'beta_initializer': initializers.serialize(self.beta_initializer),
                  'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'alpha_constraint': constraints.serialize(self.alpha_constraint),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  }
        base_config = super(ATU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def taaf(x, activation, name=''):
    """Transformative Adaptive Activation Function.
        It follows:
        `f(x) = alpha * f(beta*x + gamma) + delta`,
        where f is a given activation function.
    """

    x = ATU(name=name+'TAAF_Bottom')(x)
    x = Activation(activation)(x)
    x = ATU(name=name+'TAAF_Top')(x)

    return x

def taaf_dense(x, units, activation, name='TAAF'):
    """Transformative Adaptive Activation Function.
        It follows:
        `f(x) = alpha * f(beta*x + gamma) + delta`,
        where f is a given activation function.
    """
    x = Dense(units=units, use_bias=False, name=name+'_Dense')(x)
    x = taaf(x, activation=activation, name=name+'_Adaptive')

    return x





