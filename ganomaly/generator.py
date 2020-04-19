import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Reshape
from tensorflow.keras import Model

class Generator():

    def __init__(self, input_shape, activation_func):
        self.input_shape = input_shape
        self.activation_func = activation_func
        self.height, self.width, self.channels = input_shape

    @staticmethod
    def build_encoder(input_shape, activation_func):
        # creating a keras functional model for encoder part of generator
        # DC-GAN architecture
        inp_layer = Input(shape=input_shape, name='input_layer')
        x = Conv2D(32, 5, strides=1, padding='same', activation=activation_func, name='conv_layer_1')(inp_layer)
        x = Conv2D(64, 3, strides=2, padding='same', activation=activation_func, name='conv_layer_2')(x)
        x = BatchNormalization(name='batch_norm_layer_1')(x)
        x = Conv2D(128, 3, strides=2, padding='same', activation=activation_func, name='conv_layer_3')(x)
        x = BatchNormalization(name='batch_norm_layer_2')(x)
        x = Conv2D(128, 3, strides=2, padding='same', activation=activation_func, name='conv_layer_4')(x)
        x = BatchNormalization(name='batch_norm_layer_3')(x)
        x = GlobalAveragePooling2D(name='gen_enc_op')(x)
        gen_enc = Model(inputs=inp_layer, outputs=x, name='gen_enc')
        return gen_enc

    def build_generator(self):
        # creating the complete generator
        # DC-GAN architecture
        gen_enc = self.build_encoder(input_shape=self.input_shape, activation_func=self.activation_func)
        gen_inp_layer = Input(shape=self.input_shape, name='generator_input')
        x = gen_enc(gen_inp_layer)
        y = Dense(self.width*self.width*2, name='dec_dense_1')(x)
        y = Reshape(target_shape=(self.width//8, self.width//8, 128), name='dec_reshape')(y)
        y = Conv2DTranspose(128, 3, strides=2, padding='same', activation=self.activation_func, name='dec_conv_layer_1')(y)
        y = Conv2DTranspose(64, 3 ,strides=2, padding='same', activation=self.activation_func, name='dec_conv_layer_2')(y)
        y = Conv2DTranspose(32, 3, strides=2, padding='same', activation=self.activation_func, name='dec_conv_layer_3')(y)
        y = Conv2DTranspose(self.channels, 1, strides=1, padding='same', activation='tanh', name='dec_conv_op')(y)
        generator = Model(inputs=gen_inp_layer, outputs=y, name='generator')
        return gen_enc, generator

    def __call__(self, **kwargs):
        # method to get summary of the model
        if not kwargs:
            gen_obj = Generator(input_shape=(64, 64, 1), activation_func='selu')
            gen_enc, gen = gen_obj.build_generator()
            print(gen.summary())
        else:
            gen_enc, gen = kwargs.get('models')
            print(gen.summary())


if __name__ == '__main__':
    gen_obj = Generator(input_shape=(64, 64, 1), activation_func='selu')
    gen_enc, gen = gen_obj.build_generator()
    print(gen_enc)
    print(gen.layers)


