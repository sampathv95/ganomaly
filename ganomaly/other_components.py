from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
from generator import Generator

class OtherComponents():

    def __init__(self, input_shape, activation_func):
        self.input_shape = input_shape
        self.activation_func = activation_func
        self.height, self.width, self.channels = input_shape

    def build_re_encoder(self):
        re_enc = Generator.build_encoder(input_shape=self.input_shape, activation_func=self.activation_func)
        return re_enc

    @staticmethod
    def build_feature_extractor(input_shape, activation_func):
        # this architecture is similar to gen_enc and re_enc but without the last layer
        gen_enc = Generator.build_encoder(input_shape=input_shape, activation_func=activation_func)
        # removing the last layer from gen_enc/re_encoder
        feature_extractor = Model(inputs=gen_enc.inputs, outputs=gen_enc.layers[-2].output)
        return feature_extractor

    def build_discriminator(self):
        feature_extractor = self.build_feature_extractor(self.input_shape, self.activation_func)
        disc_inp = Input(shape=self.input_shape, name='discriminator_inp')
        d = GlobalAveragePooling2D(name='discriminator_global_pool')(disc_inp)
        d = Dense(1, activation='sigmoid')(d)
        discriminator = Model(inputs=disc_inp, outputs=d)
        return feature_extractor, discriminator

if __name__ == '__main__':
    oc = OtherComponents(input_shape=(64, 64, 1), activation_func='selu')
    print(oc.build_feature_extractor().summary())