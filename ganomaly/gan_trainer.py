import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras import Model
from generator import Generator

class AdversarialLoss(Layer):
    # layer to implement feature matching for adversarial training
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def call(self, inputs, **kwargs):
        original_image, recon_image = inputs
        original_feature = self.feature_extractor(original_image)
        gan_feature = self.feature_extractor(recon_image)
        return tf.keras.losses.mean_squared_error(original_feature, gan_feature)

class ReconLoss(Layer):
    # layer to implement reconstruction loss in the generator
    def __init__(self, **kwargs):
       super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        original_image, recon_image = inputs
        return tf.keras.losses.mean_absolute_error(original_image, recon_image)

class LatentLoss(Layer):
    # layer to implement reconstruction loss in the latent space
    def __init__(self, gen_enc, re_enc,**kwargs):
        super().__init__(**kwargs)
        self.gen_enc = gen_enc
        self.re_enc = re_enc

    def call(self, inputs, **kwargs):
        original_image, recon_image = inputs
        generator_encoding = self.gen_enc(original_image)
        re_encoder_encoding = self.re_enc(recon_image)
        return tf.keras.losses.mean_squared_error(generator_encoding, re_encoder_encoding)

if __name__ == '__main__':
    gen_obj = Generator(input_shape=(64, 64, 1), activation_func='selu')
    gen_enc, gen = gen_obj.build_generator()
    inp = Input(shape=(64, 64, 1), name='inppp')
    gan_op = gen(inp)
    recon_loss = ReconLoss(name='recon_loss')([inp, gan_op])
    model = Model(inputs=inp, outputs=recon_loss)
    print(model.layers)
