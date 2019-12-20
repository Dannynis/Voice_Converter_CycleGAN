import os
import tensorflow as tf
from tensorflow.keras import Model, layers
from module import Discriminator, Generator
from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime

class CycleGAN:

    def __init__(self, num_features, mode = 'train', log_dir = './log'):

        #super(CycleGAN,self).__init__()

        self.num_features = num_features

        self.mode = mode

        self.build_model(num_features)

        # self.saver = tf.train.Saver()
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.inputs_setup = False
        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.create_file_writer(self.log_dir)

            # self.generator_summaries, self.discriminator_summaries = self.summary()

    def setup_inputs(self,input_A):
        self.generator_A2B._set_inputs(input_A)
        self.generator_B2A._set_inputs(input_A)
        self.discriminator_A._set_inputs(input_A)
        self.discriminator_B._set_inputs(input_A)
        self.inputs_setup=True


    def build_model(self,num_features):


        self.generator_A2B = Generator(num_features)
        self.generator_B2A = Generator(num_features)

        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()



    def fit(self, input_A, input_B, lambda_cycle, lambda_identity, optimizer_generator, optimizer_discriminator):

        if not self.inputs_setup:
            self.setup_inputs(input_A)
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:

            generation_A = self.generator_A2B(inputs=input_A)

            generation_B = self.generator_B2A(inputs=input_B)

            print ('inputing {}'.format(generation_B.shape))

            cycle_A = self.generator_B2A(inputs=generation_B)

            cycle_B = self.generator_A2B(inputs = generation_A)

            generation_A_identity = self.generator_B2A(inputs=input_A)

            generation_B_identity = self.generator_A2B(inputs=input_B)

            cycle_loss = l1_loss(y = input_A, y_hat = cycle_A) + l1_loss(y = input_B, y_hat = cycle_B)

            # Identity loss
            identity_loss = l1_loss(y = input_A, y_hat = generation_A_identity) + l1_loss(y = input_B, y_hat = generation_B_identity)

            discrimination_A_fake = self.discriminator_A(inputs=generation_A)
            discrimination_B_fake = self.discriminator_B(inputs=generation_B)


            # Generator loss
            # Generator wants to fool discriminator
            generator_loss_A2B = l2_loss(y=tf.ones_like(discrimination_B_fake), y_hat=discrimination_B_fake)
            generator_loss_B2A = l2_loss(y=tf.ones_like(discrimination_A_fake), y_hat=discrimination_A_fake)


            # Merge the two generators and the cycle loss
            generator_loss = generator_loss_A2B + generator_loss_B2A + lambda_cycle * cycle_loss + lambda_identity * identity_loss

            generator_trainable_variables = self.generator_A2B.trainable_variables + self.generator_B2A.trainable_variables
            generator_gradients = g.gradient(generator_loss, generator_trainable_variables)
            optimizer_generator.apply_gradients(zip(generator_gradients, generator_trainable_variables))

        ############################################################################################################################
        with tf.GradientTape() as g:

            # Discriminator loss
            discrimination_input_A_real = self.discriminator_A(inputs=input_A)
            discrimination_input_B_real = self.discriminator_B(inputs=input_B)


            # Discriminator wants to classify real and fake correctly
            discriminator_loss_input_A_real = l2_loss(y=tf.ones_like(discrimination_input_A_real),
                                                           y_hat=discrimination_input_A_real)
            discriminator_loss_input_A_fake = l2_loss(y=tf.zeros_like(discrimination_A_fake),
                                                           y_hat=discrimination_A_fake)
            discriminator_loss_A = (discriminator_loss_input_A_real + discriminator_loss_input_A_fake) / 2

            discriminator_loss_input_B_real = l2_loss(y=tf.ones_like(discrimination_input_B_real),
                                                           y_hat=discrimination_input_B_real)
            discriminator_loss_input_B_fake = l2_loss(y=tf.zeros_like(discrimination_B_fake),
                                                           y_hat=discrimination_B_fake)
            discriminator_loss_B = (discriminator_loss_input_B_real + discriminator_loss_input_B_fake) / 2

            # Merge the two discriminators into one
            discriminator_loss = discriminator_loss_A + discriminator_loss_B

            discriminator_trainable_variables = self.discriminator_A.trainable_variables+ self.discriminator_B.trainable_variables

            discriminator_gradients = g.gradient(discriminator_loss, discriminator_trainable_variables)

            optimizer_discriminator.apply_gradients(
                zip(discriminator_gradients,discriminator_trainable_variables))

        self.identity_loss = identity_loss
        self.cycle_loss = cycle_loss
        self.generator_loss_A2B = generator_loss_A2B
        self.generator_loss_B2A= generator_loss_B2A
        self.generator_loss = generator_loss
        self.discriminator_loss_A = discriminator_loss_A
        self.discriminator_loss_B = discriminator_loss_B
        self.discriminator_loss = discriminator_loss

        self.summary()


        self.train_step += 1





        return generator_loss, discriminator_loss






    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.generator_A2B(inputs)
        elif direction == 'B2A':
            generation = self.generator_B2A(inputs)
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save_model(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)


        self.generator_A2B.save(os.path.join (directory,filename+'GA2B'))
        self.generator_B2A.save(os.path.join (directory,filename+'GB2A'))
        self.discriminator_A.save(os.path.join (directory,filename+'DA'))
        self.discriminator_B.save(os.path.join (directory,filename+'DB'))

        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with self.writer.as_default():
            tf.summary.scalar('cycle_loss', self.cycle_loss,self.train_step)
            tf.summary.scalar('identity_loss', self.identity_loss,self.train_step)
            tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B,self.train_step)
            tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A,self.train_step)
            tf.summary.scalar('generator_loss', self.generator_loss,self.train_step)

            tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A,self.train_step)
            tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B,self.train_step)
            tf.summary.scalar('discriminator_loss', self.discriminator_loss,self.train_step)
            self.writer.flush()


if __name__ == '__main__':
    
    model = CycleGAN(num_features = 24)
    print('Graph Compile Successeded.')