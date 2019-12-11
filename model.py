import os
import tensorflow as tf
from module import Discriminator, Generator
from utils import l1_loss, l2_loss, cross_entropy_loss
from datetime import datetime

class CycleGAN(object):

    def __init__(self, num_features, mode = 'train', log_dir = './log'):

        self.num_features = num_features
        self.input_shape = [None, num_features, None] # [batch_size, num_features, num_frames]


        self.mode = mode

        self.build_model()

        # self.saver = tf.train.Saver()
        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.create_file_writer(self.log_dir)
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):


        self.generator_A2B = Generator()
        self.generator_B2A = Generator()

        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()



    def train(self, input_A, input_B, lambda_cycle, lambda_identity):



        generation_A = self.generator_A2B(inputs=input_A)

        generation_B = self.generator_B2A(inputs=input_B)

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
        generator_loss_A2B = l2_loss(y=tf.keras.ones_like(discrimination_B_fake), y_hat=discrimination_B_fake)
        generator_loss_B2A = l2_loss(y=tf.keras.ones_like(discrimination_A_fake), y_hat=discrimination_A_fake)


        # Merge the two generators and the cycle loss
        generator_loss = generator_loss_A2B + generator_loss_B2A + lambda_cycle * cycle_loss + lambda_identity * identity_loss






        ############################################################################################################################

        # Discriminator loss
        discrimination_input_A_real = self.discriminator_A(inputs=input_A, reuse=True,
                                                              scope_name='discriminator_A')
        discrimination_input_B_real = self.discriminator_B(inputs=input_B, reuse=True,
                                                              scope_name='discriminator_B')


        # Discriminator wants to classify real and fake correctly
        discriminator_loss_input_A_real = l2_loss(y=tf.keras.ones_like(discrimination_input_A_real),
                                                       y_hat=discrimination_input_A_real)
        discriminator_loss_input_A_fake = l2_loss(y=tf.keras.zeros_like(discrimination_A_fake),
                                                       y_hat=discrimination_A_fake)
        discriminator_loss_A = (discriminator_loss_input_A_real + discriminator_loss_input_A_fake) / 2

        discriminator_loss_input_B_real = l2_loss(y=tf.keras.ones_like(discrimination_input_B_real),
                                                       y_hat=discrimination_input_B_real)
        discriminator_loss_input_B_fake = l2_loss(y=tf.keras.zeros_like(discrimination_B_fake),
                                                       y_hat=discrimination_B_fake)
        discriminator_loss_B = (discriminator_loss_input_B_real + discriminator_loss_input_B_fake) / 2

        # Merge the two discriminators into one
        discriminator_loss = discriminator_loss_A + discriminator_loss_B


        generator_summaries, discriminator_summaries = self.summary()

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.writer.add_summary(generator_summaries, self.train_step)

        self.train_step += 1





        return generator_loss, discriminator_loss






    def test(self, inputs, direction):

        if direction == 'A2B':
            generation = self.sess.run(self.generation_B_test, feed_dict = {self.input_A_test: inputs})
        elif direction == 'B2A':
            generation = self.sess.run(self.generation_A_test, feed_dict = {self.input_B_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.compat.v1.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)
            generator_loss_A2B_summary = tf.summary.scalar('generator_loss_A2B', self.generator_loss_A2B)
            generator_loss_B2A_summary = tf.summary.scalar('generator_loss_B2A', self.generator_loss_B2A)
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.compat.v1.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_A2B_summary, generator_loss_B2A_summary, generator_loss_summary])

        with tf.compat.v1.name_scope('discriminator_summaries'):
            discriminator_loss_A_summary = tf.summary.scalar('discriminator_loss_A', self.discriminator_loss_A)
            discriminator_loss_B_summary = tf.summary.scalar('discriminator_loss_B', self.discriminator_loss_B)
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.compat.v1.summary.merge([discriminator_loss_A_summary, discriminator_loss_B_summary, discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = CycleGAN(num_features = 24)
    print('Graph Compile Successeded.')