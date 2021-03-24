# This is the TensorFlow version of the PyTorch code below
# https://github.com/seungeunrho/minimalRL


import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


class PPO:

    def __init__(self, n_variable=4, lr=0.0005, gamma=0.98, lmbda=0.95, epsilon=0.1, epoch=3, horizon=20):

        self.n_variable = n_variable
        self.lr = lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.epoch = epoch
        self.horizon = horizon

        self.model = None
        self.init_model()
        self.smooth_l1_loss = Huber()
        self.optimizer = Adam(lr=self.lr)

        self.data = []


    def init_model(self):

        inp = Input(shape=(self.n_variable,))
        x = Dense(256, activation="relu", kernel_initializer="he_normal")(inp)
        out_pi = Dense(2, activation="softmax", kernel_initializer="glorot_normal")(x)
        out_v = Dense(1)(x)

        self.model = Model(inp, [out_pi, out_v])


    def put_data(self, data):
        self.data.append(data)


    def get_batch(self):

        samples, samples_next, probs, actions, rewards, dones = [], [], [], [], [], []

        for data in self.data:
            sample, sample_next, prob, action, reward, done = data
            
            samples.append(sample)
            samples_next.append(sample_next)
            probs.append([prob])
            actions.append([action])
            rewards.append([reward])
            mask = 0 if done else 1
            dones.append([mask])
            
        samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        samples_next = tf.convert_to_tensor(samples_next, dtype=tf.float32)
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        samples = tf.reshape(samples, shape=(-1, self.n_variable))
        samples_next = tf.reshape(samples_next, shape=(-1, self.n_variable))

        self.data = []
        
        return samples, samples_next, probs, actions, rewards, dones
        

    def train(self):

        samples, samples_next, probs, actions, rewards, dones = self.get_batch()

        for i in range(self.epoch):
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)

                pi, v = self.model(samples)
                _, v_next = self.model(samples_next)
                
                pi = tf.reshape(pi, shape=(-1, 2))
                v = tf.reshape(v, shape=(-1, 1))
                v_next = tf.reshape(v_next, shape=(-1, 1))
                
                td_target = rewards + tf.constant(self.gamma) * v_next * dones

                delta = td_target - v
                delta = tf.stop_gradient(delta)
                
                advantage_list = []
                advantage = 0.0

                for j in delta[::-1]:
                    advantage = self.gamma * self.lmbda * advantage + j[0]
                    advantage_list.append([advantage])

                advantage_list.reverse()
                advantage_list = tf.convert_to_tensor(advantage_list, dtype=tf.float32)
                advantage_list = tf.reshape(advantage_list, shape=(-1, 1))
                
                actions = tf.reshape(actions, shape=(-1, 1))
                
                pi_action = tf.gather(params=pi, indices=actions, batch_dims=1)
                ratio = tf.exp(tf.math.log(pi_action) - tf.math.log(probs))

                surrogate_loss1 = ratio * advantage_list
                surrogate_loss2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_list
                
                surrogate_loss = -tf.math.minimum(surrogate_loss1, surrogate_loss2)
                smooth_l1_loss = tf.cast(self.smooth_l1_loss(tf.stop_gradient(td_target), v), "float32")
                
                loss = tf.reduce_mean(surrogate_loss + smooth_l1_loss)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def main(n_episode=1000, print_interval=20):

    ppo = PPO()
    env = gym.make("CartPole-v1")
    score = 0.0
    scores = []

    for i in range(n_episode):
        sample = env.reset()
        sample = np.array(sample).reshape(-1, ppo.n_variable)
        done = False

        while not done:
            for j in range(ppo.horizon):
                probs,_ = ppo.model(sample)
                probs = probs.numpy()[0]

                action = np.random.choice([0, 1], size=1, p=probs)[0]
                prob = probs[action]

                sample_next, reward, done, info = env.step(action)
                sample_next = np.array(sample_next).reshape(-1, ppo.n_variable)

                ppo.put_data((sample, sample_next, prob, action, reward / 100.0, done))
                sample = sample_next

                score += reward

                if done:
                    break

            ppo.train()

        if i != 0 and i % print_interval == 0:
            print("Episode: %d  Avg Score: %.2f" % (i, score / print_interval))
            
            scores.append(score)
            score = 0.0

    env.close()
    
    return scores


if __name__ == "__main__":
    main()
    