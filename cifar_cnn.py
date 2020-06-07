'''
    - cifar net source code.
    - Auther: Ahmed Hendawy.
    - Date: 01.06.2020.
''' 

# YonoArc Utils 
from yonoarc_utils.image import to_ndarray
from yonoarc_utils.header import set_timestamp

# Import tensorflow
import tensorflow as tf
from tensorflow.keras import models, layers

# Vision
import cv2 

# Utils
from queue import Queue 
import numpy as np
import time
import math
import os

# Messages
from yonoarc_msgs.msg import Float64
from std_msgs.msg import Header


class cifar_cnn:
    def __init__(self):
        # Training Objects
        self.model=None
        self.loss_object=None
        self.optimizer=None

        self.train_loss=None
        self.train_accuracy=None

        # Hyperparameters
        self.momentum=0.9 # beta 1
        self.lr=1e-3
        self.batch_size=1
        self.epochs=1

        # Logging Parameters
        self.model_path=''
        self.model_name=''

        # Extra
        self.mini_batches=0
        self.mini_batch_counter=0
        self.dataset_size=0
        
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.image_batch=None
        self.label_batch=[]

        self.batches = Queue(maxsize = 0) 

        self.ind2label={'airplane':0,
                        'automobile':1,
                        'bird':2,
                        'cat':3,
                        'deer':4,
                        'dog':5,
                        'frog':6,
                        'horse':7,
                        'ship':8,
                        'truck':9,}

    def on_start(self):
        tf.random.set_seed(0)
        np.random.seed(0)
        os.environ["USERNAME"] = self.get_property("username")
        os.environ["PASSWORD"] = self.get_property("password")
        # Run the ssh-server
        os.system("(mkdir /home/$USERNAME && useradd $USERNAME && echo $USERNAME:$PASSWORD | chpasswd && usermod -aG sudo $USERNAME && mkdir /var/run/sshd && /usr/sbin/sshd) &")
        self.momentum=self.get_property('momentum')
        self.lr=self.get_property('lr')
        self.batch_size=self.get_property('batch_size')
        self.epochs=self.get_property('epochs')
        self.model_path=self.get_property('model_path')
        self.model_name=self.get_property('model_name')+'.h5'
        self.dataset_size=self.get_property('dataset_size')

        self.mini_batches=math.ceil(self.dataset_size/self.batch_size)
        # self.last_mini_batch_size=self.dataset_size-self.batch_size*self.mini_batches
        self.create_model()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,beta_1=self.momentum)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        

    def run(self):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        self.training()

        
    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

    def train_step(self,images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def on_new_messages(self,messages):
        self.batches.put({'image': messages['input_batch'], 'label': messages['labels']})


    def training(self):
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            self.alert("Epochs: %i / %i"%(epoch+1,self.epochs),"INFO")
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            while self.mini_batch_counter< self.mini_batches:
                if not self.batches.empty():
                    print(self.mini_batch_counter)
                    start=time.time()
                    self.batch_transform()
                    # get the inputs; data is a list of [inputs, labels]
                    images=self.image_batch/255.0
                    labels=self.label_batch
                    self.train_step(images, labels)    
                    self.mini_batch_counter+=1
                    self.image_batch=None
                    self.label_batch=[]
                    print(time.time()-start)
            self.mini_batch_counter=0
            # Publish the loss for every epoch   
            Loss=Float64()
            header=Header()
            set_timestamp(header,time.time())
            Loss.header=header
            Loss.data=self.train_loss.result()

            # Publish the accuracy for every epoch   
            Accuracy=Float64()
            Accuracy.header=header
            Accuracy.data=self.train_accuracy.result() * 100

            self.publish('loss',Loss)
            self.publish('accuracy',Accuracy)
            if not (epoch+1) == self.epochs:
                path=os.path.join(self.model_path,self.model_name)
                self.model.save(path) 
            

        self.alert("Training is Completed","INFO")

        self.alert("Saving the model is started","INFO")
        path=os.path.join(self.model_path,self.model_name)
        self.model.save(path) 
        self.alert("Saving the model is completed","INFO")

    def batch_transform(self):
        batch=self.batches.get()
        untransformed_image_batch, untransformed_label_batch = batch['image'], batch['label']
        self.image_batch=tf.convert_to_tensor(np.array([cv2.cvtColor(to_ndarray(image),cv2.COLOR_BGR2RGB) for image in untransformed_image_batch.Batch]),dtype=tf.float32)
        self.label_batch=tf.convert_to_tensor(np.array([self.ind2label[label.class_name] for label in untransformed_label_batch.labels])) # convert it to integer