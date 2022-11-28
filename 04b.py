import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import datetime
import tqdm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

class FFN(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"),
                             tf.keras.metrics.CategoricalAccuracy(name="acc")]
        
        self.optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        # layers to be used
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out_layer = tf.keras.layers.Dense(19,activation=tf.nn.softmax) 
        
    @tf.function
    def call(self, images, training=False):
        img1, img2 = images
        
        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)
        img1_x = self.dense3(img1_x)
        img1_x = self.dense4(img1_x)
        
        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)
        img2_x = self.dense3(img2_x)
        img2_x = self.dense4(img2_x)
        
        combined_x = tf.concat([img1_x, img2_x], axis=1)
        
        return self.out_layer(combined_x)
     
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model

    # 4. reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    # 5. train step method
    @tf.function
    def train_step(self, data):
        img1, img2, target = data
        
        with tf.GradientTape() as tape:
            output = self((img1, img2), training=True)
            loss = self.loss_function(target, output)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(target, output)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        img1, img2, target = data

        output = self((img1, img2), training=False)
        loss = self.loss_function(target, output)

        self.metrics[0].update_state(loss)
        # for accuracy metrics:
        for metric in self.metrics[1:]:
            metric.update_state(target, output)

        return {m.name: m.result() for m in self.metrics}

def training_loop(model, train_ds, val_ds, epochs, train_summary_writer, val_summary_writer):
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        
        # Training:
        
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
            
            # logging the validation metrics to the log file which is used by tensorboard
            with train_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)

        # print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()    
        
        # Validation:
        for data in val_ds:
            metrics = model.test_step(data)
        
            # logging the validation metrics to the log file which is used by tensorboard
            with val_summary_writer.as_default():
                for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
                    
        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

        # reset all metrics
        model.reset_metrics()
        print("\n")

(train_ds, test_ds), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(data, batch_size):
    # image should be float
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    # image should be flattened
    data = data.map(lambda x, t: (tf.reshape(x, (-1,)), t))
    # image vector will here have values between -1 and 1
    data = data.map(lambda x,t: ((x/128.)-1., t))
    # we want to have two mnist images in each example
    # this leads to a single example being ((x1,y1),(x2,y2))
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), 
                                     data.shuffle(2000)))
    # map ((x1,y1),(x2,y2)) to (x1,x2, y1==y2*) *x1[1] + x2[1] >= 5
    zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], x1[1] - x2[1]))
    # transform 
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.int32)))
    # target vector should be one-hot encoded
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.one_hot(t, 19,dtype=tf.int32)))
    # batch the dataset
    zipped_ds = zipped_ds.batch(batch_size)
    # prefetch
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds

# Define where to save the log
config_name= "config_name"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{config_name}/{current_time}/SGDwMomentum/train"
val_log_path = f"logs/{config_name}/{current_time}/SGDwMomentum/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)

# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)

model = FFN()

training_loop(model=model,
                train_ds=preprocess(train_ds,batch_size = 128), 
                val_ds=preprocess(test_ds,batch_size = 128), 
                epochs=30, 
                train_summary_writer=train_summary_writer, 
                val_summary_writer=val_summary_writer)