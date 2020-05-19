# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:12:39 2020

@author: 12538
"""


import tensorflow as tf

#%%
#首先我们先定义一个全连接层，来对其有一个基本的认识，它应该封装一些权重和一些基本的计算。
from tensorflow.keras import layers
class Linear(layers.Layer):
    
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),dtype='float32'),trainable=True)
        b_init = tf.zeros_initializer()
        
        self.b = tf.Variable(initial_value=b_init(shape=(units,),dtype='float32'),trainable=True)
    def call(self, inputs):
          return tf.matmul(inputs, self.w) + self.b
    

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
#它的实现利用了所有keras层的基类:Layer,在这中call是最重要的函数，它用于实现层的功能，
#layers中魔法函数 __call__ 会将收到的输入传递给 call 函数，然后调用 call 函数实现具体的功能。
#%%
#在对图层权重的设定中，我们更多采用add_weight，它比较方便，如下所所示
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True)
        self.b = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
#%%
#在图层我们可以定义权重是否可以被训练，我们在tf.Variable或者add_weight函数中，将trainable=False即为不参与训练
#我们可以通过layer.weights、layer.non_trainable_weights和layer.trainable_weights查看权重信息。

#在许多情况下，我们可能事先不知道输入的大小，并且想在实例化图层后的某个时间，在该值已知后再去创建权重。
#我们在图层的build（inputs_shape）方法中创建图层权重。
class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

x = tf.ones((2, 2))
linear_layer = Linear(4)
y = linear_layer(x)
print(y)
#Keras图层的内置__call__方法将在首次调用build时自动运行。
#%%层可以是递归组合
#如果您将Layer实例分配为另一个Layer的属性，则外层将开始跟踪内层的权重。通常在__init__方法中创建此类子层。

class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)
    
    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))
#%%层可以在向前传递的过程中带来损失，我们可以创建loss张量，在将在编写训练迭代循环时使用。 这可以通过调用self.add_loss（value）来实现：
class LossLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate
  
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
    
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.loss_fun=LossLayer(1e-2)
    def call(self, inputs):
        return self.loss_fun(inputs)
    
layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above
#%%此外，损失属性还包含为任何内层的权重创建的正则化损失。
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)
#%%在编写训练循环时应考虑这些损失，如下所示：
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Iterate over the batches of a dataset.
for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
    logits = layer(x_batch_train)  # Logits for this minibatch
    # Loss value for this minibatch
    loss_value = loss_fn(y_batch_train, logits)
    # Add extra losses created during this forward pass:
    loss_value += sum(tf.keras.model.losses)

  grads = tape.gradient(loss_value, tf.keras.model.trainable_weights)
  optimizer.apply_gradients(zip(grads, tf.keras.model.trainable_weights))
#%%选择性的将层序列化
  #如果想把定制的层序列化为功能模型的一部分，我们可以选择性地执行‘get_config’方法：
class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {'units': self.units}
#
x = tf.ones((2, 2))
layer = Linear(4)
print(layer(x))
# Now you can recreate the layer from its config:
layer = Linear(4)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
print(new_layer(x))
#%%注意，基层类的_init__方法接受一些关键字参数，特别是‘name’和’dtype’。在_init__中将这些参数传递给父类并将它们包含在层配置中是一个很好的做法:

class Linear(layers.Layer):

  def __init__(self, units=32, **kwargs):
    super(Linear, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
    config = super(Linear, self).get_config()
    config.update({'units': self.units})
    return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
#%%在call方法中给与训练参数特权
#在一些层中，特别是`BatchNormalization`层和`Dropout`层，在训练和推断过程中，有不同的行为。对于这样的层，标准的做法是在调用方法中公开一个训练(布尔)参数。
#通过在call中暴露这个参数，您可以使内置的训练和评估循环(例如fit)在训练和推断中正确使用该层。
class CustomDropout(layers.Layer):

  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
        return tf.nn.dropout(inputs, rate=self.rate)
    return inputs
#%%建立模型
class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save_weights(filepath)
#%%实例化：一个端到端的例子
class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()

  def call(self, inputs):
    x = self.dense_proj(inputs)
    z_mean = self.dense_mean(x)
    z_log_var = self.dense_log_var(x)
    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Converts z, the encoded digit vector, back into a readable digit."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
    self.dense_output = layers.Dense(original_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.dense_proj(inputs)
    return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               original_dim,
               intermediate_dim=64,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
    super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
    self.original_dim = original_dim
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed

original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 3

# Iterate over epochs.
for epoch in range(epochs):
  print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
  for step, x_batch_train in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      reconstructed = vae(x_batch_train)
      # Compute reconstruction loss
      loss = mse_loss_fn(x_batch_train, reconstructed)
      loss += sum(vae.losses)  # Add KLD regularization loss

    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    loss_metric(loss)

    if step % 100 == 0:
      print('step %s: mean loss = %s' % (step, loss_metric.result()))
#%%超越面向对象开发: Functional API
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

# Add KL divergence regularization loss.
kl_loss = - 0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

