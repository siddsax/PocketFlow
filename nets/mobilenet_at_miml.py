import tensorflow as tf

from nets.abstract_model_helper import AbstractModelHelper
from datasets.miml_dataset import MIMLDataset
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from nets.mobileNet import mobileNetv2
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\' ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 3e-4, 'weight decaying loss\'s coefficient')

def forward_fn(inputs, data_format, is_train):
  """Forward pass function.

  Args:
  * inputs: inputs to the network's forward pass
  * data_format: data format ('channels_last' OR 'channels_first')

  Returns:
  * inputs: outputs from the network's forward pass
  """

  # transpose the image tensor if needed
  if data_format == 'channel_first':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  image_wh, labels = 224, 5

  # base_net = tf.keras.applications.MobileNetV2(input_shape=(image_wh, image_wh, 3), include_top=False)
  # base_net.trainable = False
  # inputs = base_net(inputs)
  # inputs = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)(inputs)
  # inputs = tf.layers.dense(inputs, labels, activation=tf.nn.softmax)
  # import pdb;pdb.set_trace()
  inputs = mobileNetv2(inputs, labels, is_train=is_train)
  
  
  # tf.keras.Sequential([
  #   base_net,
  #   ,
  #   tf.keras.layers.Dense(labels, activation = 'softmax')])

  # model.compile(optimizer=tf.keras.optimizers.Adam(),
  #               loss='categorical_crossentropy',
  #               metrics=['accuracy'])

  # inputs = model(inputs)


  return inputs

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a ConvNet model for the Fashion-MNIST dataset."""

  def __init__(self,  data_format='channels_last'):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__(data_format)

    # initialize training & evaluation subsets
    self.dataset_train = MIMLDataset(is_train=True)
    self.dataset_eval = MIMLDataset(is_train=False)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build(enbl_trn_val_split)

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs, data_format='channels_last'):
    """Forward computation at training."""

    return forward_fn(inputs, data_format, is_train=True)

  def forward_eval(self, inputs, data_format='channels_last'):
    """Forward computation at evaluation."""

    return forward_fn(inputs, data_format, is_train=False)

  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])
    accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), tf.float32))
    metrics = {'accuracy': accuracy}

    return loss, metrics

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    nb_epochs = 160
    idxs_epoch = [40, 80, 120]
    decay_rates = [1.0, 0.1, 0.01, 0.001]
    batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
    nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

    return lrn_rate, nb_iters

  @property
  def model_name(self):
    """Model's name."""

    return 'mobilenet'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'MIML'