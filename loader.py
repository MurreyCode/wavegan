import tensorflow as tf


"""
  Data loader
  fps: List of tfrecords
  batch_size: Resultant batch size
  window_len: Size of slice to take from each example
  first_window: If true, always take the first window in the example, otherwise take a random window
  repeat: If false, only iterate through dataset once
  labels: If true, return (x, y), else return x
  buffer_size: Number of examples to queue up (larger = more random)
"""


def get_batch(
  fps,
  batch_size,
  window_len,
  first_window=False,
  repeat=True,
  labels=False,
  buffer_size=8192):

  def _mapper(example_proto):
    features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
    if labels:
      features['label'] = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

    example = tf.parse_single_example(example_proto, features)
    wav = example['samples']
    if labels:

      # Computes the string join across dimensions in the given string Tensor
	  # Returns a new Tensor created by joining the input strings with the given separator
	  # tensor `a` is [["a", "b"], ["c", "d"]]
	  # tf.reduce_join(a, 0) ==> ["ac", "bd"]
	  label = tf.reduce_join(example['label'], 0)

    if first_window:
      # Use first window
	  wav = wav[:window_len]
    else:
      # Select random window
	  # Retrieve number of samples in example
	  wav_len = tf.shape(wav)[0]

      # Set max starting point to slice the example
	  start_max = wav_len - window_len
      start_max = tf.maximum(start_max, 0)

      start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

      # Extract an slice of the example from a random starting point and size window_len
	  wav = wav[start:start+window_len]

    # Pads a tensor according to the paddings you specify.
	# Paddings is an integer tensor with shape [n, 2],
	# where n is the rank of tensor.
	# For each dimension D of input, paddings[D, 0]
	# indicates how many values to add before the contents
	# of tensor in that dimension, and paddings[D, 1] indicates
	# how many values to add after the contents of tensor in that dimension.
	wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])

    wav.set_shape([window_len, 1])

    if labels:
      return wav, label
    else:
      return wav

  dataset = tf.data.TFRecordDataset(fps)
  dataset = dataset.map(_mapper)

  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)

  # This transformation combines consecutive elements of this dataset into batches.
  # However, if the batch size does not evenly divide the input dataset size,
  # this transformation will drop the final smaller element.
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)) # DEPRECATED
  # dataset = dataset.apply(from_tensor_slices().batch(batch_size, drop_remainder=True))

  if repeat:
    dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next()