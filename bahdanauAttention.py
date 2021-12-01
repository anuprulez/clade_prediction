import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super().__init__()
    # For Eqn. (4), the  Bahdanau attention
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)
    self.W2 = tf.keras.layers.Dense(units, use_bias=False)

    self.attention = tf.keras.layers.AdditiveAttention()

  def call(self, query, value, mask):

    w1_query = self.W1(query)

    w2_key = self.W2(value)

    query_mask = [] #tf.ones(tf.shape(query)[:-1], dtype=bool)
    value_mask = [] #mask

    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value, w2_key],
        mask=[], #[query_mask, value_mask],
        return_attention_scores = True,
    )

    return context_vector, attention_weights
