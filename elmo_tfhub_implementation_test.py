import tensorflow as tf
import tensorflow_hub as hub
import pickle


raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .',
    'hello world'
]


tokens_input = [["the", "cat", "is", "on", "the", "mat", "", ""],
                ["dogs", "are", "in", "the", "fog", "", "", ""]]

def elmo_embedding_1(context, tag='word_emb'):
    with tf.Graph().as_default():
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        embeddings = elmo(
            context,
            signature="default",
        as_dict=True)[tag]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            ret = sess.run(embeddings)
            print(ret)
            print(ret.shape)
            print(type(ret))
    return ret

def elmo_embedding_2(context, tag='word_emb'):
    with tf.Graph().as_default():
        elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        embeddings = elmo(
            inputs={
                'tokens': context,
                'sequence_len': [len(i) for i in context],
            },
            signature="tokens",
        as_dict=True)[tag]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            ret = sess.run(embeddings)
            print(ret)
            print(ret.shape)
            print(ret.dtype)
            print(type(ret))
    return ret



if __name__ == '__main__':
    elmo_embedding_2(tokens_input) 





