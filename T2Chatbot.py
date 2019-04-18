from __future__ import absolute_import, division, print_function

import tensorflow as tf
# import os
# os.environ['DISABLE_COLAB_TF_IMPORT_HOOK'] = '1'
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
import re
import os
import io
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import psutil

tf.enable_eager_execution()

# p = psutil.Process(os.getpid())
# p.nice(19)

# path_zip = tf.keras.utils.get_file('cornell_movie_dialogs_corpus.zip',
#                                   'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
#                                   extract=True)
path_zip = tf.keras.utils.get_file('spa.txt',
                                   'https://download2262.mediafire.com/og22qeeb7qgg/ladcl5u218fqs5n/spa.txt', )
#                                   extract=True)

path = os.path.dirname(path_zip) + '/spa.txt'
#path ='./spa.txt'
print(path)


def preproccess(sentence):
    sentence = re.sub(r"([?!.,¿])", r" \1 ", sentence)
    sentence = re.sub(r"([' ']+)", r" ", sentence)
    sentence = re.sub(r"([^a-zA-z?.!,¿]+)", r" ", sentence)

    sentence = sentence.rstrip().strip()
    return '<start> ' + sentence.lower() + ' <end>'


def creat_dataset(path, num_examples):
    file = io.open(path, encoding='UTF-8', errors='ignore')

    lines = file.read()
    lines = lines.split('\n')

    pops = []
    real_index = 0
    for i, x in enumerate(lines):
        if len(x.split(' ')) > 20:
            pops.append(i)
    for c in pops:
        # lines.pop(c - real_index)

        if c % 2 == 0:

            lines.pop(c - real_index)
            lines.pop(c - real_index)
            real_index += 1

            real_index += 1
        else:

            lines.pop(c - real_index - 1)
            real_index += 1
            lines.pop(c - real_index)
            real_index += 1

    '''for i in range(len(lines)):
        if lines[i].rfind('+') != -1:
            lines[i] = lines[i][lines[i].rfind('+') + 1:]
        else:
            lines[i] = lines[i][lines[i].rfind(':') + 1:]'''
    lines = [preproccess(i) for i in lines]
    print(len(lines))
    request_list = lines[:num_examples:2]

    response_list = lines[1:num_examples:2]

    '''for i, x in enumerate(request_list):
        if len(x.split(' ')) > 10:
            request_list.pop(i)
            response_list.pop(i)

    for i, x in enumerate(response_list):
        if len(x.split(' ')) > 10:
            request_list.pop(i)
            response_list.pop(i)'''

    return request_list, response_list


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples):
    input, target = creat_dataset(path, num_examples)

    input_tensor, input_tokenizer = tokenize(input)
    target_tensor, target_tokenizer = tokenize(target)

    return input_tensor, target_tensor, input_tokenizer, target_tokenizer


def max_tensor(tensor):
    return max([len(c) for c in tensor])


input_tensor, target_tensor, input_tokenizer, target_tokenizer = load_dataset(path, 40000)

max_input_length = max_tensor(input_tensor)
max_target_length = max_tensor(target_tensor)

print("MAX")
print(max_input_length)
print(max_target_length)


def convert(tensor, lang):
    for i in tensor:
        if i != 0:
            print("{} -----> {}".format(i, lang.index_word[i]))


convert(input_tensor[0], input_tokenizer)

train_input_tensor, test_input_tensor, train_target_tensor, test_target_tensor = train_test_split(input_tensor,
                                                                                                  target_tensor,
                                                                                                  test_size=0.2)

BUFFER_SIZE = len(train_input_tensor)
BATCH_SIZE = 64
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
# steps_per_epoch = steps_per_epoch // 8
print("here")
print(steps_per_epoch)
embedding_dim = 512
units = 1024

vocab_input_size = len(input_tokenizer.word_index) + 1
vocab_target_size = len(target_tokenizer.word_index) + 1

print(vocab_input_size)
print(vocab_target_size)

dataset = tf.data.Dataset.from_tensor_slices((train_input_tensor, train_target_tensor)).shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)

example_input_tensor, example_target_tensor = next(iter(dataset))


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.enc_units = enc_units
        self.batch_size = batch_size

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=0.2,
                                       recurrent_dropout=0.2)

    def call(self, tensors, hidden):
        v = self.embedding(tensors)
        output, hidden = self.gru(v, initial_state=hidden)
        return output, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)

enc_hidden = encoder.initialize_hidden_state()

enc_output, enc_hidden = encoder(example_input_tensor, enc_hidden)

print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, tensor, hidden):
        hidden = tf.expand_dims(hidden, axis=1)

        score = self.V(tf.nn.tanh(self.W1(tensor) + self.W2(hidden)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * tensor

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=0.2,
                                       recurrent_dropout=0.2)

        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)

    def call(self, dec_input, dec_hidden, enc_output):
        context_vector, attention_weights = self.attention(enc_output, dec_hidden)
        v = self.embedding(dec_input)
        # we expand dimention for context vector because we reduced the axis in the BahdanauAttention
        v = tf.concat([tf.expand_dims(context_vector, 1), v], axis=-1)

        dec_output, dec_hidden = self.gru(v, initial_state=dec_hidden)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        v = self.fc(dec_output)

        return v, dec_hidden, attention_weights


decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)
dec_hidden = enc_hidden
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), dec_hidden, enc_output)

optimizer = tf.train.AdamOptimizer()


def loss_fun(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


checkpoints_dir = './training_checkpoints3'
checkpoint_prefix = os.path.join(checkpoints_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# print(tf.expand_dims([target_lang.word_index['<start>']] * BATCH_SIZE, 1))
# print('------')
# for x, y in dataset.take(1):
#    for t in range(y.shape[1]):
#        print(tf.expand_dims(y[:, t], 1))

def train_step(inp, target, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_state = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # we start from 1 because we already checked <start>
        for t in range(1, target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_fun(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = loss / int(target.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


'''EPOCHS = 40

for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    print(steps_per_epoch)
    for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input, target, enc_hidden)
        total_loss += batch_loss
        print("I ======> {}".format(batch))
        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch, batch, batch_loss))

    # if epoch % 2 == 0:
    checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch, total_loss / steps_per_epoch))

    print("The time taken for 1 epoch is {} sec \n".format(time.time() - start))'''


def evaluate(sentence):
    attention_plot = np.zeros((max_target_length, max_input_length))
    print(attention_plot.shape)
    sentence = preproccess(sentence)

    inputs = [input_tokenizer.word_index[c] for c in sentence.split(' ')]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_input_length,
                                                           padding='post')

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]

    enc_output, enc_hidden = encoder(inputs, hidden)

    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)
    #    print('-----')
    #    print(inputs)
    #    print([inputs])

    dec_hidden = enc_hidden

    for t in range(max_target_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)
        # print(predictions.shape)
        # print(attention_weights.shape)
        attention_weights = tf.reshape(attention_weights, (-1,))
        # print(attention_weights.shape)
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_tokenizer.index_word[predicted_id] + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
        if target_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels(['' + sentence], fontdict=fontdict, rotation=90)

    ax.set_yticklabels(['' + predicted_sentence], fontdict=fontdict)
    plt.show()


def response(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    print(sentence.split(' '))
    print(result.split(' '))
    print(attention_plot.shape)

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

    plot_attention(attention_plot, sentence, result)

    print("Input: {}".format(sentence))
    print("Predicted: {}".format(result))

checkpoints_dir = './14.4 training'
checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir))
#checkpoint.restore(checkpoints_dir+'/ckpt-19')
# print(tf.train.latest_checkpoint(checkpoints_dir))
# print(input_tokenizer.word_index['they'])
# response('Where i can find the Coordintaesٌ')

# print(steps_per_epoch)
# print(len(train_input_tensor))

#response('hello')
for i in range(50):
    sent = input('Sentence:')
    if sent == '123':
        break

    response(sent)

