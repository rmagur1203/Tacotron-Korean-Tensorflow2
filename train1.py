import os
import glob
import random
import traceback
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.tacotron import Tacotron
from util.hparams import *
from util.plot_alignment import plot_alignment
from util.text import sequence_to_text

data_dir = "./data"
text_list = sorted(glob.glob(os.path.join(data_dir + "/text", "*.npy")))
mel_list = sorted(glob.glob(os.path.join(data_dir + "/mel", "*.npy")))
dec_list = sorted(glob.glob(os.path.join(data_dir + "/dec", "*.npy")))

fn = os.path.join(data_dir + "/mel_len.npy")
if not os.path.isfile(fn):
    mel_len_list = []
    for i in range(len(mel_list)):
        mel_length = np.load(mel_list[i]).shape[0]
        mel_len_list.append([mel_length, i])
    mel_len = sorted(mel_len_list)
    np.save(os.path.join(data_dir + "/mel_len.npy"), np.array(mel_len))

text_len = np.load(os.path.join(data_dir + "/text_len.npy"))
mel_len = np.load(os.path.join(data_dir + "/mel_len.npy"))


class DataGenerator(tf.keras.utils.Sequence):
    def __data_generation(self, idx_list):
        text = [np.load(text_list[mel_len[i][1]]) for i in idx_list]
        dec = [np.load(dec_list[mel_len[i][1]]) for i in idx_list]
        mel = [np.load(mel_list[mel_len[i][1]]) for i in idx_list]
        text_length = [text_len[mel_len[i][1]] for i in idx_list]

        text = pad_sequences(text, padding="post")
        dec = pad_sequences(dec, padding="post", dtype="float32")
        mel = pad_sequences(mel, padding="post", dtype="float32")

        return text, dec, mel, text_length


@tf.function(experimental_relax_shapes=True)
def train_step(enc_input, dec_input, dec_target, text_length):
    with tf.GradientTape() as tape:
        pred, alignment = model(enc_input, text_length, dec_input, is_training=True)
        loss = tf.reduce_mean(MAE(dec_target, pred))
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, pred[0], alignment[0]


model = Tacotron(K=16, conv_dim=[128, 128])
model.compile(optimizer=Adam(), loss=tf.keras.losses.MeanSquaredError())
optimizer = Adam()
step = tf.Variable(0)

checkpoint_dir = "./checkpoint/1"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, step=step)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restore checkpoint from {}".format(manager.latest_checkpoint))

try:
    model.fit_generator(
        DataGenerator(),
        steps_per_epoch=1,
        epochs=1,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)],
    )
    # for text, dec, mel, text_length in DataGenerator():
    #     print(mel.shape)
    #     model.fit([text, dec], mel, steps_per_epoch=1)
    # loss, pred, alignment = train_step(text, dec, mel, text_length)
    # checkpoint.step.assign_add(1)
    # print("Step: {}, Loss: {:.5f}".format(int(checkpoint.step), loss))

    # if int(checkpoint.step) % checkpoint_step == 0:
    #     checkpoint.save(
    #         file_prefix=os.path.join(
    #             checkpoint_dir, "step-{}".format(int(checkpoint.step))
    #         ),
    #     )

    #     input_seq = sequence_to_text(text[0])
    #     input_seq = input_seq[: text_length[0]]
    #     alignment_dir = os.path.join(
    #         checkpoint_dir, "step-{}-align.png".format(int(checkpoint.step))
    #     )
    #     plot_alignment(alignment, alignment_dir, input_seq)

except Exception:
    traceback.print_exc()
