# Import Libaray
from nltk.stem import ISRIStemmer
import torch
import scipy
import tensorflow as tf
from scipy.sparse import csr
import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template
import soundfile as sf
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import signal as sg
import math
import os
import subprocess
from pydub import AudioSegment
import tensorflow as tf
from scipy.sparse import csr_matrix
from file import predict_poem_category

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import string
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename

import numpy as np
import os
from pyarabic import araby
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

import pickle
import markovify
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import Tokenizer
import string
from flask import Flask, redirect, render_template, request, url_for
from sklearn.calibration import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

import pandas as pd

# Create a Flask app
app = Flask(__name__)
CORS(app)

# =================================================>[Start] markovify <=================================================================

with open('vocabs.pkl', 'rb') as pickle_load:
    voc_list = pickle.load(pickle_load)

allowed_chars = ['ذ', 'ض', 'ص', 'ث', 'ق', 'ف', 'غ', 'ع', 'ه', 'خ', 'ح', 'ج', 'د',
                 'ش', 'س', 'ي', 'ب', 'ل', 'ا', 'أ', 'ت', 'ن', 'م', 'ك', 'ط', 'ئ', 'ء', 'ؤ', 'ر', 'ى',
                 'ة', 'و', 'ز', 'ظ', 'ّ', ' ']

max_word_length = 9


def rhymes_with_last_n_chars(word, n):
    if word not in ['الله', 'والله', 'بالله', 'لله', 'تالله', 'فالله']:
        word = word.replace('ّ', '')
    ending = word[-n:]
    rhymes = []
    for w in voc_list:
        if len(w) < max_word_length and w.endswith(ending):
            rhymes.append(w)
    return rhymes


def markov(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text_model = markovify.NewlineText(text)
    return text_model


def generate_poem_single_rhyme(poet_name, rhyme, iterations=3000, use_tqdm=False):
    n_of_rhyme_letters = len(rhyme)
    input_file = 'input/{}.txt'.format(poet_name)
    text_model = markov(input_file)
    rhymes_list = rhymes_with_last_n_chars(rhyme, n_of_rhyme_letters)
    bayts = set()
    used_rhymes = set()

    poem = ""

    if use_tqdm == True:
        if hasattr(tqdm, '_instances'):
            tqdm._instances.clear()
        it_range = tqdm(range(iterations))
    else:
        it_range = range(iterations)

    generated_count = 0  # Initialize a count variable

    for i in it_range:
        bayt = text_model.make_short_sentence(280, tries=100)
        if bayt is not None:
            last_word = bayt.split()[-1]
            if (last_word in rhymes_list) and (last_word not in used_rhymes) and (bayt not in bayts):
                bayts.add(bayt)
                used_rhymes.add(last_word)
                poem += "{}\n".format(bayt)
                generated_count += 1  # Increment the count when a poem is generated
                if not use_tqdm:
                    print(bayt)

    # Return the count along with the poem
    return f"Number of Poems Generated: {generated_count}\n\n{poem}"


# =================================================>[End] markovify <=================================================================

# =================================================>[Start] Load model ande functions of denoise audio <=================================================================


def LSD_loss(y_true, y_pred):
    LSD = backend.mean((y_true - y_pred) ** 2, axis=2)
    LSD = backend.mean(backend.sqrt(LSD), axis=1)
    return LSD


model_path = 'audio-denoise.h5'
# model_path=""
audio_model = tf.keras.models.load_model(model_path, custom_objects={'LSD_loss': LSD_loss}, compile=False)

down_sample = 16000  # Downsampling rate (Hz) [Default]16000
frame_length = 0.032  # STFT window width (second) [Default]0.032
frame_shift = 0.016  # STFT window shift (second) [Default]0.016
num_frames = 16  # The number of frames for an input [Default]16


def pre_processing(data, Fs, down_sample):
    # Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5 * data[:, 0] + 0.5 * data[:, 1]
    else:
        wavdata = data
    return wavdata, Fs


def read_evaldata(file_path, down_sample, frame_length, frame_shift, num_frames):
    # Inicialize list
    x = []
    ang_x = []
    # Read .wav file and get pre-process
    wavdata, Fs = sf.read(file_path)
    wavdata, Fs = pre_processing(wavdata, Fs, down_sample)
    # Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    # Execute STFT
    _, _, dft = sg.stft(wavdata, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    dft = dft[:-1].T  # Remove the last point and get transpose
    # Preserves phase information >> to accurately reconstruct the original audio signal from the modified or processed spectrogram
    ang = np.angle(dft)  # Preserve the phase
    spec = np.log10(np.abs(dft))
    # Crop the temporal frames into input size
    num_seg = math.floor(spec.shape[0] / num_frames)
    for j in range(num_seg):
        # Add results to list sequentially
        x.append(spec[int(j * num_frames): int((j + 1) * num_frames), :])
        ang_x.append(ang[int(j * num_frames): int((j + 1) * num_frames), :])
    # Convert into numpy array
    x = np.array(x)
    ang_x = np.array(ang_x)
    return wavdata, Fs, x, ang_x


def reconstruct_wave(eval_y, ang_x, Fs, frame_length, frame_shift):
    # Construct the spectrogram by concatenating all segments
    Y = np.reshape(eval_y, (-1, eval_y.shape[-1]))
    ang = np.reshape(ang_x, (-1, ang_x.shape[-1]))
    # The Y and arg can be transpose for Tensorflow format
    Y, ang = Y.T, ang.T
    # Restore the magnitude of STFT
    Y = np.power(10, Y)
    # Restrive the phase from original wave
    Y = Y * np.exp(1j * ang)
    # Add the last frequency bin along with frequency axis
    Y = np.append(Y, Y[-1, :][np.newaxis, :], axis=0)
    # Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hann', nperseg=FL, noverlap=OL)
    return rec_wav, Fs


def predict_with_model(input_data):
    input_data = input_data[:, :, :, np.newaxis]
    predictions = audio_model.predict(input_data)
    predictions = predictions[:, :, :, 0]
    return predictions


# =================================================>[End] Load model ande Functions of denoise audio <=================================================================

# =================================================>[Start] Defined pages <=================================================================


@app.route('/')
def upload_page():
    return render_template('denoise.html')


def index():
    return render_template('denoise.html')


@app.route('/model1')
def model1_page():
    return render_template('index.html')


@app.route('/model2')
def model2_page():
    return render_template('generation_&_classification.html')


@app.route('/model3')
def model3_page():
    return render_template('input.html')


@app.route('/model8')
def model8_page():
    return render_template('result_3.html')


# classification category

@app.route('/model4')
def model4_page():
    return render_template('inputCat.html')


@app.route('/model5')
def model5_page():
    return render_template('input_3.html')


@app.route('/model6')
def model6_page():
    return render_template('input_4.html')


@app.route('/model9')
def model9_page():
    return render_template('classBahr.html')


@app.route('/modelSpeech')
def model10_page():
    return render_template('indexIdentify.html')


# =================================================>[End] Defined pages <=================================================================

# =================================================>[Start] predict Deniose page <=================================================================


def convert_wav_to_mp3(input_file, output_file):
    try:

        subprocess.run(['ffmpeg', '-i', input_file, '-b:a', '192K', output_file], check=True)
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


@app.route('/denoise_page')
def denoise_page():
    return render_template('denoise.html')


@app.route('/denoise', methods=['POST'])
def denoise_audio():
    try:
        uploaded_file = request.files['audio']
        if uploaded_file.filename == '':
            print("no selected file")
            return render_template('denoise.html', error='No selected file')
        audio_file_path = 'input_sound/input.wav'
        uploaded_file.save(audio_file_path)
        mix_wav, Fs, eval_x, ang_x = read_evaldata(
            audio_file_path, down_sample, frame_length, frame_shift, num_frames)
        min_x = np.min(eval_x)
        max_x = np.max(eval_x)
        eval_x = (eval_x - min_x) / (max_x - min_x)
        eval_y = predict_with_model(eval_x)
        eval_y = eval_y * (max_x - min_x) + min_x
        denoised_wav, Fs = reconstruct_wave(
            eval_y, ang_x, Fs, frame_length, frame_shift)
        print(Fs)
        if denoised_wav.size == 0:
            print("size 0")
            return render_template('denoise.html', message='Denoised audio is empty')
        print(denoised_wav)
        scipy.io.wavfile.write("input_sound/output.wav", Fs, denoised_wav)  # here save the output in the
        input_wav = 'input_sound/output.wav'
        output_mp3 = 'input_sound/output.mp3'
        convert_wav_to_mp3(input_wav, output_mp3)
        return send_file(output_mp3, as_attachment=True)
    except Exception as e:
        print(e)
        return render_template('denoise.html', message=f'Error: {str(e)}')


# =================================================>[End] predict Deniose page <=================================================================

# =================================================>[Start] predict LSTM MODel <=================================================================


@app.route('/generate_poetry', methods=['GET'])
def poetry_form():
    return render_template('input.html')


@app.route('/generate_poetry', methods=['POST'])
def generate_poem():
    poet_name = request.form.get('poet_name')
    rhyme = request.form['rhyme']
    poem = generate_poem_single_rhyme(poet_name, rhyme)
    return render_template('ress.html', poem=poem)


# ==================================================================================================================


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):

        ## Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = False
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


@app.route("/Lstm", methods=['POST', 'GET'])
def generate_poetry():
    if request.method == 'POST':
        data = request.form.get('text')
        prime = data
        top_k = 5
        size = 100
        for word in prime.split():
            for ch in word:
                if not ('\u0600' <= ch <= '\u06FF'):
                    a = {"error": "جميع الحروف يجب ان تكون عربية",
                         "error2": "All letters should be in Arabic"}
                    return render_template('input_3.html', sentiment=a, title=data)

        try:
            size = int(request.form.get('size'))
        except Exception:
            a = {"error": "عدد الحروف يجب ان يكون عددأ صحيحأ",
                 "error2": "size should be a valid number"}
            return render_template('input_3.html', sentiment=a, title=data)

        print(net.load_state_dict(checkpoint['state_dict']))

        net.cpu()
        net.eval()  # eval mode
        train_on_gpu = False
        # First off, run through the prime characters
        chars = [ch for ch in prime]
        h = net.init_hidden(1)
        for char in prime:
            # tensor inputs
            x = np.array([[net.char2int[char]]])
            n_labels = len(net.chars)
            # Initialize the the encoded array
            one_hot = np.zeros((x.size, n_labels), dtype=np.float32)

            # Fill the appropriate elements with ones
            one_hot[np.arange(one_hot.shape[0]), x.flatten()] = 1.

            # Finally reshape it to get back to the original array
            one_hot = one_hot.reshape((*x.shape, n_labels))

            x = one_hot
            inputs = torch.from_numpy(x)

            if (train_on_gpu):
                inputs = inputs.cuda()

            # detach hidden state from history
            h = tuple([each.data for each in h])
            # get the output of the model
            out, h = net(inputs, h)

            # get the character probabilities
            p = F.softmax(out, dim=1).data
            if (train_on_gpu):
                p = p.cpu()  # move to cpu

            # get top characters
            if top_k is None:
                top_ch = np.arange(len(net.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()

            # select the likely next character with some element of randomness
            p = p.numpy().squeeze()
            char = np.random.choice(top_ch, p=p / p.sum())

            char, h = net.int2char[char], h
            # char, h = predict(net, ch, h, top_k=top_k)

        chars.append(char)

        # Now pass in the previous character and get a new one
        for ii in range(size):
            x = np.array([[net.char2int[char[-1]]]])
            n_labels = len(net.chars)
            # Initialize the the encoded array
            one_hot = np.zeros((x.size, n_labels), dtype=np.float32)

            # Fill the appropriate elements with ones
            one_hot[np.arange(one_hot.shape[0]), x.flatten()] = 1.

            # Finally reshape it to get back to the original array
            one_hot = one_hot.reshape((*x.shape, n_labels))

            x = one_hot
            inputs = torch.from_numpy(x)

            if (train_on_gpu):
                inputs = inputs.cuda()

            # detach hidden state from history
            h = tuple([each.data for each in h])
            # get the output of the model
            out, h = net(inputs, h)

            # get the character probabilities
            p = F.softmax(out, dim=1).data
            if (train_on_gpu):
                p = p.cpu()  # move to cpu

            # get top characters
            if top_k is None:
                top_ch = np.arange(len(net.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()

            # select the likely next character with some element of randomness
            p = p.numpy().squeeze()
            char = np.random.choice(top_ch, p=p / p.sum())

            # return the encoded value of the predicted char and the hidden state
            # return net.int2char[char], h

            char, h = net.int2char[char], h
            print(char)
            chars.append(char)
        output = ''.join(chars)
        reshaped_text = output

        a = {}
        s = ""
        for ii in range(len(reshaped_text.split(' '))):
            if (reshaped_text.split(' ')[ii] == "الله"):
                continue

            s += reshaped_text.split(' ')[ii] + " "
            if ii % 5 == 0:
                a[ii] = s
                s = ""

        # return reshaped_text
        return render_template('input_3.html', sentiment=a, title=data)
    else:
        return render_template('input_3.html', sentiment='')


# ======================================<start of classification category>============================================================================

model = pickle.load(open("poem_category_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route('/')
def index():
    return render_template('inputCat.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input poem from the form
        input_poem = request.form.get('poem')
        print(input_poem)
        # Predict the category of the input poem
        predicted_category = predict_poem_category(input_poem.encode('utf-8'))
        return render_template("inputCat.html", result=predicted_category)
    except Exception as e:
        return jsonify({'error': str(e)})


# ======================================<End of Classfication Category>======================================
# =====================================<Start of Speaker Recognition >=============================================
app.config['UPLOAD_FOLDER'] = 'static/Uploaded audio'  # Configure upload folder

model = tf.keras.models.load_model("identification.h5")
class_names = ['Nelson Mandela', 'Benjamin Netanyau', 'Jens Stoltenberg', 'Magaret Tarcher', 'Julia Gillard']


def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64))
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def predict(path):
    sample, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1, desired_samples=16000)
    sample = tf.expand_dims(sample, axis=0)
    fft = audio_to_fft(sample)
    prediction = model.predict(fft)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]


# Set up a route for file upload and speaker identification
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the file was uploaded
        if 'audio' not in request.files:
            return 'No file uploaded'

        uploaded_file = request.files['audio']

        # Check if a file was uploaded
        if uploaded_file.filename == '':
            return 'No file selected'

        if uploaded_file:
            # Set the filename and extension
            filename = secure_filename(uploaded_file.filename)
            filename_with_extension = filename + '.wav'  # Ensure the correct extension
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_with_extension)
            uploaded_file.save(file_path)

            # Predict the speaker ID
            speaker_id = predict(file_path)

            # Return the identified speaker ID as JSON response
            return speaker_id

    return 'Failed to upload or identify speaker'



@app.route('/', methods=['GET'])
def identify():
    return render_template('indexidentify.html')

# =====================================<End of Speaker Recognition >=============================================
# ===============================================<start of classification bahr>==========================================#
# Load the model
verse_model = tf.keras.models.load_model('full_verse.h5', compile=False)

# Define label names
label2name = [
    "السريع", "الكامل", "المتقارب", "المتدارك", "المنسرح",
    "المديد", "المجتث", "الرمل", "البسيط", "الخفيف",
    "الطويل", "الوافر", "الهزج", "الرجز"
]

# load the char2idx mapping
with open('char2idx.pickle', 'rb') as file:
    char2idx = pickle.load(file)


@app.route('/')
def home():
    return render_template('classBahr.html')


@app.route('/predictBahr', methods=['POST'])
def predictt():
    sentence = request.form['sentence']

    # Preprocess the input sentence
    sentence = araby.strip_tashkeel(sentence)
    sequence = [char2idx.get(char, 0) for char in sentence]  # Use 0 for unknown characters
    sequence = pad_sequences([sequence], maxlen=100, padding='post', value=0)

    # Make predictions
    pred = verse_model.predict(sequence)[0]
    predicted_label = label2name[np.argmax(pred, 0).astype('int')], np.max(pred)

    return render_template('resultBahr.html', sentence=sentence, predicted_label=predicted_label)


if __name__ == '__main__':
    global net
    with open('Temp.net', 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))
    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    app.run(debug=False,host='0.0.0.0')
