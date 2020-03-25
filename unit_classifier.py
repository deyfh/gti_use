import datetime
import os
import sys

import cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import csv
from gensim.models import KeyedVectors
import pandas as pd

"""
DIR PATH
"""
IMAGE_DIR = 'merged_image1/'
TEST_IMAGE_DIR = 'merged_image_test/'
LOG_DIR = 'log'
TRAIN_DATA_DIR = 'data/lstm_n.csv'
TEST_DATA_DIR = 'data/lstm_test.csv'

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

WEIGHTS_DIR = os.path.join(LOG_DIR, 'weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

TENSORBOARD_DIR = os.path.join(LOG_DIR, 'tensorboard')
if not os.path.exists(TENSORBOARD_DIR):
    os.mkdir(TENSORBOARD_DIR)

"""
Global Var
"""
IMAGE_SIZE = (64, 64, 3)
IMAGE_W_H = 64
FIX_CHARS = 200  # 句子长度
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def create_model(vocab_size, word_num, embedding_dim, embedding_matrix):
    """
        Create Model
    """
    print('Create Model')
    # Inputs
    txt_input = keras.layers.Input(shape=vocab_size, name='txt_input')
    img_input = keras.layers.Input(shape=IMAGE_SIZE, name='image_input')

    # Text layer
    txt_layer = keras.layers.Embedding(input_dim=word_num, output_dim=embedding_dim,
                                       weights=[embedding_matrix], input_length=vocab_size,
                                       trainable=False, name='Embedding')(txt_input)
    txt_layer = keras.layers.Bidirectional(keras.layers.LSTM(64), name='RNN')(txt_layer)
    txt_layer = keras.layers.Dense(64, activation='relu', name='txt_Dense')(txt_layer)

    # Image layer
    img_layer = keras.layers.Conv2D(32, (3, 3), activation='relu', name='Con1')(img_input)
    img_layer = keras.layers.MaxPooling2D((2, 2), name='MaxPooling1')(img_layer)
    img_layer = keras.layers.Conv2D(32, (3, 3), activation='relu', name='Con2')(img_layer)
    img_layer = keras.layers.MaxPooling2D((2, 2), name='MaxPooling2')(img_layer)
    img_layer = keras.layers.Conv2D(32, (3, 3), activation='relu', name='Con3')(img_layer)
    img_layer = keras.layers.Flatten(name='img_Flattern')(img_layer)
    img_layer = keras.layers.Dense(100, activation='relu', name='img_Dense1')(img_layer)

    # Connect
    layer = keras.layers.Concatenate(axis=1, name='Concatenate')([txt_layer, img_layer])
    layer = keras.layers.Dense(64, activation='relu', name='All_Dense1')(layer)

    # Output
    out = keras.layers.Dense(3, name='Output')(layer)

    # Model [txt_input, img_input]
    model = keras.models.Model(inputs=[txt_input, img_input], outputs=out, name='DoubleNet')

    # Show message
    print(model.summary())
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def image_load(id_list,dir):
    """
    Load Images and resize it .
    :param id_list: the load id list
    :return: image list
    """
    print('Loading images')
    counter = 0
    image_list = []
    DIR = dir
    for id in id_list:
        # read image
        name = os.path.join(DIR, (id + '.jpg'))

        # no this image
        if not os.path.exists(name):
            image_list.append(np.zeros(shape=(IMAGE_W_H, IMAGE_W_H, 3), dtype='uint8'))
            counter += 1
            continue
        image = cv2.imread(name)
        image = cv2.resize(image, (IMAGE_W_H, IMAGE_W_H))
        image_list.append(image)
    image_list = np.asarray(image_list)
    print('Summary: {} images not found, use ZeroImage instead'.format(counter))
    return image_list


def tb_init():
    """
    tensorboard init
    :return: callback
    """
    log_dir = os.path.join(TENSORBOARD_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # '/' +
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


def word_all(file_path, cn_model):
    """
    统计训练集中单词
    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        words = []
        num = 0

        for row in reader:
            if row[1] != 'text':
                text = row[1].split()
                for j in text:
                    if j not in words:
                        words.append(j)

        for word in words:
            if word in cn_model:
                num += 1

    return words, num


def get_vects(num_words, all_word, embedding_dim, cn_model):
    '''
    从词典中返回训练集中用到的词向量
    :param num_words:
    :param embedding_dim:
    :return:
    '''
    index = 0
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word in all_word:
        if word in cn_model:
            embedding_matrix[index, :] = cn_model[word]
            index += 1
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix


def text_tokenizer(file_path, cn_model, num_words, max_tokens=FIX_CHARS):
    '''
    单词序列化
    :param file_path:
    :param max_tokens:
    :return:  定长的句子向量
    '''
    train_tokens = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] != 'text':
                cut_list = row[1].split()
                for i, word in enumerate(cut_list):
                    try:
                        # 将词转换为索引index  'data/val.csv'
                        cut_list[i] = cn_model.vocab[word].index
                    except KeyError:
                        # 如果词不在字典中，则输出0
                        cut_list[i] = 0
                train_tokens.append(cut_list)
    train_pad = keras.preprocessing.sequence.pad_sequences(train_tokens, maxlen=max_tokens,
                                                           padding='pre', truncating='pre')  # 补0位置
    train_pad[train_pad >= num_words] = 0
    return train_pad


def get_label(file_path):
    '''
    label向量化
    :param file_path:
    :return:
    '''
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        labels = []
        for row in reader:
            if row[2] != 'label':
                text = int(row[2])
                labels.append(text)
    labels = np.array(labels)
    label = labels  # keras.utils.np_utils.to_categorical(labels, num_classes=3)
    return label, labels


def text_load(src_path):
    '''
    load text and vectorize
    :return:
    '''
    print('Loading text informations from {}'.format(src_path))
    cn_model = KeyedVectors.load_word2vec_format(src_path,
                                                 binary=False, unicode_errors="ignore")
    embedding_dim = cn_model['帅'].shape[0]
    all_word, num_words = word_all(TRAIN_DATA_DIR, cn_model)
    embedding_matrix = get_vects(num_words, all_word, embedding_dim, cn_model)
    txt = text_tokenizer(TRAIN_DATA_DIR, cn_model, num_words)
    train_label, txt_target = get_label(TRAIN_DATA_DIR)
    print('load text maybe OK!')
    return txt, txt_target, embedding_dim, embedding_matrix, num_words


def predict_text_load(src_path):
    '''
    Load the Predict data set
    :param src_path:
    :return:
    '''
    print('Loading predict text from {}'.format(src_path))
    cn_model = KeyedVectors.load_word2vec_format(src_path,
                                                 binary=False, unicode_errors="ignore")
    all_word, num_words = word_all(TRAIN_DATA_DIR, cn_model)
    txt = text_tokenizer(TEST_DATA_DIR, cn_model, num_words)
    print('Load OK')
    return txt


def get_id(file_path):
    '''
    返回可用id
    :param file_path:
    :return: id list
    '''
    print('Loading id')
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        id_list = []
        for row in reader:
            if row[0] != 'id':
                id_list.append(row[0][:-1])
        df = pd.DataFrame(columns=None, data=id_list)
        df.to_csv('data/id_list.csv', encoding='utf-8', index=False, header=None)
    return id_list


def just_for_test(file_path, id_list):
    '''
    test 4 id
    :param file_path:
    :param id_list:
    :return:
    '''
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        iters = []

        for row in reader:
            if row[0] in id_list:
                iters.append(row)
        df = pd.DataFrame(columns=None, data=iters)
        df.to_csv('data/test.csv', encoding='utf-8', index=False, header=None)


def csv_submit(values):
    '''
    prob to index
    :param values:
    :return: csv file
    '''
    result = []
    result_dict = {}
    data = pd.read_csv(TEST_DATA_DIR, encoding='utf-8',usecols=['id','label'])
    data.to_csv('data/submit_null.csv', index=False, encoding='utf-8')
    for line in values:
        line = line.tolist()
        l = line.index(max(line))
        result.append(l - 1)
    with open('data/submit_null.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if row[0] == 'id':
                continue
            result_dict[row[0]] = str(result[i])
            i += 1
    print(result_dict)
    df = pd.DataFrame(list(result_dict.items()))
    df.to_csv('data/submit.csv', index=False, encoding='utf-8',
              header=['id', 'label'])
    print('Submit your csv file from {}'.format('data/submit.csv'))
    return


def train():
    # tensorboard init
    tb_callback = tb_init()

    # id_list
    # id_list = ['446873841959958', '447002827360527', '447125915930530', '447236090258588']

    # text load
    txt, txt_target, embedding_dim, embedding_matrix, num_words = text_load('embeddings/sgns.zhihu.bigram')

    # save model parameters
    np.save(os.path.join(WEIGHTS_DIR,'FIX_CHAR.npy'), FIX_CHARS)
    np.save(os.path.join(WEIGHTS_DIR, 'num_words'), num_words)
    np.save(os.path.join(WEIGHTS_DIR, 'embedding_dim'), embedding_dim)

    with open(os.path.join(WEIGHTS_DIR,'parameters.txt'),mode='w' ) as f:
        content = str(FIX_CHARS)+'-'+str(num_words)+'-'+str(embedding_dim)
        f.write(content)
        f.close()
    np.save(os.path.join(WEIGHTS_DIR,'embedding_matrix'), embedding_matrix)

    # load image
    id_list = get_id(TRAIN_DATA_DIR)
    img = image_load(id_list,dir=IMAGE_DIR)

    # model init
    model = create_model(FIX_CHARS, num_words, embedding_dim, embedding_matrix)

    """
    TODO If you want to custom validation set , change the dir bellow, and change fit() function
    """
    # validate_id_list = get_id(TRAIN_DATA_DIR)
    # validate_txt, validate_txt_target, _, _, _ = text_load('embeddings/sgns.zhihu.bigram')
    # validate_img = image_load(validate_id_list)
    # validate_feed = [validate_txt, validate_img]

    # input data
    feed = [txt, img]

    # Train
    model.fit(feed, txt_target, batch_size=256, epochs=50,
              callbacks=[tb_callback],  # tensorboard callback
              # validation_data=(validate_feed, validate_txt_target)
              validation_split=0.2
              )
    # Save Answer
    model.save_weights(filepath=os.path.join(WEIGHTS_DIR, 'weights.h5'), overwrite=True)


def predict():
    # load model parameters
    print('Running Predict')
    try:
        with open(os.path.join(WEIGHTS_DIR,'parameters.txt'), mode='r') as f:
            content = f.readline()
            f.close()
        content = content.split('-')
        char_len = int(content[0])
        num_words = int(content[1])
        embedding_dim = int(content[2])
        embedding_matrix = np.load(os.path.join(WEIGHTS_DIR,'embedding_matrix.npy'))
    except Exception:
        print("Load parameters Failed")
        os._exit(0)

    model = create_model(char_len, num_words, embedding_dim, embedding_matrix)
    try:
        print('Loading weights')
        model.load_weights(os.path.join(WEIGHTS_DIR, 'weights.h5'))
    except Exception:
        print('Load model weights Failed')
        os._exit(0)

    # text load
    txt = predict_text_load('embeddings/sgns.zhihu.bigram')
    # TODO input predict dir
    id_list = get_id(TEST_DATA_DIR)
    img = image_load(id_list,dir=TEST_IMAGE_DIR)
    feed = [txt, img]

    ans = model.predict(feed)
    # TODO process ans , save in csv
    print(ans)
    csv_submit(ans)
    print('Predict finish, Good luck bro !')


if __name__ == '__main__':
    args = sys.argv
    print(tf.version.VERSION)

    if(args.__contains__('predict')):
        predict()
    else:
        train()