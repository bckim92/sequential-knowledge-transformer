import tensorflow as tf
import re
import requests
import os
import zipfile


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))).split(' ')


def universal_sentence_embedding(sentences, sentences_length):
    """
    Perform Universal Sentence Encoder averaging (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
    sentences_sum = tf.reduce_sum(sentences, axis=1)
    divisor = tf.expand_dims(tf.sqrt(tf.cast(sentences_length, tf.float32)), axis=-1)
    sentences_sum = sentences_sum / (divisor + tf.keras.backend.epsilon())
    return sentences_sum
    # need to mask out the padded chars
    #sentence_sums = th.bmm(
    #    sentences.permute(0, 2, 1),
    #    mask.float().unsqueeze(-1)
    #).squeeze(-1)
    #divisor = mask.sum(dim=1).view(-1, 1).float()
    #if sqrt:
    #    divisor = divisor.sqrt()
    #sentence_sums /= divisor
    #return sentence_sums


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_from_google_drive(gd_id, destination):
    """Use the requests package to download a file from Google Drive."""
    URL = 'https://docs.google.com/uc?export=download'

    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()


def unzip(path, fname, deleteZip=True):
    """
    Unzip the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool deleteZip:
        If true, the archive will be deleted after extraction.
    """
    print('unzipping ' + fname)
    fullpath = os.path.join(path, fname)
    with zipfile.ZipFile(fullpath, "r") as zip_ref:
        zip_ref.extractall(path)
    if deleteZip:
        os.remove(fullpath)


def main():
    a = normalize_answer("I am a boy")


if __name__ == '__main__':
    main()
