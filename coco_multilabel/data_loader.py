from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import json, os, string, random, time, pickle, pdb, sys
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

# Definition of dataset.
class CocoMultilabelWords(data.Dataset):
    def __init__(self, annotation_dir, image_dir, split = 'train', transform = None):
        # Load training data.
        self.split = split
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations.
        print(('\nLoading %s object annotations...') % self.split)
        self.objData = json.load(open(os.path.join(annotation_dir, 'captions_' + self.split + '2014.json')))
        self.imageIds = [entry['id'] for entry in self.objData['images']]
        self.imageNames = [entry['file_name'] for entry in self.objData['images']]
        self.imageId2index = {image_id: idx for (idx, image_id) in enumerate(self.imageIds)}

        if os.path.exists("coco_words_vocabulary.p"):
            self.vocabulary = pickle.load(open('coco_words_vocabulary.p'))
        else:
            self.vocabulary = get_vocab(self.objData)

        print('\nPreparing label space...')
        lem = WordNetLemmatizer()
        self.labels = np.zeros((len(self.objData['images']), len(self.vocabulary[0])))
        for (i, entry) in enumerate(self.objData['annotations']):
            if i % 10000 == 0: print('.'),
            image_id = entry['image_id']
            caption = entry['caption']
            for word in word_tokenize(caption.lower()):
                word = lem.lemmatize(word)
                if word in self.vocabulary[1].keys():
                    self.labels[self.imageId2index[image_id], self.word2id(word)] = 1

    def getLabelWeights(self):
        return (self.labels == 0).sum(axis = 0) / self.labels.sum(axis = 0)

    def decodeCategories(self, labelVector):
        return [self.id2word(idx) for idx in np.nonzero(labelVector)[0]]

    def id2word(self, idx):
        return self.vocabulary[0][idx]

    def word2id(self, word):
        return self.vocabulary[1][word]

    def imageName(self, index):
        return self.split + '2014/' + self.imageNames[index]

    def __getitem__(self, index):
        split_str = self.split if (self.split != 'test') else 'val'
        imageName_ = split_str + '2014/' + self.imageNames[index]
        img_ = pil_loader(os.path.join(self.image_dir, imageName_))
        if self.transform is not None:
            img_ = self.transform(img_)
        return img_, torch.Tensor(self.labels[index, :]), index

    def __len__(self):
        return len(self.imageIds)

    def numCategories(self):
        return len(self.vocabulary[0])

def loadData(args, trainTransform, testTransform, vocab=None, vocab_aug=None):
    trainData = CocoMultilabelWords(args.annDir, args.imageDir, split = 'train', transform = trainTransform)
    valData = CocoMultilabelWords(args.annDir, args.imageDir, split = 'val', transform = testTransform)
    return trainData, valData

def loadValData(args, testTransform, vocab=None):
    if os.path.exists("coco_words_vocabulary.p"):
        vocabulary = pickle.load(open('coco_words_vocabulary.p'))
    else:
        objData = json.load(open(os.path.join(args.annDir, 'captions_' + 'train' + '2014.json')))
        vocabulary = get_vocab(objData)
    valData = CocoMultilabelWords(args.annDir, args.imageDir, split = 'val', transform = testTransform)
    return valData

def loadTestData(args, testTransform, vocab=None):
    if os.path.exists("coco_words_vocabulary.p"):
        vocabulary = pickle.load(open('coco_words_vocabulary.p'))
    else:
        objData = json.load(open(os.path.join(args.annDir, 'captions_' + 'train' + '2014.json')))
        vocabulary = get_vocab(objData)
    testData = CocoMultilabelWords(args.annDir, args.imageDir, split = 'test', transform = testTransform)
    return testData

def get_vocab(objData):
    spunctuation = set(string.punctuation)
    swords = set(stopwords.words('english'))
    print('Building vocabulary of words...')
    lem = WordNetLemmatizer()
    word_counts = dict()
    for (i, entry) in enumerate(objData['annotations']):
        if i % 10000 == 0: print('.'),
        caption = entry['caption']
        for word in word_tokenize(caption.lower()):
            word = lem.lemmatize(word)
            if word not in swords and word not in spunctuation:
                word_counts[word] = 1 + word_counts.get(word, 0)
    sword_counts = sorted(word_counts.items(), key = lambda x: -x[1])
    id2word = {idx: word for (idx, (word, count)) in enumerate(sword_counts[:1000])}
    id2count = {idx: count for (idx, (word, count)) in enumerate(sword_counts[:1000])}
    word2id = {word: idx for (idx, word) in id2word.iteritems()}
    vocabulary = (id2word, word2id, id2count)
    pickle.dump(vocabulary, open('coco_words_vocabulary.p', 'wb'))

    return vocabulary

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
