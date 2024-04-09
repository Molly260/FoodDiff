import os
import os.path as osp
import sys
import time
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import clip as clip
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)


def get_fix_data(train_dl, test_dl, text_encoder, IFBlock, args):
    fixed_image_train, _, _, fixed_sent_train, fixed_word_train, fixed_key_train= get_one_batch_data(train_dl, text_encoder,  IFBlock, args)
    fixed_image_test, _, _, fixed_sent_test, fixed_word_test, fixed_key_test= get_one_batch_data(test_dl, text_encoder,  IFBlock, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    fixed_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_word, fixed_noise


def get_one_batch_data(dataloader, text_encoder, IFBlock, args):
    data = next(iter(dataloader))
    imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, IFBlock, args.device)
    return imgs, captions, CLIP_tokens, sent_emb, words_embs, keys


def prepare_data(data, text_encoder, device):
    imgs, captions, CLIP_tokens, keys, ingre_tokens, instruc_tokens = data
    imgs, CLIP_tokens = imgs.to(device), CLIP_tokens.to(device)
    sent_emb, words_embs = encode_tokens(text_encoder, CLIP_tokens)

    #sent_ingre_tokens_emb, words_ingre_tokens_embs = encode_tokens(text_encoder,ingre_tokens)
    sent_instruc_tokens_emb=[]
    sent_ingre_tokens_emb = []
    for token in instruc_tokens:
        token=token.to(device)
        instruc_tokens_emb, _ = encode_tokens(text_encoder,token)
        sent_instruc_tokens_emb.append(instruc_tokens_emb)
    for token in ingre_tokens:
        token=token.to(device)
        ingre_tokens_emb, _ = encode_tokens(text_encoder,token)
        sent_ingre_tokens_emb.append(ingre_tokens_emb)
    
    #fus_emb=IFBlock(sent_emb, sent_instruc_tokens_emb, sent_ingre_tokens_emb)
    fus_emb=sent_emb
    return imgs, captions, CLIP_tokens, fus_emb, sent_emb, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 

def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

def get_vireo(title,ingredients):
    ingre_tokens = []
    instruc_tokens=[]
    for ingredient in ingredients:
        ingre_tokens.append(clip.tokenize(ingredient,truncate=True)[0])
    tmp=torch.as_tensor(torch.zeros([77]),dtype=torch.long)    
    if len(ingre_tokens)>20:
        ingre_tokens=ingre_tokens[:20]
    else:
        for i in range(20-len(ingre_tokens)):
            ingre_tokens.append(tmp)
    ingredients=",".join(ingredients)
    caption = title + ";" +ingredients
    tokens=clip.tokenize(caption,truncate=True)
    return caption, tokens[0], ingre_tokens, instruc_tokens

def get_recipe(title, ingredients ,instructions):

    ingre_tokens = [ ]
    for ingredient in ingredients:
        ingre_tokens.append(clip.tokenize(ingredient,truncate=True)[0])

    instruc_tokens = []
    for instruction in instructions:
        instruc_tokens.append(clip.tokenize(instruction,truncate=True)[0])

    caption = []
    ingredients=",".join(ingredients)
    instructions = ",".join(instructions)
    #caption = title
    #title_tokens = clip.tokenize(title,truncate=True)
    #ingredient_tokens=clip.tokenize(ingredients,truncate=True)
    #tokens = torch.concat((title_tokens, ingredient_tokens))
    caption = title +";" +ingredients + ";" + instructions
    tokens=clip.tokenize(caption,truncate=True)
    #if len(instruc_tokens)>40 or len(ingre_tokens)>25:
    #    print('len_instruc:',len(instruc_tokens),'len_ingre:',len(ingre_tokens))
    tmp=torch.as_tensor(torch.zeros([77]),dtype=torch.long)
    
    if len(ingre_tokens)>20:
        ingre_tokens=ingre_tokens[:20]
    else:
        for i in range(20-len(ingre_tokens)):
            ingre_tokens.append(tmp)
        
    if len(instruc_tokens)>20:
        instruc_tokens=instruc_tokens[:20]
    else:
        for i in range(20-len(instruc_tokens)):
            instruc_tokens.append(tmp)
    #relation_embds = encode_relation(relation, tokenizer, text_encoder, accelerator )
    relation_embds = [ ]
    return caption, tokens[0], ingre_tokens, instruc_tokens


def encode_relation(relation, tokenizer, text_encoder,accelerator):
    relation_arr = re.findall(r'[(](.*?)[)]', relation)
    relation_list = list(map(lambda x: x.strip(),relation_arr))
    relation_ids = tokenizer(relation_list, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    relation_embeddings = text_encoder(relation_ids.to(accelerator.device)).last_hidden_state # N * 77 * 768
    
    padding = 20 - relation_embeddings.shape[0]
    if padding > 0:
        relation_embeddings=torch.cat([relation_embeddings,torch.zeros([padding,77,768])],dim=0)
    else:
        relation_embeddings=relation_embeddings[:20]
    return relation_embeddings

def prepare_dataset(args, split, transform, tokenizer, text_encoder,accelerator):
    imsize = args.imsize
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])
    dataset = TextImgDataset(split=split, transform=image_transform, args=args, tokenizer=tokenizer, text_encoder=text_encoder, accelerator=accelerator)
    return dataset

################################################################
#                    Dataset
################################################################
class TextImgDataset(data.Dataset):
    def __init__(self, split, transform=None, args=None, tokenizer=None, text_encoder=None, accelerator=None):
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.accelerator=accelerator
        #self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.files = self.get_files(self.data_dir,split)
        self.number_example = len(self.files)
        self.relations = self.get_relations(self.data_dir,split)
        self.textnames = self.load_filetextnames(self.data_dir, split)

    
    def get_files(self,data_dir,split):
        filepath='%s/%s.json'% (data_dir,split)
        if os.path.isfile(filepath):
            with open(filepath,'r',encoding='utf-8') as f:
                contents=json.load(f)
            print('Load contents from: %s (%d)' % (filepath, len(contents)))
        else:
            contents=[]
        return contents
    
    def load_filetextnames(self, data_dir, split):
        filepath = '%s/images/' % (data_dir)
        if os.path.isfile(filepath):
            textnames = os.listdir(filepath).split('.')[0]
            print('Load textnames from: %s (%d)' % (filepath, len(textnames)))
        else:
            textnames = []
        return textnames
    
    def get_relations(self,data_dir,split):
        filepath='%s/relation/%s.txt'% (data_dir,split)
        if os.path.isfile(filepath):
            with open(filepath,'r') as f:
                relations=f.read().splitlines()
            print('Load relations from: %s (%d)' % (filepath, len(relations)))
        else:
            relations=[]
        return relations


    def __getitem__(self, index):
        #
        key = self.files[index] 
        textnames = self.textnames[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        
        if self.dataset_name.lower().find('recipe') != -1:
            if self.split=='train':
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/train/%s' % (data_dir, imgs[0])

            
            elif self.split=='test':
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/test/%s' % (data_dir, imgs[0])
            else:
                title= key['title']
                ingredients = key['ingredients']
                instructions = key['instructions']
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/val/%s' % (data_dir, imgs[0])

        elif self.dataset_name.lower().find('fooddata') != -1:
            if self.split=='train':
                img_name = '%s/images/%s' % (data_dir, key)
                text_name = '%s/captions/%s.txt' % (data_dir, textnames)
            else:
                img_name = '%s/test_images/%s' % (data_dir, key)
                text_name = '%s/test_captions/%s.txt' % (data_dir, textnames)

        else:
            relation = None
            if self.split=='train':
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/%s' % (data_dir, imgs[0])

            elif self.split=='test':
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/%s' % (data_dir, imgs[0])

            else:
                title= key['title']
                ingredients = key['ingredients']
                instructions= []
                imgs=np.random.choice(key['images'], size=1)
                img_name = '%s/images/%s' % (data_dir, imgs[0])
        #
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        if self.dataset_name.lower().find('recipe') != -1:
            caps, tokens, ingre_tokens, instruc_tokens= get_recipe(title,ingredients, instructions)
            relation = self.relations[index]
        else:
            caps, tokens, ingre_tokens, instruc_tokens= get_vireo(title,ingredients)
            relation = None
        key=key['id']
        return imgs, caps,tokens, key, ingre_tokens, instruc_tokens, relation

    def __len__(self):
        return len(self.files)