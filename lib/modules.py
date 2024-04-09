import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
import re
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torchvision.models.inception import inception_v3
from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist
from scipy.stats import entropy

from lib.utils import mkdir_p, get_rank, transf_to_CLIP_input
from lib.datasets import prepare_data


def load_clip(device):
    import clip as clip
    #model = clip.load(clip_info['type'], device=device)[0]
    model,_=clip.load("ViT-B/32",device=device)
    return model


def test(dataloader, text_encoder, tokenizer, vae, unet, CLIP, scheduler, accelerator, m1, s1, epoch, times, args):
    FID, TI_sim,IS = calculate_metrics(dataloader, text_encoder, tokenizer, vae, unet, CLIP, scheduler, accelerator, m1, s1, epoch, times, args)
    return FID, TI_sim, IS

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def sample(dataloader, unet, tokenizer, text_encoder, vae, scheduler, accelerator, epoch, args):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        device = accelerator.device
        real, captions, _, _, _, _ = data
        ######################################################
        # (2) Generate fake images
        ######################################################
        height = args.imsize                   
        width = args.imsize                      
        num_inference_steps = args.num_inference_steps          
        guidance_scale = args.guidance_scale                
        generator = torch.manual_seed(0)    
        batch_size = args.batch_size

        with torch.no_grad():
            text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(accelerator.device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(num_inference_steps)
            for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            fake_imgs = [Image.fromarray(image) for image in images]
            nrow = 8
            ncol = batch_size // nrow 
            grid = image_grid(fake_imgs, rows = nrow, cols = ncol)
            #fake_imgs = torch.clamp(fake_imgs, -1., 1.)
            if args.multi_gpus==True:
                batch_img_name = 'epoch_%04d.png'%(epoch)
                batch_img_save_dir  = osp.join(args.img_save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'epoch_%04d.txt'%(epoch)
                batch_txt_save_dir  = osp.join(args.img_save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'epoch_%04d.png'%(epoch)
                batch_img_save_dir  = osp.join(args.img_save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'epoch_%04d.txt'%(epoch)
                batch_txt_save_dir  = osp.join(args.img_save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            grid.save(batch_img_save_name)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('epoch: %d' % (epoch))

def get_one_batch_data(dataloader):
    data = next(iter(dataloader))
    imgs, prompts, _, _, _, _, _ = data
    return imgs, prompts

def get_fixed_data(test_dl):
    fixed_test_img, fixed_test_txt = get_one_batch_data(test_dl)
    return fixed_test_img, fixed_test_txt

def encode_relation(relations, tokenizer, text_encoder,accelerator):
    ingredients=[]
    relations_=[]
    for relation in relations:
        relation_arr = re.findall(r'[(](.*?)[)]', relation)
        ingredient_list = list(map(lambda x: x.strip(),[relation.split(',')[0] for relation in relation_arr]))
        relation_list = list(map(lambda x: x.strip(),[relation.split(',')[-1] for relation in relation_arr]))
        padding = 20 - len(relation_arr)
        if padding>0:
            ingredient_list = ingredient_list+padding*['None']
            relation_list =relation_list +padding*['None']
        else:
            ingredient_list = ingredient_list[:20]
            relation_list =relation_list[:20] 
        ingredients += ingredient_list
        relations_ += relation_list
    ingredient_ids = tokenizer(ingredients, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    ingredient_embs = text_encoder(ingredient_ids.to(accelerator.device)).last_hidden_state.view(-1,20,77,768) # B * N * 77 * 768
    relation_ids = tokenizer(relations_, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    relation_embs = text_encoder(relation_ids.to(accelerator.device)).last_hidden_state.view(-1,20,77,768) # B * N * 77 * 768
    return ingredient_embs, relation_embs

def sample_one_batch(relation_embs, captions, unet, tokenizer, text_encoder, vae, scheduler, accelerator, epoch, args):
    height = args.imsize                    # default height of Stable Diffusion
    width = args.imsize                        # default width of Stable Diffusion
    num_inference_steps = args.num_inference_steps          # Number of denoising steps
    guidance_scale = args.guidance_scale                # Scale for classifier-free guidance
    generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise
    batch_size = args.batch_size
    text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(accelerator.device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(accelerator.device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, relation_embs = relation_embs).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    # scale and decode the image latents with vae
    if args.mixed_precision == 'fp16':
        latents = 1 / 0.18215 * latents.to(dtype=torch.float16)
    else:
        latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    fake_imgs = [Image.fromarray(image) for image in images]
    ncol = 4
    nrow = batch_size // ncol
    grid = image_grid(fake_imgs, rows = nrow, cols = ncol)
    #fake_imgs = torch.clamp(fake_imgs, -1., 1.)
    if args.multi_gpus==True:
        batch_img_name = 'epoch_%04d.png'%(epoch)
        batch_img_save_dir  = osp.join(args.img_save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
        batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
        batch_txt_name = 'epoch_%04d.txt'%(epoch)
        batch_txt_save_dir  = osp.join(args.img_save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
        batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
    else:
        batch_img_name = 'epoch_%04d.png'%(epoch)
        batch_img_save_dir  = osp.join(args.img_save_dir, 'batch', 'imgs')
        batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
        batch_txt_name = 'epoch_%04d.txt'%(epoch)
        batch_txt_save_dir  = osp.join(args.img_save_dir, 'batch', 'txts')
        batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
    mkdir_p(batch_img_save_dir)
    grid.save(batch_img_save_name)
    mkdir_p(batch_txt_save_dir)
    txt = open(batch_txt_save_name,'w')
    for cap in captions:
        txt.write(cap+'\n')
    txt.close()



def calculate_metrics(dataloader, text_encoder, tokenizer, vae, unet, CLIP, scheduler, accelerator, m1, s1, epoch, times, args):
    device = accelerator.device
    max_epoch = args.max_train_epochs
    batch_size = args.batch_size
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    incep_score=torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    unet.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = 1
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))

    for time in range(times):
        for i, data in enumerate(dataloader,0):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, _, _, _, _ = data
            ######################################################
            # (2) Generate fake images
            ######################################################
            height = args.imsize                  
            width = args.imsize                      
            num_inference_steps = args.num_inference_steps        
            guidance_scale = args.guidance_scale                
            generator = torch.manual_seed(0)  
            text_input = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(device)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(num_inference_steps)
            for t in scheduler.timesteps:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            latents = latents.to(dtype=torch.float16)
            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).float()
            fake_imgs = torch.clamp(image, 0., 1.)
            fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)

            #clipsim
            clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
            clip_cos = clip_cos + clip_sim
            #IS
            inception_score,_=calc_IS(fake_imgs,batch_size,device)
            incep_score=incep_score+inception_score
            fake = norm(fake_imgs)
            pred = model(fake)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            # concat pred from multi GPUs
            output = list(torch.empty_like(pred) for _ in range(n_gpu))
            #dist.all_gather(output, pred)
            pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
            pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    accelerator.wait_for_everyone()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    #dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    #IS
    IS_gather=list(torch.empty_like(incep_score) for _ in range(n_gpu))
    #dist.all_gather(IS_gather, incep_score)
    IS=torch.cat(IS_gather, dim=0).mean().item()/(dl_length*times)

    return fid_value,clip_score,IS

def calc_IS(imgs,batch_size,device,splits=1):

    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
        
    N=len(imgs)
    split_scores = []
    preds = np.zeros((N, 1000))
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    #batch = torch.from_numpy(imgs).type(torch.FloatTensor)
    #batch = batch.to(device)
    #preds[i:i + batch_size] = get_pred(batch)
    for i in range(0, N, batch_size):
        
        batch = imgs.type(torch.FloatTensor)
        batch = batch.to(device)
        #y = get_pred(batch)
        
        preds[i:i + batch_size] = get_pred(batch)

    #compute the mean kl-div 
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)



def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    caps_clip = caps_clip.to(device)
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim

def calc_inception_score(imgs, i,batch,cuda=True, batch_size=4, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    #assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    
    batch = batch.type(dtype)
    batchv = Variable(batch)
    batch_size_i = batch.size()[0]
    preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

class GRU(torch.nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
 
    def forward(self, input_seq, device):
        # input(batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.gru(input_seq, (h_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred

class RelationInjector(torch.nn.Module):
    def __init__(self, input_size = 20, hidden_size = 1, num_layers=1, device=None):
        super().__init__()
        #self.fc = torch.nn.Linear(77* input_size*2, 77* input_size)
        self.relu = torch.nn.ReLU()
        #self.gru = GRU(hidden_size, output_size, num_layers)
        self.num_directions = 1
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = 1

    def forward(self, input_ingredients, input_relations):
        batch_size, seq_len, token_len ,dim= input_ingredients.shape[0], input_ingredients.shape[1], input_ingredients.shape[2], input_ingredients.shape[3]
        h_0 = torch.randn(self.num_directions * self.num_layers, dim*token_len, self.hidden_size).to(self.device)
        concat_feature = torch.add(input_ingredients, input_relations)
        concat_feature = concat_feature.view(batch_size, seq_len, -1).transpose(1,2)
        #concat_feature = self.fc(concat_feature).to(self.device) #.view(batch_size, seq_len, -1, 768)
        concat_feature = self.relu(concat_feature)
        relation_feature = self.gru(concat_feature,(h_0))
        relation_feature = torch.tensor(relation_feature[0]).transpose(1,2).view(batch_size,token_len, -1)
        return relation_feature
