import os, time, gc, json, pickle, argparse, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
from data.util import *
from util import *
from model import *

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

import matplotlib
import pdb

from myDataset import myDataset_mobile



def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, kl_loss


def train_step(device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    output = []
    optimizer.zero_grad()
    loss, ce_loss, kl_loss = compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                          target_tokens, mask, loss_fn, beta)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    optimizer.step()
    output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))
    return output


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'


        # torch.Size([1, 11, 768])
        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, attention_mask=x_mask[:, :-1], representations=z)
        prev = x_tokens[:, -1].view(batch_size, -1)

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length): #trange
            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def main():
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--experiment', type=int, default=0)

        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument("--seed", type=int, default=0)

        parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
        parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
        parser.add_argument('--iterations', type=int, default=80000)
        parser.add_argument('--warmup', type=int, default=10000,
                            help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

        parser.add_argument('--batch_sizes', nargs='+', type=int, default=[8],
                            help='batch size per GPU. Lists the schedule.')
        parser.add_argument('--seq-lens', nargs='+', type=int, default=[256],
                            help='seq length per sample. Lists the schedule.')

        parser.add_argument('--switch-time', type=float, default=0,
                            help="Percentage of iterations to spend on short sequence training.")
        parser.add_argument('--data-dir', type=str, default='data')
        parser.add_argument('--out-dir', type=str, default='out')
        parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
        parser.add_argument('--workers', default=1, type=int, metavar='N',
                            help='number of data loading workers')
        # use GPU
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--no_gpu', action="store_true")

        parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
        parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

        # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
        parser.add_argument('--beta_0', default=1.00, type=float)
        parser.add_argument('--beta_warmup', type=int, default=50000)
        # cyc_vae parameters
        parser.add_argument('--cycle', type=int, default=101640)

        parser.add_argument('--add_input', action="store_true")
        parser.add_argument('--add_attn', action="store_true")
        parser.add_argument('--add_softmax', action="store_true")
        parser.add_argument('--attn_proj_vary', action="store_true")
        parser.add_argument('--learn_prior', action="store_true")

        parser.add_argument('--split_data_dir', default="/data/zmy/dataset/mobile/split_label", type=str)
        parser.add_argument('--out_dir', default="./results/", type=str)
        parser.add_argument('--frequency_dir', default="/data/zmy/dataset/mobile/frequency.pkl", type=str)

        parser.add_argument('--max_pos', default=168, type=int, help="7*24, the max num of grids per day")
        parser.add_argument('--hidden_size', default=128, type=int)

        parser.add_argument('--control', default='age', type=str, help="")
        parser.add_argument('--dataset', default='mobile', type=str, help="")

        args = parser.parse_args()
    
    args.learn_prior = True
    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    # if gpu:
    #     print('Using GPU devices {}'.format(args.gpu))
    #     torch.cuda.set_device(args.gpu)
    #     print('Current single GPU: {}'.format(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # randomness
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, f"{args.dataset}_{args.control}_exp_{args.experiment}")
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    print('Loading models...')
    config = GPT2Config()
    config.output_past = True
    config.vocab_size = 32450
    config.n_positions = 256
    config.n_embd = args.hidden_size

    print(config)
    print("-"*100)

    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
    if args.learn_prior:
        init_para_frompretrained(VAE.encoder_prior, VAE.encoder, share_para=True)
        VAE.encoder_prior.averageSelfAttention.attention_weights = VAE.encoder.averageSelfAttention.attention_weights
    print('VAE_params:', num_params(VAE))  # 286694400

    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    print('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 40000
    tuning_all = False
    for name, parameter in VAE.named_parameters():
        # print((name, parameter.requires_grad))
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2', 'lm_head_rep']

        if not any([True if n in name else False for n in new_pars]):
           parameter.requires_grad = False

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)

    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)

    batch_size = args.batch_sizes[0] #########
    splits = ['train', 'val', 'test']
    dataset = {}
    dataloader = {}
    for i in splits:
        dataset[i] = myDataset_mobile(split=i,control=args.control)
        dataloader[i] = torch.utils.data.DataLoader(dataset=dataset[i], 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
    train_loader = dataloader['train']
    val_loader =  dataloader['val']
    test_loader =  dataloader['test']
    

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(device)
    VAE.train()

    optimizer = AdamW(VAE.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    print('Begin training iterations')
    max_val_batches = 20000  # max num. of val batches
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    # endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        VAE.eval()
        n_words_bpe = 0
        n_words = 0
        logp_sum = 0.0
        kl_loss_sum = 0.0

        with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
            for i, item in enumerate(train_loader):
                # x_mask = item['x_mask']
                # x_tokens = item['x_tokens']
                # y_mask = item['y_mask']
                # y_tokens = item['y_tokens']
                # input_tokens = item['input_tokens']
                # target_tokens = item['target_tokens']
                # mask = item['mask']

                # bs=1
                x_mask = item['x_mask'].unsqueeze(0)
                x_tokens = item['x_tokens'].unsqueeze(0)
                y_mask = item['y_mask'].unsqueeze(0)
                y_tokens = item['y_tokens'].unsqueeze(0)
                input_tokens = item['input_tokens'].unsqueeze(0)
                target_tokens = item['target_tokens'].unsqueeze(0)
                mask = item['mask'].unsqueeze(0)


                with torch.no_grad():
                    loss, ce_loss, kl_loss = compute_loss(device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                            input_tokens, target_tokens, mask, loss_fn, 1.0)


                if len(target_tokens.size()) == 1:
                    target_tokens = target_tokens.unsqueeze(0)
                n, l = target_tokens.size()

                text = target_tokens[0, :].tolist()
                logprob = ce_loss.tolist()
                assert len(text) == len(logprob)
                # only for story
                idx = text.index(32411)
                text = text[idx + 1:] # traj
                logprob = logprob[idx + 1:] # traj

                if 32411 in text:
                    idx = text.index(32411)
                    text = text[:idx]
                    logprob = logprob[:idx]

                logp_sum += sum(logprob)
                n_words_bpe += len(text)
                kl_loss_sum += kl_loss.item()
                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        kl = kl_loss_sum / len(val_loader)
        v_writer.add_scalar('loss', loss_bpe, num_iters)
        v_writer.add_scalar('kl', kl, num_iters)
        print('val loss    : %.4f' % loss_bpe)
        print('val   kl    : %.4f' % kl)
        VAE.train()

    def generate(test_loader, num_iters):
        VAE.eval()

        n_samples = 0
        bleu4_sum = 0.0
        rouge_scores_values_sum = [0.0] * 9

        args.nsamples = 1
        args.batch_size = 1
        args.temperature = 0.95
        args.top_k = 100
        args.top_p = 0.95
        model_type = args.model_type

        # write samples to file
        samples_file = open(os.path.join(save_folder, 'generate-' + '%07d' % num_iters + '.txt'), 'w', encoding='utf8')

        # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
        with tqdm(total=len(test_loader)) as pbar:
            for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(
                    test_loader):

                if i_test >= 10: break

                length = -1
                if length == -1:
                    length = VAE.config.n_ctx - x_tokens.size(1) - 1
                elif length > VAE.config.n_ctx - x_tokens.size(1) - 1:
                    raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

                eff_samples = []
                n, l = target_tokens.size()
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
                storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                              storys]

                for _ in range(args.nsamples // args.batch_size):
                    # model, batch_size, temperature, top_k, top_p, eos_token, sample = VAE, args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                    out, _ = sample_sequence(
                        model=VAE,
                        tokenizer=None,
                        length=length,
                        batch_size=args.batch_size,
                        x_mask=x_mask,
                        x_tokens=x_tokens,
                        y_mask=y_mask,
                        y_tokens=y_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                        eos_token=None,
                        model_type=model_type
                    )
                    out = out.tolist()

                    # extract story, check metrics
                    for i in range(len(out)):
                        text = out[i]
                        text = text[text.index(endoftext) + 1:]

                        if endoftext in text:
                            idx = text.index(endoftext)
                            text = text[:idx]
                        text = tokenizer.decode(text).strip()

                        try:
                            # check bleu
                            bleu4 = sentence_bleu([storys_str[i].split()], text,
                                                  smoothing_function=SmoothingFunction().method7)

                            # check rouge
                            rouge = Rouge()
                            rouge_scores = rouge.get_scores(text, storys_str[i])
                            rouge_scores_values = [v for k in rouge_scores[0].keys() for v in
                                                   rouge_scores[0][k].values()]

                            bleu4_sum += bleu4
                            rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                       zip(rouge_scores_values_sum, rouge_scores_values)]
                            n_samples += 1
                        except:
                            bleu4 = 0.0
                            rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                        eff_samples.append((text, bleu4, rouge_scores))

                    pbar.update(1)

                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                    samples_file.write('\n' * 2)
                    samples_file.write("=" * 40 + " Story " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(storys_str[i])
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Generated " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(eff_samples[i][0])
                    samples_file.write('\n' * 4)
                    samples_file.flush()

        print('Test complete with %05d samples.' % n_samples)

        bleu4 = round(bleu4_sum / n_samples, 3)
        rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
        print(' bleu-4:', bleu4)
        print(' rouge :', rouge_scores_values)
        VAE.train()


    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))
    while num_iters < args.iterations:
        st = time.time()
        # Training
        print('Training loop. Batches:', len(train_loader))

        with tqdm(total=len(train_loader)) as pbar:
            for i, item in enumerate(train_loader):
                x_mask = item['x_mask']
                x_tokens = item['x_tokens']
                y_mask = item['y_mask']
                y_tokens = item['y_tokens']
                input_tokens = item['input_tokens']
                target_tokens = item['target_tokens']
                mask = item['mask']

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        # print((name, parameter.requires_grad))
                        parameter.requires_grad = True
                    tuning_all = True

                output = train_step(device, VAE, optimizer, x_mask, x_tokens, y_mask, y_tokens,
                                    input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)
                loss, ce_loss, kl_loss = output[-1]

                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)
                t_writer.add_scalar('kl', kl_loss, num_iters)
                t_writer.add_scalar('beta', beta, num_iters)

                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()
                if end: break
                num_iters += 1
                pbar.update(1)
                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                # valid
                # if num_iters % 2500 == 0:
                #     val_step(val_loader)
                # save
                if num_iters % 5000 == 0:
                    print('Saving model...')
                    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))
        if not end:
            e += 1
            print("Training loop. The ith epoch completed: %d" % e)

    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    print('Training complete.')

if __name__ == "__main__":
    main()
