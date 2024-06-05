import torch
from torch.autograd import Variable
from torch.nn import LSTM, GRU, Linear, LSTMCell, Module
import torch.nn.functional as F
import pdb
import cPickle as pickle
import os, random,pdb, torch
from itertools import groupby
from tqdm import tqdm
import numpy as np
import argparse
import models
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm
import dataloader
from visdom import Visdom

class dataloader():
    def __init__(self, batchSize, epochs, vocab, train_path, test_path, max_article_size=400, max_abstract_size=140, test_mode=False):
        self.maxEpochs = epochs
        self.epoch = 1
        self.batchSize = batchSize
        self.iterInd = 0
        self.globalInd = 1
        
        self.word2id, self.id2word = self.getVocabMap(vocab)
        self.vocabSize = len(vocab)
        self.max_article_size = max_article_size
        self.max_abstract_size = max_abstract_size
        self.test_mode = test_mode

        assert os.path.isfile(train_path) and os.path.isfile(test_path), 'Invalid paths to train/test datafiles'
        self.train_path = train_path
        self.test_path = test_path
        if not self.test_mode:
            print 'Loading training data from disk...will take a minute...'
            with open(self.train_path,'rb') as f:
                self.train_data = pickle.load(f)        
            self.trainSamples = len(self.train_data)
        else:
            print 'Initializing Dataloader in test mode with only eval-dataset...'
        
        print 'Loading eval data from disk...'
        with open(self.test_path,'rb') as f:
            self.test_data = pickle.load(f)        
        self.testSamples = len(self.test_data)

        
        if not self.test_mode:
            self.stopFlag = False
            self.pbar = tqdm(total=self.trainSamples * self.maxEpochs)
            self.pbar.set_description('Epoch : %d/%d' % (self.epoch, self.maxEpochs))

    def getVocabMap(self, vocab):
        word2id, id2word = {}, {}
        for i, word in enumerate(vocab):        
            word2id[word] = i+1
            id2word[i+1] = word
        id2word[0] = ''
        
        return word2id, id2word        

    def makeEncoderInput(self, article):        
        self.encUnkCount = 1
        _intArticle, extIntArticle = [], []
        article_oov = []
        art_len = min(self.max_article_size, len(article))
        for word_ind, word in enumerate(article[:art_len]):
            try:
                _intArticle.append(self.word2id[word.lower().strip()])        
                extIntArticle.append(self.word2id[word.lower().strip()])        
            except KeyError:
                _intArticle.append(self.word2id['<unk>'])        
                extIntArticle.append(self.vocabSize + self.encUnkCount)        
                article_oov.append(word)
        
                self.encUnkCount += 1            
        
        return _intArticle, extIntArticle, article_oov, art_len

    def makeDecoderInput(self, abstract, article_oov):
        _intAbstract, extIntAbstract = [], []
        abs_len = min(self.max_abstract_size, len(abstract)) 
        
        self.decUnkCount = 0
        for word in abstract[:abs_len]:
            try:
                _intAbstract.append(self.word2id[word.lower().strip()])
                extIntAbstract.append(self.word2id[word.lower().strip()])        
            except KeyError        
                _intAbstract.append(self.word2id['<unk>'])        
                #check if OOV word present in article
                if word in article_oov:
                    extIntAbstract.append(self.vocabSize + article_oov.index(word) + 1)        
                else:
                    extIntAbstract.append(self.word2id['<unk>'])
                self.decUnkCount += 1
        return _intAbstract, extIntAbstract, abs_len

    def preproc(self, samples):           
        extIntArticles, intRevArticles, intAbstract, intTargets, extIntAbstracts = [], [], [], [], []
        art_lens, abs_lens= [], []
        maxLen = 0
        max_article_oov = 0
        for sampl in samples:        
            article = sampl['article'].split(' ')
            abstract = sampl['abstract'].split(' ')
            # get article and abstract int-tokenized
            _intArticle, _extIntArticle, article_oov, art_len = self.makeEncoderInput(article)
            if max_article_oov < len(article_oov):
                max_article_oov = len(article_oov)
            _intRevArticle = list(reversed(_intArticle))
            _intAbstract, _extIntAbstract, abs_len = self.makeDecoderInput(abstract, article_oov)

        
            intAbstract.append([self.word2id['<go>']] + _intAbstract)        
        
            intTargets.append(_extIntAbstract + [self.word2id['<end>']])        
            abs_len += 1        
            extIntArticles.append(_extIntArticle)        
            intRevArticles.append(_intRevArticle        
            art_lens.append(art_len)
            abs_lens.append(abs_len)        
        
        padExtArticles = [torch.LongTensor(item + [0] * (max(art_lens) - len(item))) for item in extIntArticles]                
        padRevArticles = [torch.LongTensor(item + [0] * (max(art_lens) - len(item))) for item in intRevArticles]                
        padAbstracts = [torch.LongTensor(item + [0] * (max(abs_lens) - len(item))) for item in intAbstract]
        padTargets = [torch.LongTensor(item + [0] * (max(abs_lens) - len(item))) for item in intTargets]  

        batchExtArticles = torch.stack(padExtArticles, 0)
        
        batchArticles = batchExtArticles.clone().masked_fill_((batchExtArticles > self.vocabSize), self.word2id['<unk>'])
        batchRevArticles = torch.stack(padRevArticles, 0)
        batchAbstracts = torch.stack(padAbstracts, 0)
        batchTargets = torch.stack        (padTargets, 0)        
        art_lens = torch.LongTensor(art_lens)
        abs_lens = torch.LongTensor(abs_lens)        
        return batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets, art_lens, abs_lens, max_article_oov, article_oov        

                def getBatch(self, num_samples=None):
        if num_samples is None:
            num_samples = self.batchSize

        if self.epoch > self.maxEpochs:
            print 'Maximum Epoch Limit reached'
            self.stopFlag = True
            return None
        
        if self.iterInd + num_samples > self.trainSamples:
            data = [self.train_data[i] for i in xrange(self.iterInd, self.trainSamples)]
        else:
            data = [self.train_data[i] for i in xrange(self.iterInd, self.iterInd + num_samples)]
        
        batchData = self.preproc(data)
        
        self.globalInd += 1
        self.iterInd += num_samples
        if self.iterInd > self.trainSamples:
            self.iterInd = 0            
            self.epoch += 1
            self.globalInd = 1
            self.pbar.set_description('Epoch : %d/%d' % (self.epoch, self.maxEpochs))

        return batchData
        
    def getEvalBatch(self, num_samples=1):
        
        data = [self.test_data[i] for i in range(num_samples)]
        batchData = self.evalPreproc(data[0])
        return batchData   

                def evalPreproc(self, sample):	 
        
        extIntArticles, intRevArticles = [], []
        max_article_oov = 0        
        article = sample['article'].split(' ')					
        
        _intArticle, _extIntArticle, article_oov, _ = self.makeEncoderInput(article)
        if max_article_oov < len(article_oov):
            max_article_oov = len(article_oov)
        _intRevArticle = list(reversed(_intArticle))
        
        extIntArticles.append(_extIntArticle)        
        intRevArticles.append(_intRevArticle)
        
        padExtArticles = [torch.LongTensor(item) for item in extIntArticles]        
        padRevArticles = [torch.LongTensor(item) for item in intRevArticles]        

        batchExtArticles = torch.stack(padExtArticles, 0)
        
        batchArticles = batchExtArticles.clone().masked_fill_((batchExtArticles > self.vocabSize), self.word2id['<unk>'])
        batchRevArticles = torch.stack(padRevArticles, 0)
        
        return batchArticles, batchRevArticles, batchExtArticles, max_article_oov, article_oov, sample['article'], sample['abstract']

    def getEvalSample(self, index=None):
        if index is None:
            rand_index = np.random.randint(0, self.testSamples-1)
            data = self.test_data[rand_index]
            return self.evalPreproc(data)

        elif isinstance(index, int) and (index>=0 and index < self.testSamples):
            data =	self.test_data[index]
            return self.evalPreproc(data)

    def getInputTextSample(self, tokenized_text):
        extIntArticles, intRevArticles = [], []
        max_article_oov = 0        
        
        _intArticle, _extIntArticle, article_oov, _ = self.makeEncoderInput(tokenized_text)
        if max_article_oov < len(article_oov):
            max_article_oov = len(article_oov)
        _intRevArticle = list(reversed(_intArticle))
        
        extIntArticles.append(_extIntArticle)        
        intRevArticles.append(_intRevArticle)
        
        padExtArticles = [torch.LongTensor(item) for item in extIntArticles]        
        padRevArticles = [torch.LongTensor(item) for item in intRevArticles]        

        batchExtArticles = torch.stack(padExtArticles, 0)
        
        batchArticles = batchExtArticles.clone().masked_fill_((batchExtArticles > self.vocabSize), self.word2id['<unk>'])
        batchRevArticles = torch.stack(padRevArticles, 0)
        
        return batchArticles, batchRevArticles, batchExtArticles, max_article_oov, article_oov


class Hypothesis(object):
    def __init__(self, token_id, hidden_state, cell_state, log_prob):
        self._h = hidden_state
        self._c = cell_state
        self.log_prob = log_prob
        self.full_prediction = token_id # list
        self.survivability = self.log_prob/ float(len(self.full_prediction))

    def extend(self, token_id, hidden_state, cell_state, log_prob):
        return Hypothesis(token_id= self.full_prediction + [token_id],
                           hidden_state=hidden_state,
                           cell_state=cell_state,
                           log_prob= self.log_prob + log_prob)

class Encoder(Module):	
    def __init__(self, input_size, hidden_size, wordEmbed):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.word_embed = wordEmbed
        self.fwd_rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.bkwd_rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.output_cproj = Linear(self.hidden_size * 2, self.hidden_size)
        self.output_hproj = Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, _input, rev_input):
        batch_size, max_len = _input.size(0), _input.size(1)            
        embed_fwd = self.word_embed(_input)
        embed_rev = self.word_embed(rev_input)
        
        mask = _input.eq(0).detach()

        fwd_out, fwd_state = self.fwd_rnn(embed_fwd)
        bkwd_out, bkwd_state = self.bkwd_rnn(embed_rev)
        hidden_cat = torch.cat((fwd_out, bkwd_out), 2)

        # inverse of mask
        inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, self.hidden_size * 2).float().detach()
        hidden_out = hidden_cat * inv_mask
        final_hidd_proj = self.output_hproj(torch.cat((fwd_state[0].squeeze(0), bkwd_state[0].squeeze(0)), 1))
        final_cell_proj = self.output_cproj(torch.cat((fwd_state[1].squeeze(0), bkwd_state[1].squeeze(0)), 1))

        return hidden_out, final_hidd_proj, final_cell_proj, mask

        
class PointerAttentionDecoder(Module):
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed):
        super(PointerAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embed = wordEmbed

        
        self.decoderRNN = LSTM(self.input_size, self.hidden_size, batch_first=True)
        
        self.Wh = Linear(2 * self.hidden_size, 2*self. hidden_size)
        self.Ws = Linear(self.hidden_size, 2*self.hidden_size)
        self.w_c = Linear(1, 2*self.hidden_size)
        self.v = Linear(2*self.hidden_size, 1)

        
        self.w_h = Linear(2 * self.hidden_size, 1)
        self.w_s = Linear(self.hidden_size, 1)
        self.w_x = Linear(self.input_size, 1)


        self.V = Linear(self.hidden_size * 3, self.vocab_size)
        self.min_length = 40
        
    def setValues(self, start_id, stop_id, unk_id, beam_size, max_decode=40, lmbda=1):
        self.start_id = start_id
        self.stop_id = stop_id
        self.unk_id = unk_id
        self.max_decode_steps = max_decode        
        self.max_article_oov = None
        self.beam_size = beam_size
        self.lmbda = lmbda

    def forward(self, enc_states, enc_final_state, enc_mask, _input, article_inds, targets, decode=False):           

        if decode is True:
            return self.decode(enc_states, enc_final_state, enc_mask, article_inds)

        
        batch_size, max_enc_len, enc_size = enc_states.size()
        max_dec_len = _input.size(1)
        
        coverage =  Variable(torch.zeros(batch_size, max_enc_len).cuda())            
        dec_lens = (_input > 0).float().sum(1)
        state = enc_final_state[0].unsqueeze(0),enc_final_state[1].unsqueeze(0)

        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)            
        embed_input = self.word_embed(_input)

        lm_loss, cov_loss = [], []
        hidden, _ = self.decoderRNN(embed_input, state)
        
        for _step in range(max_dec_len):
            _h = hidden[:, _step, :        ]
            target = targets[:, _step].unsqueeze(1)            
            
            dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
            cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)            
            e_t = self.v(F.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

            attn_scores = e_t.view(batch_size, max_enc_len)
            del e_t
            attn_scores.masked_fill_(enc_mask, -float('inf'))
            attn_scores = F.softmax(attn_scores)
            
            context = attn_scores.unsqueeze(1).bmm(enc_states).squeeze(1)            
            p_vocab = 	F.softmax(self.V(torch.cat((_h, context), 1)))
            p_gen = F.sigmoid(self.w_h(context) + self.w_s(_h) + self.w_x(embed_input[:, _step, :]))
            p_gen = p_gen.view(-1, 1)
            weighted_Pvocab = p_gen * p_vocab
            weighted_attn = (1-p_gen)* attn_scores
            
            if self.max_article_oov > 0:
                ext_vocab = Variable(torch.zeros(batch_size, self.max_article_oov).cuda())
                combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)            
                del ext_vocab
            else:
                combined_vocab = weighted_Pvocab

            del weighted_Pvocab
            assert article_inds.data.min() >=0 and article_inds.data.max() <= (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'

   
            article_inds_masked = article_inds.add(-1).masked_fill_(enc_mask, 0)
            combined_vocab = combined_vocab.scatter_add(1, article_inds_masked, weighted_attn)            

            target_mask_0 = target.ne(0).detach()            
            target_mask_p = target.eq(0).detach()            
            target = target - 1
            output = combined_vocab.gather(1, target.masked_fill_(target_mask_p, 0))            
            lm_loss.append(output.log().mul(-1) * target_mask_0.float())
            
            coverage = coverage + attn_scores

        
            _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
            cov_loss.append(_cov_loss.sum(1))            

        
        total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda*torch.stack(cov_loss, 1).sum(1).div(dec_lens)            
        return total_masked_loss

    def decode_step(self, enc_states, state, _input, enc_mask, article_inds):
        
        batch_size, max_enc_len, enc_size = enc_states.size()

        coverage =  Variable(torch.zeros(batch_size, max_enc_len).cuda())            
        
        enc_proj = self.Wh(enc_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, -1)            
        embed_input = self.word_embed(_input)

        _h, _c = self.decoderRNN(embed_input, state)[1]            
        _h = _h.squeeze(0)
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)            
        e_t = self.v(F.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.masked_fill_(enc_mask, -float('inf'))
        attn_scores = F.softmax(attn_scores)

        context = attn_scores.unsqueeze(1).bmm(enc_states)            
        p_vocab = 	F.softmax(self.V(torch.cat((_h, context.squeeze(1)), 1)))
        p_gen = F.sigmoid(self.w_h(context.squeeze(1)) + self.w_s(_h) + self.w_x(embed_input[:, 0, :]))
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_gen * p_vocab
        weighted_attn = (1-p_gen)* attn_scores
        
        if self.max_article_oov > 0:            
            ext_vocab = Variable(torch.zeros(batch_size, self.max_article_oov).cuda())
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)            
            del ext_vocab
                else:
            combined_vocab = weighted_Pvocab            
                assert article_inds.data.min() >=0 and article_inds.data.max() <= (self.vocab_size+ self.max_article_oov), 'Recheck OOV indexes!'    
                combined_vocab = combined_vocab.scatter_add(1, article_inds.add(-1), weighted_attn)            
                
                return combined_vocab, _h, _c.squeeze(0)


    def getOverallTopk(self, vocab_probs, _h, _c, all_hyps, results):
        probs, inds = vocab_probs.topk(k=self.beam_size, dim=1)            
        probs = probs.log().data
        inds = inds.data
        inds.add_(1)
        candidates = []
        assert len(all_hyps) == probs.size(0), '# Hypothesis and log-prob size dont match'                
        for i, hypo in enumerate(probs.tolist()):
            for j, _ in enumerate(hypo):
                new_cand = all_hyps[i].extend(token_id=inds[i,j],
                                              hidden_state=_h[i].unsqueeze(0),
                                              cell_state=_c[i].unsqueeze(0),
                                              log_prob= probs[i,j])
                candidates.append(new_cand)
                candidates = sorted(candidates, key=lambda x:x.survivability, reverse=True)
                new_beam, next_inp = [], []
                next_h, next_c = [], []
                for h in candidates:
            if h.full_prediction[-1] == self.stop_id:
                if len(h.full_prediction)>=self.min_length:
                    results.append(h.full_prediction)
            else:
                new_beam.append(h)
                next_inp.append(h.full_prediction[-1])
                next_h.append(h._h.data)
                next_c.append(h._c.data)
            if len(new_beam) >= self.beam_size:
                break
        assert len(new_beam) >= 1, 'Non-existent beam'
        return new_beam, torch.LongTensor([next_inp]), results, torch.cat(next_h, 0), torch.cat(next_c, 0)

        
    def decode(self, enc_states, enc_final_state, enc_mask, article_inds):
        _input = Variable(torch.LongTensor([[self.start_id]]).cuda(), volatile=True)
        init_state = enc_final_state[0].unsqueeze(0),enc_final_state[1].unsqueeze(0)
        decoded_outputs = []
        
        all_hyps = [Hypothesis([self.start_id], None, None, 0)]            
        
        for _step in range(self.max_decode_steps):            
        
            curr_beam_size = _input.size(0)            
            beam_enc_states = enc_states.expand(curr_beam_size, enc_states.size(1), enc_states.size(2)).contiguous().detach()
            beam_article_inds = article_inds.expand(curr_beam_size, article_inds.size(1)).detach()            

            vocab_probs, next_h, next_c = self.decode_step(beam_enc_states, init_state, _input, enc_mask, beam_article_inds)

        
            all_hyps, decode_inds, decoded_outputs, init_h, init_c = self.getOverallTopk(vocab_probs, next_h, next_c, all_hyps, decoded_outputs)            

            decode_inds.masked_fill_((decode_inds > self.vocab_size), self.unk_id)
            decode_inds = decode_inds.t()
            _input = Variable(decode_inds.cuda(), volatile=True)
            init_state = (Variable(init_h.unsqueeze(0), volatile=True), Variable(init_c.unsqueeze(0), volatile=True))

        
        
        non_terminal_output = [item.full_prediction for item in all_hyps]
        all_outputs = decoded_outputs + non_terminal_output
        return all_outputs

    
class SummaryNet(Module):
    def __init__(self, input_size, hidden_size, vocab_size, wordEmbed, start_id, stop_id, unk_id, beam_size=4, max_decode=40, lmbda=1):
        super(SummaryNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = Encoder(self.input_size, self.hidden_size, wordEmbed)
        self.pointerDecoder = PointerAttentionDecoder(self.input_size, self.hidden_size, vocab_size, wordEmbed)
        self.pointerDecoder.setValues(start_id, stop_id, unk_id, beam_size, max_decode, lmbda)

    def forward(self, _input, max_article_oov, decode_flag=False):
        self.pointerDecoder.max_article_oov = max_article_oov            
        if decode_flag:
            enc_input, rev_enc_input, article_inds = _input
            enc_states, enc_hn, enc_cn, enc_mask = self.encoder(enc_input, rev_enc_input)
            model_summary = self.pointerDecoder(enc_states, (enc_hn, enc_cn), enc_mask, None, article_inds, targets=None, decode=True)
            return model_summary
            
        else:

            enc_input, article_inds, rev_enc_input, dec_input, dec_target = _input
            enc_states, enc_hn, enc_cn, enc_mask = self.encoder(enc_input, rev_enc_input)

            total_loss = self.pointerDecoder(enc_states, (enc_hn, enc_cn), enc_mask, dec_input, article_inds, targets=dec_target)
            return total_loss
                                  

parser = argparse.ArgumentParser()

parser.add_argument("--train-file", dest="train_file", help="Path to train datafile", default='finished_files/train.bin', type=str)
parser.add_argument("--test-file", dest="test_file", help="Path to test/eval datafile", default='finished_files/test.bin', type=str)
parser.add_argument("--vocab-file", dest="vocab_file", help="Path to vocabulary datafile", default='finished_files/vocabulary.bin', type=str)

parser.add_argument("--max-abstract-size", dest="max_abstract_size", help="Maximum size of abstract for decoder input", default=110, type=int)
parser.add_argument("--max-article-size", dest="max_article_size", help="Maximum size of article for encoder input", default=300, type=int)
parser.add_argument("--num-epochs", dest="epochs", help="Number of epochs", default=10, type=int)
parser.add_argument("--batch-size", dest="batchSize", help="Mini-batch size", default=32, type=int)
parser.add_argument("--embed-size", dest="embedSize", help="Size of word embedding", default=300, type=int)
parser.add_argument("--hidden-size", dest="hiddenSize", help="Size of hidden to model", default=128, type=int)

parser.add_argument("--learning-rate", dest="lr", help="Learning Rate", default=0.001, type=float)
parser.add_argument("--lambda", dest="lmbda", help="Hyperparameter for auxillary cost", default=1, type=float)
parser.add_argument("--beam-size", dest="beam_size", help="beam size for beam search decoding", default=4, type=int)
parser.add_argument("--max-decode", dest="max_decode", help="Maximum length of decoded output", default=40, type=int)
parser.add_argument("--grad-clip", dest="grad_clip", help="Clip gradients of RNN model", default=2, type=float)
parser.add_argument("--truncate-vocab", dest="trunc_vocab", help="size of truncated Vocabulary <= 50000 [to save memory]", default=50000, type=int)
parser.add_argument("--bootstrap", dest="bootstrap", help="Bootstrap word embeds with GloVe?", default=0, type=int)
parser.add_argument("--print-ground-truth", dest="print_ground_truth", help="Print the article and abstract", default=1, type=int)

parser.add_argument("--eval-freq", dest="eval_freq", help="How frequently (every mini-batch) to evaluate model", default=20000, type=int)
parser.add_argument("--save-dir", dest="save_dir", help="Directory to save trained models", default='Saved-Models/', type=str)
parser.add_argument("--load-model", dest="load_model", help="Directory from which to load trained models", default=None, type=str)

opt = parser.parse_args()
vis = Visdom()

assert opt.load_model is not None and os.path.isfile(opt.vocab_file), 'Invalid Path to trained model file'



assert opt.trunc_vocab <= 50000, 'Invalid value for --truncate-vocab'
assert os.path.isfile(opt.vocab_file), 'Invalid Path to vocabulary file'
with open(opt.vocab_file) as f:
    vocab = pickle.load(f)                                                          
    vocab = [item[0] for item in vocab[:-(5+ 50000 - opt.trunc_vocab)]]             
vocab += ['<unk>', '<go>', '<end>', '<s>', '</s>']                                 

dl = dataloader.dataloader(opt.batchSize, None, vocab, opt.train_file, opt.test_file, 
                          opt.max_article_size, opt.max_abstract_size, test_mode=True)


wordEmbed = torch.nn.Embedding(len(vocab) + 1, opt.embedSize, 0)
print 'Building SummaryNet...'
net = models.SummaryNet(opt.embedSize, opt.hiddenSize, dl.vocabSize, wordEmbed,
                       start_id=dl.word2id['<go>'], stop_id=dl.word2id['<end>'], unk_id=dl.word2id['<unk>'],
                       max_decode=opt.max_decode, beam_size=opt.beam_size, lmbda=opt.lmbda)
net = net.cuda()

print 'Loading weights from file...might take a minute...'
saved_file = torch.load(opt.load_model)
net.load_state_dict(saved_file['model'])
print '\n','*'*30, 'LOADED WEIGHTS FROM MODEL FILE : %s' %opt.load_model,'*'*30
    
net.eval()
print '\n\n'

for _ in range(5):
    if opt.article_path is not None and os.path.isfile(opt.article_path):
        with open(opt.article_path,'r') as f:
            article_string = f.read().strip()
            article_tokenized = word_tokenize(article_string)
        _article, _revArticle,  _extArticle, max_article_oov, article_oov = dl.getInputTextSample(article_tokenized)
        abs_string = '**No abstract available**'
    else:
    # pull random test sample
        data_batch = dl.getEvalSample()
        _article, _revArticle,  _extArticle, max_article_oov, article_oov, article_string, abs_string = dl.getEvalSample()

    _article = Variable(_article.cuda(), volatile=True)
    _extArticle = Variable(_extArticle.cuda(), volatile=True)
    _revArticle = Variable(_revArticle.cuda(), volatile=True)    
    all_summaries = net((_article, _revArticle, _extArticle), max_article_oov, decode_flag=True)

    displayOutput(all_summaries, article_string, abs_string, article_oov, show_ground_truth=opt.print_ground_truth)
                                  
                                  
def evalModel(model):
    model.eval()
    print '\n\n'
    print '*'*30, ' MODEL EVALUATION ', '*'*30

    _article, _revArticle,  _extArticle, max_article_oov, article_oov, article_string, abs_string = dl.getEvalBatch()        
    _article = Variable(_article.cuda(), volatile=True)
    _extArticle = Variable(_extArticle.cuda(), volatile=True)
    _revArticle = Variable(_revArticle.cuda(), volatile=True)    
    all_summaries = model((_article, _revArticle, _extArticle), max_article_oov, decode_flag=True)
    model.train()
    return all_summaries, article_string, abs_string, article_oov

def displayOutput(all_summaries, article, abstract, article_oov, show_ground_truth=False):    
    print '*' * 80
    print '\n'
    if show_ground_truth:
        print 'ARTICLE TEXT : \n', article
        print 'ACTUAL ABSTRACT : \n', abstract
    for i, summary in enumerate(all_summaries):    
        generated_summary = ' '.join([dl.id2word[ind] if ind<=dl.vocabSize else article_oov[ind % dl.vocabSize] for ind in summary])
        print 'GENERATED ABSTRACT #%d : \n' %(i+1), generated_summary    
    print '*' * 80
    return

def save_model(net, optimizer,all_summaries, article_string, abs_string):
    save_dict = dict({'model': net.state_dict(), 'optim': optimizer.state_dict(), 'epoch': dl.epoch, 'iter':dl.iterInd, 'summaries':all_summaries, 'article':article_string, 'abstract_gold':abs_string})
    print '\n','-' * 60
    print 'Saving Model to : ', opt.save_dir
    save_name = opt.save_dir + 'savedModel_E%d_%d.pth' % (dl.epoch, dl.iterInd)
    torch.save(save_dict, save_name)
    print '-' * 60  
    return

            

dl = dataloader.dataloader(opt.batchSize, opt.epochs, vocab, opt.train_file, opt.test_file, 
                          opt.max_article_size, opt.max_abstract_size)


if opt.bootstrap: 
    wordEmbed = torch.nn.Embedding(len(vocab) + 1, 300, 0)
    print 'Bootstrapping with pretrained GloVe word vectors...'
    assert os.path.isfile('embeds.pkl'), 'Cannot find pretrained Word embeddings to bootstrap'
    with open('embeds.pkl', 'rb') as f:
        embeds = pickle.load(f)
    assert wordEmbed.weight.size() == embeds.size()
    wordEmbed.weight.data[1:,:] = embeds
else:
    wordEmbed = torch.nn.Embedding(len(vocab) + 1, opt.embedSize, 0)

print 'Building and initializing SummaryNet...'
net = models.SummaryNet(opt.embedSize, opt.hiddenSize, dl.vocabSize, wordEmbed,
                       start_id=dl.word2id['<go>'], stop_id=dl.word2id['<end>'], unk_id=dl.word2id['<unk>'],
                       max_decode=opt.max_decode, beam_size=opt.beam_size, lmbda=opt.lmbda)
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

if opt.load_model is not None and os.path.isfile(opt.load_model):
    saved_file = torch.load(opt.load_model)
    net.load_state_dict(saved_file['model'])
    optimizer.load_state_dict(saved_file['optim'])
    dl.epoch = saved_file['epoch']
    dl.iterInd = saved_file['iter']
    dl.pbar.update(dl.iterInd)        
    print '\n','*'*30, 'RESUME FROM CHECKPOINT : %s' %opt.load_model,'*'*30
    
else:
    print '\n','*'*30, 'START TRAINING','*'*30
    
all_loss = []
win = None
while dl.epoch <= opt.epochs:
    data_batch = dl.getBatch(opt.batchSize)
    batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets, _, _, max_article_oov, article_oov  = data_batch
    if data_batch is None:
        print '-'*50, 'END OF TRAINING', '-'*50
        break    
    
    batchArticles = Variable(batchArticles.cuda())
    batchExtArticles = Variable(batchExtArticles.cuda())
    batchRevArticles = Variable(batchRevArticles.cuda())
    batchTargets = Variable(batchTargets.cuda())
    batchAbstracts = Variable(batchAbstracts.cuda())        

    losses = net((batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets), max_article_oov)
    batch_loss = losses.mean()
    
    batch_loss.backward()
    clip_grad_norm(net.parameters(), opt.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
       
    dl.pbar.set_postfix(loss=batch_loss.cpu().data[0])
    dl.pbar.update(opt.batchSize)        
    
    if dl.iterInd % 50:
        all_loss.append(batch_loss.cpu().data.tolist()[0])
        title = 'Pointer Model with Coverage'        
        if win is None:
            win = vis.line(Y=np.array(all_loss), X=np.arange(1, len(all_loss)+1), opts=dict(title=title, xlabel='#Mini-Batches (x%d)' %(opt.batchSize),
                           ylabel='Train-Loss'))
        vis.line(Y=np.array(all_loss), X=np.arange(1, len(all_loss)+1), win=win, update='replace', opts=dict(title=title, xlabel='#Mini-Batches (x%d)' %(opt.batchSize),
                           ylabel='Train-Loss'))
    
    if dl.iterInd % opt.eval_freq < opt.batchSize and dl.iterInd > opt.batchSize:       
        all_summaries, article_string, abs_string, article_oov = evalModel(net)
        displayOutput(all_summaries, article_string, abs_string, article_oov, show_ground_truth=opt.print_ground_truth)       
        if dl.iterInd % (6*opt.eval_freq) < opt.batchSize and dl.iterInd > opt.batchSize:       
            save_model(net, optimizer, all_summaries, article_string, abs_string)

    del batch_loss, batchArticles, batchExtArticles, batchRevArticles, batchAbstracts, batchTargets