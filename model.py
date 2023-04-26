# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import numpy as np

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None, args=None):
        super(Seq2Seq, self).__init__()
        self.source_encoder = encoder
        self.target_encoder = copy.deepcopy(self.source_encoder)
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.source_encoder.embeddings.word_embeddings)
        
    def forward(self, state, source_ids, source_mask, target_ids = None, target_mask = None, args = None,
                weightAB = None, weightBB = None, weightcon = None):
        codeA_outputs = self.source_encoder(source_ids, attention_mask=source_mask)
        codeA_encoder_output = codeA_outputs[0].permute([1, 0, 2]).contiguous()

        if (state == 'train' or state == 'finetune'):
            target_code_ids = target_ids[:, :-args.max_comment_length]
            target_code_mask = target_mask[:, :-args.max_comment_length]
            decoder_attn_mask = -1e4 * (1 - self.bias[:target_code_ids.shape[1], :target_code_ids.shape[1]])
            # loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            mseloss = nn.MSELoss()

            loss_A_B = 0
            if (weightAB != 0):
                tgt_embeddings = self.source_encoder.embeddings(target_code_ids).permute([1, 0, 2]).contiguous()
                codeA_decoder_output = self.decoder(tgt_embeddings, codeA_encoder_output, tgt_mask=decoder_attn_mask,
                                                    memory_key_padding_mask=(1 - source_mask).bool())
                codeA_hidden_states = torch.tanh(self.dense(codeA_decoder_output)).permute([1, 0, 2]).contiguous()
                codeA_lm_logits = self.lm_head(codeA_hidden_states)
                # Shift so that tokens < n predict n
                active_loss = target_code_mask[..., 1:].ne(0).view(-1) == 1
                shift_logits = codeA_lm_logits[..., :-1, :].contiguous()
                shift_labels = target_code_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_A_B = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                    shift_labels.view(-1)[active_loss])

            # B -> encoderB -> h(B) -> decoderB -> B(B)
            loss_B_B = 0
            if (weightBB != 0):
                codeB_outputs = self.target_encoder(target_ids, attention_mask=target_mask)
                codeB_encoder_output = codeB_outputs[0].permute([1, 0, 2]).contiguous()
                tgt_embeddings = self.target_encoder.embeddings(target_code_ids).permute([1, 0, 2]).contiguous()
                codeB_decoder_output = self.decoder(tgt_embeddings, codeB_encoder_output, tgt_mask=decoder_attn_mask,
                                                    memory_key_padding_mask=(1 - target_mask).bool())
                hidden_states = torch.tanh(self.dense(codeB_decoder_output)).permute([1, 0, 2]).contiguous()

                lm_logits = self.lm_head(hidden_states)
                # Shift so that tokens < n predict n
                active_loss = target_code_mask[..., 1:].ne(0).view(-1) == 1
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = target_code_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss_B_B = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                    shift_labels.view(-1)[active_loss])

            loss_mse = 0
            if (weightcon != 0):
                input_embed = codeA_decoder_output.permute([1, 0, 2])
                comment_embed = self.target_encoder.embeddings(target_ids).contiguous()[:,-args.max_comment_length:, :]
                input_embed = torch.cat([input_embed, comment_embed], dim=1)
                #print(input_embed.shape, target_mask.shape)
                codeABB_outputs = self.target_encoder(inputs_embeds=input_embed,
                                                      attention_mask=target_mask)
                codeABB_encoder_output = codeABB_outputs[0].contiguous()
                codeB_features = codeB_encoder_output.permute([1, 0, 2]).cuda()
                codeABB_features = codeABB_encoder_output.cuda()
                loss_mse = mseloss(codeB_features, codeABB_features)


            loss = weightAB * loss_A_B + weightBB * loss_B_B + weightcon * loss_mse
            outputs = loss
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = codeA_encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    decoder_attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.source_encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    # print(tgt_embeddings.shape, context.shape,decoder_attn_mask.shape)
                    out = self.decoder(tgt_embeddings, context, tgt_mask=decoder_attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)

                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds




class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        #prevK = bestScoresId // numWords
        prevK = (bestScoresId / numWords).long()
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
