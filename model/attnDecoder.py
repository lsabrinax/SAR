import operator
import torch.nn as nn
import torch
import torch.nn.functional as F
from queue import PriorityQueue
from model.ResNet import ResNet

class Attn(nn.Module):
    def __init__(self, hidden_size, depth_size, feature_length, batch_size):
        super(Attn, self).__init__()
        self.depth_size = depth_size
        self.feature_length = feature_length
        #self.batch_size = batch_size
        self.linear = nn.Linear(hidden_size, depth_size)
        self.conv1 = nn.Conv2d(feature_length, depth_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth_size, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, feature_map, hidden_state):#hidden_state:seq_len * b * h, feature_map: b * D * h * w
        #feature_length = feature_map.size(1)#D
        seq_len = hidden_state.size(0)
        featureH = feature_map.size(2)
        featureW = feature_map.size(3)
        feature_size = feature_map.size(0)
        generate_feature = self.conv1(feature_map).unsqueeze(0)#b * depth_size * h * w
        generate_feature = generate_feature.repeat(seq_len, 1, 1, 1, 1)#seq_len * b * depth_size * h * w
        title_hidden_state = self.linear(hidden_state).unsqueeze(3).unsqueeze(3)# seq_len * b * d * 1 * 1
        #title_hidden_state.permute(1, 3, 0, 2)#b * 1 * 1 * d
        title_hidden_state = title_hidden_state.repeat(1, 1, 1, featureH, featureW)
        attention = F.tanh(generate_feature + title_hidden_state)#seq_len * b * depth_size * h * w
        attention = attention.view(seq_len * feature_size, self.depth_size, featureH, featureW)
        attention_weights = self.conv2(attention)#seq_len * b,1,  h, w
        attention_alpha = F.softmax(attention_weights.view(seq_len, feature_size, -1), dim=2)
        attention_weights = attention_alpha.view(seq_len, feature_size, 1, featureH, featureW) #seq_len * b *1 * h * w
        attention_weights = attention_weights.repeat(1, 1, self.feature_length, 1, 1)

        attention_weights = attention_weights * (feature_map.unsqueeze(0).repeat(seq_len, 1, 1,1, 1))#seq_len * b * D * h * w
        alpha = torch.sum(attention_weights, (3, 4)) #seq_len * b * D

        return alpha#seq_len * b * D


class attenDecoder(nn.Module):
    def __init__(self, output_size, batch_size, p, hidden_size=512, depth_size=512, emd_size=512, feature_length=512):
        super(attenDecoder, self).__init__()
        self.output_size = output_size
        self.embedding = nn.Linear(output_size, emd_size)
        self.lstm = nn.LSTM(emd_size, hidden_size, num_layers=2)

        self.attn = Attn(hidden_size, depth_size, feature_length, batch_size)
        self.dropout = nn.Dropout(p=p)
        self.linear = nn.Linear(feature_length + hidden_size, output_size)

    def forward(self,x, feature_map,  hidden_state):#hidden_state(hn, cn): 2(layers) * batch_size * hidden_size
        x = self.embedding(x).transpose(0, 1)#seq_len * b * emb_size
        # if encoder_input is not None:
        #     x = torch.cat((encoder_input, x), 0)

        output, hidden = self.lstm(x, hidden_state) # seq_len * b * hidden_size,没有初始化hidden有没有影响

        ct = self.attn(feature_map, output)
        out = torch.cat((output, ct), 2)
        out = self.dropout(out)
        out = self.linear(out) # seq_len * b *output_size

        return out, hidden

class SAR(nn.Module):
    """docstring for SAR"""
    def __init__(self, output_size, p, nchannel=1, nhidden=512, depth_size=512, emd_size=512, feature_length=512, batch_size=32):
        super(SAR, self).__init__()
        self.nchannel = nchannel
        self.nhidden = nhidden
        self.depth_size = depth_size
        self.emb_size = emd_size
        self.feature_length = feature_length
        self.batch_size = batch_size
        self.output_size = output_size
        self.encoder = ResNet(nchannel, nhidden)
        self.decoder = attenDecoder(hidden_size=nhidden, depth_size=depth_size, emd_size=emd_size, output_size=output_size, feature_length=feature_length, batch_size=batch_size, p=p)

    def forward(self, imgs, labels):
        hidden_state, feature_map = self.encoder(imgs)
        out, hidden = self.decoder(labels, feature_map, hidden_state)
        
        return (out[:-1,:, :].view(-1, self.output_size), hidden)

class BeamSearchNode(object):
    """docstring for BeamSearchNode"""
    def __init__(self, hidden_state, previousNode, decoder_input, logProb, length):
        super(BeamSearchNode, self).__init__()
        self.h = hidden_state
        self.prevNode = previousNode
        self.input = decoder_input
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.input <= other.input

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng -1 + 1e-6) + alpha * reward
        

def beam_decode(decoder, converter, decoder_hiddens, opt, feature_maps=None):
    '''
    target_tensor:[B, T]
    decoder_hiddens: [1, B, H]
    return: decoded_batch
   
    '''

    beam_width = 5#不清楚这个是用来干什么的
    topk = 1
    decoded_batch = []

    for idx in range(decoder_hiddens[0].size(1)):
        if isinstance(decoder_hiddens, tuple):#LSTM case
            #decoder_hidden = decoder_hiddens[:]
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(1).contiguous(),
                decoder_hiddens[1][:, idx, :].unsqueeze(1).contiguous())
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(1).contiguous() #2*1 *H
        feature_map = feature_maps[idx].unsqueeze(0)


        startid = converter.ch2ix['<START>']
        endid = converter.ch2ix['<END>']
        # decoder_input = torch.LongTensor(converter.cn).fill_(0)
        # decoder_input[startid] = 1

        # if opt.cuda:
        #     decoder_input.cuda()

        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        #starting node
        node = BeamSearchNode(decoder_hidden, None, startid, 0, 1)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            if qsize > 2000: break

            score, n = nodes.get()
            chid = n.input
            decoder_hidden = n.h

            if chid == endid and n.prevNode != None:
                endnodes.append((score, n))

                if len(endnodes) >= number_required:#这里应该不在判断条件里面吧
                    break
                else:
                    continue

            decoder_input = torch.FloatTensor(1, 1, converter.nc).fill_(0)
            decoder_input[0, 0, chid] = 1.0
            #print('n.input is', chid)
            #print('decoder_input is', decoder_input)
            if opt.cuda:
                #print('convert to GPU')
                decoder_input = decoder_input.cuda()
            #print(decoder_hidden[0].shape)
            decoder_output, decoder_hidden = decoder(
                decoder_input, feature_map, decoder_hidden)
            decoder_output = decoder_output.squeeze(0)
            decoder_output = F.log_softmax(decoder_output, dim=1)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].item()
                #print('beam_width %d, index %d' %( new_k, decoded_t))
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.input)

            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.input)

            utterance = utterance[::-1]
            utterances.extend(utterance)
        decoded_batch.append(utterances)
    #decoded_batch = torch.Tensor(decoded_batch)
    #fprint(decoded_batch.size())
    return decoded_batch


