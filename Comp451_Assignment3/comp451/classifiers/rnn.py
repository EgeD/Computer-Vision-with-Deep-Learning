from builtins import range
from builtins import object
import numpy as np

from comp451.layers import *
from comp451.rnn_layers import *


class CaptioningRNN(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32, gclip=0):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.gclip = gclip
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        init_hidden_out, init_cache_out = affine_forward(features,W_proj,b_proj)
        
        word_embed_out, word_cache_out = word_embedding_forward(captions_in,W_embed)
        ##To RNN
        if self.cell_type == "rnn":
            rnn_h, rnn_cache = rnn_forward(word_embed_out,init_hidden_out,Wx,Wh,b)
            scores, scores_cache = temporal_affine_forward(rnn_h,W_vocab,b_vocab)
            
            loss,scores_grads = temporal_softmax_loss(scores,captions_out,mask,verbose=False)
            backward_dscores, backward_dscores_w, backward_dscores_b =temporal_affine_backward(scores_grads,scores_cache)
            
            rnn_backward_x,rnn_backward_h,rnn_backward_Wx,rnn_backward_Wh,rnn_backward_db = rnn_backward(backward_dscores,rnn_cache)
            word_dW = word_embedding_backward(rnn_backward_x, word_cache_out)
            proj_dW = features.T.dot(rnn_backward_h)
            proj_db = rnn_backward_h.sum(axis=0)
            grads = {'W_vocab':backward_dscores_w, 'b_vocab':backward_dscores_b, 'Wx':rnn_backward_Wx, 'Wh':rnn_backward_Wh, 
        'b':rnn_backward_db, 'W_embed':word_dW, 'W_proj':proj_dW, 'b_proj':proj_db}
            if self.gclip > 0:
                clipped_grads = self.clip_grad_norm(grads,self.gclip)
                grads = clipped_grads
            else:
                grads = {'W_vocab':backward_dscores_w, 'b_vocab':backward_dscores_b, 'Wx':rnn_backward_Wx, 'Wh':rnn_backward_Wh, 
        'b':rnn_backward_db, 'W_embed':word_dW, 'W_proj':proj_dW, 'b_proj':proj_db}                
            
        else:
            print("Wrong Cell Type Entry")
            
            
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def clip_grad_norm(self, grads, gclip):
        """
        Inputs:
        - grads: Dictionary of gradients
        - gclip: Max norm for gradients

        Returns a tuple of:
        - clipped_grads: Dictionary of clipped gradients parallel to grads
        """
        clipped_grads = None
        ###########################################################################
        # TODO: Implement gradient clipping using gclip value as the threshold.   #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        backward_dscores_w = grads['W_vocab']
        if np.sum(backward_dscores_w) > gclip:
            backward_dscores_w = backward_dscores_w * gclip / np.linalg.norm(backward_dscores_w,ord = 2)
        
        backward_dscores_b = grads['b_vocab']
        if np.sum(backward_dscores_b) > gclip:
            backward_dscores_b = backward_dscores_b * gclip / np.linalg.norm(backward_dscores_b,ord = 2)
        
        rnn_backward_Wx = grads['Wx']
        if np.sum(rnn_backward_Wx) > gclip:
            rnn_backward_Wx = rnn_backward_Wx * gclip / np.linalg.norm(rnn_backward_Wx,ord = 2)
        
        rnn_backward_Wh = grads['Wh']
        if np.sum(rnn_backward_Wh) > gclip:
            rnn_backward_Wh = rnn_backward_Wh * gclip / np.linalg.norm(rnn_backward_Wh,ord = 2)
        
        rnn_backward_db = grads['b']
        if np.sum(rnn_backward_db) > gclip:        
            rnn_backward_db = rnn_backward_db * gclip / np.linalg.norm(rnn_backward_db,ord = 2)
        
        word_dW = grads['W_embed']
        if np.sum(word_dW) > gclip:        
            word_dW = word_dW * gclip / np.linalg.norm(word_dW,ord = 2)
        
        proj_dW = grads['W_proj']
        if np.sum(proj_dW) > gclip:        
            proj_dW = proj_dW * gclip / np.linalg.norm(proj_dW, ord = 2)
        
        proj_db = grads['b_proj']
        if np.sum(proj_db) > gclip:        
            proj_db = proj_db * gclip / np.linalg.norm(proj_db, ord = 2)

        clipped_grads = {'W_vocab':backward_dscores_w, 'b_vocab':backward_dscores_b, 'Wx':rnn_backward_Wx, 'Wh':rnn_backward_Wh, 
        'b':rnn_backward_db, 'W_embed':word_dW, 'W_proj':proj_dW, 'b_proj':proj_db}    
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return clipped_grads


    def sample_greedily(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        V,W = W_embed.shape
        first_hidden = features.dot(W_proj) + b_proj
        init_x = np.ones((N,W))*W_embed[self._start]
        for i in range(max_length):
            next_hidden,cache = rnn_step_forward(init_x,first_hidden,Wx,Wh,b)
            score = next_hidden.dot(W_vocab)+b_vocab
            captions[:,i] = score.argmax(axis=1)
            init_x = W_embed[score.argmax(axis=1)]
            first_hidden = next_hidden
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions


    def sample_randomly(self, features, max_length=30):

        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        Instead of picking the word with the highest probability, you will sample
        it using the probability distribution. You can use np.random.choice
        to sample the word using probabilities.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """

        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        V,W = W_embed.shape
        first_hidden = features.dot(W_proj) + b_proj
        init_x = np.ones((N,W))*W_embed[self._start]
        for i in range(max_length):
            next_hidden,cache = rnn_step_forward(init_x,first_hidden,Wx,Wh,b)
            score = next_hidden.dot(W_vocab)+b_vocab
            score_softmax = np.exp(score) / np.sum(np.exp(score), axis=1)
            captions[:,i] = np.random.choice(V,1,p=score_softmax[0,:])
            init_x = W_embed[np.random.choice(V,1,p=score_softmax[0,:])]
            first_hidden = next_hidden

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions

