# listen-attend-spell



## The model

The Listen, Attend and Spell model is an encoder-decoder neural network used for the task of automatic speech recognition (from speech to text). The encoder, named the *listener*, is a pyramidal RNN network that converts a speech signlal into a higher level feature reprentation. The decoder, named the *speller*, is an RNN takes these high level features and outputs a probability distribution over sequences of sequences of characters. The model is trained end-to-end.

### Listener

The listener is an acoustic encoder that take as input a spectrogram-like representation **x** = (x<sub>1</sub>, ..., x<sub>T</sub>), where each x<sub>i</sub> is a time frame of our spectrogram representation. The goal of the listener is to map this input representation into some high level feature **h** = (h<sub>1</sub>, ..., h<sub>U</sub>), with the key constraint that U < T. Thus, the listener must reduced the number of time steps of the original signal into a more compressed representation **h**, allowing the attend and spell layer to extract relevant information from a reduced number of time steps.

The listener architecture is constructed by stacking multiple Bidirectional Long Short Term Memory RNN (BLSTM), that creates a pyramidal structure with multiple BLSM layers.
The time step reduction is achive by concatenating two successive (in time) BLSTM outputs at each layer before feading them to the next BLSTM layer in the pyramid. Thus, the time resolution is reduce by a factor of 2 for each layer in the pyramid, i.e a 3 BLSTM layers pyramid performs a time reduction of 2<sup>3</sup> = 8.

### Attend and Spell

The attend and spell is an attention-based LSTM transducer. Thus, at every output step it produces the probability distribution for the next character  (over all the possible characters in the dictionary) **conditioned** on all the characters previously produced in output. This solve the issue of CTC, that assumes that the label outputs are conditionally **independent** of each other. Also, by directly producing characters in output there is no problem for Out-Of-Vocabulary (OOV) words.

The attend and spell architecture can be described as: 
<div align='center'>
  c<sub>i</sub> = AttentionContext(s<sub>i</sub>, <b>h</b>) </br>
  s<sub>i</sub> = RNN(s<sub>i-1</sub>, y<sub>i-1</sub>, c<sub>i-1</sub>) </br>
  P(y<sub>i</sub> | x, y<sub>< i</sub>) = CharacterDistribution(s<sub>i</sub>, c<sub>i</sub>) </br>
</div>

where i is the current time step, c is the context and s is the RNN state. 
The context c is computed using an attention mechanism and encapsulate the information of the acoustic signal needed to generate the next character. The
attention model is content based - the contents of the decoder state at each time step i s<sub>i</sub> are matched to the contents of all h<sub>u</sub> in **h**. Thus, at each time step we compare the current RNN state s<sub>i</sub> with all the acoustic information of input signal x encoded in **h** and keep the most relevant ones in the context c<sub>i</sub>. On convergence,the network learns to focused on only a few frames of **h**; c<sub>i</sub> can be seen as a continuous bag of weighted features of **h**.

The RNN network is a multi-layer LSTM network and the CharatectDistribution is an MLP network with softmax output over all the characters in the dictonary.

### Training

During training, teacher forcing is used. Thus, the network maximizes the log probability: 
<div align='center'>
  log P(y<sub>i</sub> | x, y<sup>*</sup><sub>< i</sub>) </br>
</div>

where y<sup>*</sup><sub>< i</sub> is the ground-truth character at time step i. During training the input of the multi-layer LSTM network in the attend and spell layers is the ground-truth sequence.
