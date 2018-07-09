# Content
Extends the standard keras LSTM and GRU layer with layer normalization, as described in here https://arxiv.org/pdf/1607.06450.pdf .

# Example usage
The layers can be easily used like the normal layers:

```python
from LayerNormalizationRNN import LSTM, GRU

inputs = Input(shape=(maxlen,))
x = Embedding(max_features, 128)(inputs)

x = LSTM(64, layer_to_normalize=("input", "output", "recurrent"), normalize_seperately=True)(x)
# x = GRU(64, layer_to_normalize=("input_gate", "input_recurrent", "recurrent_gate", "recurrent_recurrent"), normalize_seperately=True)(x)

predictions = Dense(1, activation='sigmoid')(x)


model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

# Explanation of arguments 
The layer ```layer_to_normalize``` arguments specifies, after which matrix multiplication the layer normalization should be applied (see equations below). 

The ```normalize_seperately``` argument specifies, whether the matrix multiplication for the forget, input, output... gates should be interpreted as one big 
one, or whether they should be split up in 4(LSTM)/2(GRU) smaller matrix multiplications, on which the layer normalization is applied.

![alt text](https://github.com/Binbose/keras-layer-normalization-rnn/blob/master/images/LSTM_explanation_with_arrows.jpg)
![alt text](https://github.com/Binbose/keras-layer-normalization-rnn/blob/master/images/GRU_explanation_with_arrows.jpg)

# Notes
For the LSTM layer, this implementation works with the ```implementaion=1``` and ```implementation=0``` flag. For GRU only ```implementation=1``` is supported.

# TODO
Test implementation on one of the experimental setups from the paper
Think of better names for the flags
