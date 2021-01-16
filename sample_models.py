from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, layers=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        #Modify code for multiple layers of CNN
        output_length = input_length
        for i in range(layers):
            output_length = output_length - dilated_filter_size + 1
            output_length = (output_length + stride - 1) // stride
        return output_length
    
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    inputs = [input_data]
    for i in range(recur_layers):
        simp_rnn = SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2)(inputs[i])
        bn_rnn = BatchNormalization()(simp_rnn)
        inputs.append(bn_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def test_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, pool = 3, pool_stride=1, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network  
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(bn_cnn)
    bn_cnn = BatchNormalization(name='bn_conv_1d_1')(conv_1d)

    
    max_pool = MaxPooling1D(pool_size=pool, strides=pool_stride, padding='same' )(bn_cnn)
    bidir_rnn = Bidirectional(SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2), merge_mode='concat')(max_pool)
    bn_bidir_rnn = BatchNormalization()(bidir_rnn)
    dropout_bidir_rnn = Dropout(0.2)(bn_bidir_rnn, training=True)

    bidir_rnn = Bidirectional(SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2), merge_mode='concat')(dropout_bidir_rnn)
    bn_bidir_rnn = BatchNormalization()(bidir_rnn)
    dropout_bidir_rnn = Dropout(0.2)(bn_bidir_rnn, training=True)
    
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(dropout_bidir_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #Modify code to account for MaxPooling1D layer
    model.output_length = lambda x: (cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride) - pool + 1) // pool_stride
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, pool = 3, pool_stride=1, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network    
    #First cnn layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    #Batch normalization after cnn layer
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    #Second cnn layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d_1')(bn_cnn)
    #Batch normalization after cnn layer
    bn_cnn = BatchNormalization(name='bn_conv_1d_1')(conv_1d)
    
    #Max pooling to prevent overfitting
    max_pool = MaxPooling1D(pool_size=pool, strides=pool_stride, name='max_pooling')(bn_cnn)
    
    #First bidirectional SimpleRNN layer
    bidir_rnn = Bidirectional(SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2), merge_mode='concat', name='Bidirectional_RNN')(bn_cnn)
    #Batch normalization to speed up training
    bn_bidir_rnn = BatchNormalization(name='Batch_Normalisation')(bidir_rnn)
    #Dropout layer after bidirectional RNN layer to prevent overfitting
    dropout_bidir_rnn = Dropout(0.2, name='Dropout')(bn_bidir_rnn, training=True)

    #Second bidirectional SimpleRNN layer
    bidir_rnn = Bidirectional(SimpleRNN(units, activation='tanh',
        return_sequences=True, implementation=2), merge_mode='concat', name='Birectional_RNN_1')(dropout_bidir_rnn)
    #Batch normalization to speed up training
    bn_bidir_rnn = BatchNormalization(name='Batch_Normalization_1')(bidir_rnn)
    
    #Dropout layer after bidirectional RNN layer to prevent overfitting
    dropout_bidir_rnn = Dropout(0.2, name='Dropout_1')(bn_bidir_rnn, training=True)
   
    #Time distributed layer to combine outputs from previous layers
    time_dense = TimeDistributed(Dense(output_dim), name='time_dense')(dropout_bidir_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #MOdify code to take into account max pooling layer
    model.output_length = lambda x: (cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride) - pool + 1) // pool_stride
    print(model.summary())
    return model
