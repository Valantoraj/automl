import os
import json
import random
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM, Dropout, Dense, 
                                     Add, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, Layer)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------
def clean_text(text):
    """Lowercase and remove punctuation for consistency."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def load_intents(file_path):
    """Load intents from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def prepare_data(intents):
    """
    Preprocess data from the intents JSON.
    
    Returns:
        padded_sequences: Numpy array of padded, tokenized texts.
        labels: Numeric labels for each pattern.
        max_len: Maximum sequence length.
        tokenizer: Fitted Keras Tokenizer.
        label_encoder: Fitted LabelEncoder.
        responses: Dictionary mapping each tag to its responses.
        patterns: List of cleaned text patterns.
        sequences: List of tokenized sequences.
    """
    patterns = []
    tags = []
    responses = {}
    for intent in intents["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            patterns.append(clean_text(pattern))
            tags.append(tag)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(tags)
    
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)
    sequences = tokenizer.texts_to_sequences(patterns)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")
    
    return padded_sequences, np.array(labels), max_len, tokenizer, label_encoder, responses, patterns, sequences

# -----------------------------------
# DYNAMIC TUNING CONFIGURATION
# -----------------------------------
def compute_dynamic_tuned_params(patterns, sequences, vocab_size):
    """
    Compute base tuning parameters dynamically from dataset statistics.
    
    Returns a dictionary with keys:
      - a: Multiplier for embedding dimension.
      - b: Offset for average sequence length.
      - c: Multiplier for LSTM units.
      - layer_offset: Extra layers to add.
      - dropout_small, dropout_medium, dropout_large: Dropout rates based on dataset size.
    """
    num_patterns = len(patterns)
    avg_seq_len = np.mean([len(seq) for seq in sequences])
    
    a = 20 * (vocab_size / (vocab_size + 1000))  # Approaches 20 for large vocabs.
    b = 0.8 * avg_seq_len                        # 80% of average sequence length.
    c = 5 * (avg_seq_len / (avg_seq_len + 10))     # Scales with avg_seq_len.
    layer_offset = min(2, int(num_patterns / 1000))  # Extra layer per ~1000 samples (max 2).
    
    dropout_small = 0.5 if num_patterns < 200 else max(0.2, 0.5 * (200 / num_patterns))
    dropout_medium = 0.3 if num_patterns < 1000 else max(0.1, 0.3 * (1000 / num_patterns))
    dropout_large = 0.2  # Fixed for larger datasets.
    
    return {'a': a, 'b': b, 'c': c, 'layer_offset': layer_offset,
            'dropout_small': dropout_small, 'dropout_medium': dropout_medium, 'dropout_large': dropout_large}

def get_dynamic_candidate_grid(patterns, sequences, vocab_size):
    """
    Build a candidate grid around dynamically computed base tuning parameters.
    
    Returns:
        candidate_grid (dict): Candidate lists for 'a', 'c', and 'layer_offset'.
        fixed_params (dict): Fixed parameters such as 'b' and dropout rates.
        tuning_epochs (int): Number of tuning epochs, set dynamically.
    """
    base = compute_dynamic_tuned_params(patterns, sequences, vocab_size)
    candidate_grid = {
        'a': [base['a'] * 0.9, base['a'], base['a'] * 1.1],
        'c': [base['c'] * 0.9, base['c'], base['c'] * 1.1],
        'layer_offset': [max(0, base['layer_offset'] - 1), base['layer_offset'], base['layer_offset'] + 1]
    }
    fixed_params = {
        'b': base['b'],
        'dropout_small': base['dropout_small'],
        'dropout_medium': base['dropout_medium'],
        'dropout_large': base['dropout_large']
    }
    num_samples = len(patterns)
    tuning_epochs = 3 if num_samples < 200 else (5 if num_samples < 1000 else 7)
    return candidate_grid, fixed_params, tuning_epochs

def tune_heuristics(padded_sequences, labels, patterns, sequences, vocab_size, max_len, num_classes):
    """
    Run a grid search to tune heuristic parameters.
    For each candidate combination, a temporary LSTM model (without residual blocks for speed)
    is trained for a few epochs on a hold-out validation set.
    
    Returns:
        best_params (dict): Tuned heuristic parameters that yield the lowest validation loss.
    """
    candidate_grid, fixed_params, tuning_epochs = get_dynamic_candidate_grid(patterns, sequences, vocab_size)
    best_val_loss = float('inf')
    best_params = None
    
    # Stratified split to ensure each tag is represented in train and validation.
    train_data, val_data, train_labels, val_labels = train_test_split(
        padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    for a in candidate_grid['a']:
        for c in candidate_grid['c']:
            for layer_offset in candidate_grid['layer_offset']:
                candidate_params = {
                    'a': a,
                    'b': fixed_params['b'],
                    'c': c,
                    'layer_offset': layer_offset,
                    'dropout_small': fixed_params['dropout_small'],
                    'dropout_medium': fixed_params['dropout_medium'],
                    'dropout_large': fixed_params['dropout_large']
                }
                num_patterns = len(patterns)
                avg_seq_len = np.mean([len(seq) for seq in sequences])
                embedding_dim = int(np.clip(np.log2(vocab_size) * candidate_params['a'], 50, 300))
                lstm_units = int(np.clip(32 + max(0, avg_seq_len - candidate_params['b']) * candidate_params['c'], 32, 128))
                base_layers = 1 + int(math.log10(max(num_patterns, 10)))
                num_layers = max(1, base_layers + candidate_params['layer_offset'])
                num_layers = min(6, num_layers) if num_patterns > 10000 else min(4, num_layers)
                dropout_rate = (candidate_params['dropout_small'] if num_patterns < 200 
                                else candidate_params['dropout_medium'] if num_patterns < 1000 
                                else candidate_params['dropout_large'])
                
                temp_model = build_lstm_model(vocab_size, max_len, num_classes, embedding_dim, lstm_units, num_layers, dropout_rate, use_residual=False)
                history = temp_model.fit(train_data, train_labels, epochs=tuning_epochs, batch_size=32, 
                                         validation_data=(val_data, val_labels), verbose=0)
                val_loss = history.history['val_loss'][-1]
                print(f"Tuning: a={a:.2f}, c={c:.2f}, layer_offset={layer_offset}, val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = candidate_params
    print("Best tuned heuristic parameters:", best_params, "with val_loss:", best_val_loss)
    return best_params

# -----------------------------------
# FINAL HYPERPARAMETER COMPUTATION
# -----------------------------------
def get_hyperparameters_final(patterns, sequences, vocab_size, max_len, tuned_params):
    """
    Compute final hyperparameters using the tuned parameters.
    
    Returns:
        embedding_dim, lstm_units, num_layers, dropout_rate
    """
    num_patterns = len(patterns)
    avg_seq_len = np.mean([len(seq) for seq in sequences])
    
    embedding_dim = int(np.clip(np.log2(vocab_size) * tuned_params['a'], 50, 300))
    added_units = max(0, avg_seq_len - tuned_params['b']) * tuned_params['c']
    lstm_units = int(np.clip(32 + added_units, 32, 128))
    
    base_layers = 1 + int(math.log10(max(num_patterns, 10)))
    num_layers = max(1, base_layers + tuned_params['layer_offset'])
    num_layers = min(6, num_layers) if num_patterns > 10000 else min(4, num_layers)
    
    if num_patterns < 200:
        dropout_rate = tuned_params['dropout_small']
    elif num_patterns < 1000:
        dropout_rate = tuned_params['dropout_medium']
    else:
        dropout_rate = tuned_params['dropout_large']
    
    return embedding_dim, lstm_units, num_layers, dropout_rate

# -----------------------------------
# MODEL BUILDING: LSTM-BASED (with Residual Blocks)
# -----------------------------------
def build_lstm_model(vocab_size, max_len, num_classes, embedding_dim, lstm_units, num_layers, dropout_rate, use_residual=True):
    """
    Build an LSTM-based model with dynamic stacking.
    If use_residual is True and num_layers >= 2, layers are grouped in blocks with residual connections.
    A GlobalAveragePooling1D layer is applied if the output is 3D.
    """
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    
    if use_residual and num_layers >= 2:
        num_blocks = num_layers // 2
        remainder = num_layers % 2
        for i in range(num_blocks):
            y = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
            y = Dropout(dropout_rate)(y)
            y = LayerNormalization()(y)
            ret_seq = True if (i < num_blocks - 1 or remainder == 1) else False
            y = Bidirectional(LSTM(lstm_units, return_sequences=ret_seq))(y)
            y = Dropout(dropout_rate)(y)
            y = LayerNormalization()(y)
            if x.shape[-1] != y.shape[-1]:
                x = Dense(y.shape[-1])(x)
            x = Add()([x, y])
            x = LayerNormalization()(x)
        if remainder == 1:
            x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
    else:
        for i in range(num_layers):
            ret_seq = True if i < num_layers - 1 else False
            x = Bidirectional(LSTM(lstm_units, return_sequences=ret_seq))(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
    
    # Ensure output is 2D.
    if len(x.shape) == 3:
        x = GlobalAveragePooling1D()(x)
    
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

# -----------------------------------
# MODEL BUILDING: TRANSFORMER-BASED
# -----------------------------------
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(vocab_size, max_len, num_classes, embedding_dim, num_layers, dropout_rate):
    """
    Build a Transformer-based model for text classification.
    Uses learnable positional embeddings and stacks Transformer blocks.
    """
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
    
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_embedding = Embedding(input_dim=max_len, output_dim=embedding_dim)(positions)
    x = x + pos_embedding
    
    for _ in range(num_layers):
        transformer_block = TransformerBlock(embed_dim=embedding_dim, num_heads=4, ff_dim=embedding_dim*2, rate=dropout_rate)
        x = transformer_block(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

# -----------------------------------
# TRAINING & INTERACTIVE CHAT LOOP
# -----------------------------------
def train_model(model, padded_sequences, labels, epochs=200, batch_size=32, stratified=True):
    """Train the model with early stopping using a stratified split if specified."""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    if stratified:
        train_data, val_data, train_labels, val_labels = train_test_split(
            padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42
        )
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                  validation_data=(val_data, val_labels), callbacks=[early_stopping], verbose=1)
    else:
        model.fit(padded_sequences, labels, epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, callbacks=[early_stopping], verbose=1)
    return model

def chatbot_response(text, tokenizer, max_len, model, label_encoder, responses):
    """Generate a chatbot response for the given user input."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(padded_seq)
    tag_index = np.argmax(pred)
    tag = label_encoder.inverse_transform([tag_index])[0]
    return random.choice(responses[tag])

def chat_loop(tokenizer, max_len, model, label_encoder, responses):
    """Interactive chat loop for testing the chatbot."""
    print("\nChatbot is ready! Type 'quit' or 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        reply = chatbot_response(user_input, tokenizer, max_len, model, label_encoder, responses)
        print("Chatbot:", reply)

# -----------------------------------
# MAIN FUNCTION: AUTOMATIC ARCHITECTURE & DYNAMIC TUNING
# -----------------------------------
def main(intents):
    # Prepare data.
    padded_sequences, labels, max_len, tokenizer, label_encoder, responses, patterns, sequences = prepare_data(intents)
    vocab_size = len(tokenizer.word_index) + 1  # Account for OOV.
    num_classes = len(label_encoder.classes_)
    
    # Automatically tune heuristic parameters dynamically.
    tuned_params = tune_heuristics(padded_sequences, labels, patterns, sequences, vocab_size, max_len, num_classes)
    
    # Compute final hyperparameters using tuned parameters.
    embedding_dim, lstm_units, num_layers, dropout_rate = get_hyperparameters_final(patterns, sequences, vocab_size, max_len, tuned_params)
    print("\nFinal Determined Hyperparameters:")
    print("Embedding Dimension:", embedding_dim)
    print("LSTM Units:", lstm_units)
    print("Number of Layers:", num_layers)
    print("Dropout Rate:", dropout_rate)
    
    # Automatically decide on model architecture based on dataset characteristics.
    avg_seq_len = np.mean([len(seq) for seq in sequences])
    use_transformer = True if (len(patterns) > 500 and avg_seq_len > 20) else False
    print("Using Transformer-based model:", use_transformer)
    
    # Build the chosen model.
    if use_transformer:
        model = build_transformer_model(vocab_size, max_len, num_classes, embedding_dim, num_layers, dropout_rate)
    else:
        model = build_lstm_model(vocab_size, max_len, num_classes, embedding_dim, lstm_units, num_layers, dropout_rate, use_residual=True)
    
    # Train the model (using stratified split).
    print("\nTraining model...")
    model = train_model(model, padded_sequences, labels, epochs=200, batch_size=32, stratified=True)
    
    # Interactive chat loop.
    chat_loop(tokenizer, max_len, model, label_encoder, responses)

