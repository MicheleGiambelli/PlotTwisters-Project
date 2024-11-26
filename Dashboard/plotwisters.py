from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

base_path = r"C:\Users\miche.LAPTOP-KKEENNGV\OneDrive\Desktop\Università\2° Anno\Data Visualization and Text Mining\Project"


def plotwisters_model(text, model_path, model_name):
    # Model BiLSTM
    if model_name == "GRU":
        # Upload tokenizer
        with open("word2idx.pkl", 'rb') as f:
            word2idx = pickle.load(f)
        #Upload dicitonary for tag
        with open("idx2tag.pkl", 'rb') as f:
            idx2tag = pickle.load(f)
        
        model = load_model("GRU-best-model.keras")
        sequence = word2idx.texts_to_sequences([text])
        sequence_padded = pad_sequences(sequence, maxlen=28)

        predictions = model.predict(sequence_padded)
        predictions = np.argmax(predictions, axis=-1)
        predicted_labels = []
        for i, sentence in enumerate(predictions):
            sentence_labels = []  # Temporary list for single sentence tags
            for tag in sentence[-len(sequence[i]):]:   # Exclude the padding at the beginning of each sentence
                sentence_labels.append(idx2tag[tag])
            predicted_labels.append(sentence_labels)

        result = "\n".join(f"{word2idx.index_word[token]} {tag}" for token, tag in zip(sequence[0], predicted_labels[0]))
        # result = [(word2idx.index_word[token], tag) for token, tag in zip(sequence[0], predicted_labels[0])]
        return result

    elif model_name == "DistilBERT":
        with open("id2tag_bert.pkl", 'rb') as f:
            id2tag = pickle.load(f)

        # Modello BERT
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForTokenClassification.from_pretrained(base_path)

        # Tokenizza la frase
        inputs = tokenizer.encode_plus(
            text,
            return_tensors='pt',             # Restituisce tensori PyTorch
            is_split_into_words=False,       # Input come frase completa
            truncation=True,
            max_length=28                    # Lunghezza massima come nei dati di training
        )
        input_ids = inputs['input_ids']

        model.eval()  # Metti il modello in modalità valutazione

        # Ottieni le predizioni
        with torch.no_grad():
            outputs = model(input_ids)  # Ottieni i logits
        predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()

        # Converti input_ids in token
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Ricostruisci parole e associa etichette
        words = []
        labels = []
        
        for idx, (token, pred) in enumerate(zip(tokens, predictions)):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
            if token.startswith('##'):
                words[-1] += token[2:]
            else:
                words.append(token)
                labels.append(id2tag[pred])

        # Associa etichette ai token
        result = "\n".join(f"{word} {label}" for word, label in zip(words, labels))
        return result

    else:
        raise ValueError("Model type not supported. Choose 'BiLSTM' or 'BERT'.")