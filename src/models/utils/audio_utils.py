import numpy as np
import soundfile
import librosa
import torch
import torch.nn.functional as F
import difflib

def audio_to_time_aligned_text_features(inputs, processor, model, tokenizer, bert_model):  
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # shape: (1, time_steps, vocab_size)

    predicted_ids_per_timestep = torch.argmax(logits, dim=-1)  # shape: (1, time_steps)
    predicted_ids_per_timestep = predicted_ids_per_timestep[0].cpu().numpy()
    vocab = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens_per_timestep = [id_to_token[id] for id in predicted_ids_per_timestep]

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    inputs_bert = tokenizer(transcription, return_tensors='pt')
    input_ids = inputs_bert['input_ids'][0]  
    tokens_bert = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs_bert = bert_model(**inputs_bert.to(inputs.input_values.device))
    all_token_embeddings = outputs_bert.last_hidden_state[0]  
    per_timestep_chars = []
    per_timestep_char_indices = []
    for idx, t in enumerate(tokens_per_timestep):
        if t not in ('<pad>', '|'):
            per_timestep_chars.append(t.lower())
            per_timestep_char_indices.append(idx)
    bert_chars = []
    bert_char_indices = []
    for idx, token in enumerate(tokens_bert):
        if token in ('[CLS]', '[SEP]'):
            continue
        token_str = token.replace('##', '')
        for c in token_str:
            bert_chars.append(c)
            bert_char_indices.append(idx)

    s = difflib.SequenceMatcher(None, per_timestep_chars, bert_chars)
    opcodes = s.get_opcodes()
    per_timestep_to_bert_token_idx = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i2 - i1):
                per_timestep_idx = per_timestep_char_indices[i1 + k]
                bert_token_idx = bert_char_indices[j1 + k]
                per_timestep_to_bert_token_idx[per_timestep_idx] = bert_token_idx
    features_per_timestep = []
    check = []
    for i, per_token in enumerate(tokens_per_timestep):
        if i == 0:
            embedding = all_token_embeddings[0]
            check.append("cls")
        elif per_token in ('<pad>', '|'):
            embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
            check.append(0)
        else:
            if i in per_timestep_to_bert_token_idx:
                bert_idx = per_timestep_to_bert_token_idx[i]
                embedding = all_token_embeddings[bert_idx]
                check.append(tokens_bert[bert_idx])
            else:
                embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
                check.append(0)
        features_per_timestep.append(embedding)
    features_per_timestep = torch.stack(features_per_timestep)  

    updated_check = check.copy()
    for i in range(len(check)):
        if check[i] == 0:
            left = i - 1
            right = i + 1
            left_found = False
            right_found = False

            while left >= 0:
                if check[left] != 0:
                    left_found = True
                    break
                left -= 1

            while right < len(check):
                if check[right] != 0:
                    right_found = True
                    break
                right += 1

            if left_found and right_found:
                if (i - left) <= (right - i):
                    nearest = left
                else:
                    nearest = right
            elif left_found:
                nearest = left
            elif right_found:
                nearest = right
            else:
                continue
            updated_check[i] = updated_check[nearest]
            features_per_timestep[i] = features_per_timestep[nearest]
    features_per_timestep = features_per_timestep.unsqueeze(0)
    return transcription, features_per_timestep, all_token_embeddings 


