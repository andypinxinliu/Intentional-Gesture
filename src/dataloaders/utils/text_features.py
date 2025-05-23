import textgrid as tg
import numpy as np
import os
from transformers import AutoTokenizer, BertModel
from loguru import logger

def process_word_data(data_dir, word_file, args, data, f_name, selected_file):
    """Process word/text data with support for different encoders."""
    logger.info(f"# ---- Building cache for Word {f_name} ---- #")

    if not os.path.exists(word_file):
        logger.warning(f"# ---- file not found for Word {f_name}, skip all files with the same id ---- #")
        selected_file.drop(selected_file[selected_file['id'] == f_name].index, inplace=True)
        return None

    word_save_path = f"{data_dir}{args.t_pre_encoder}/{f_name}.npy"
    if os.path.exists(word_save_path):
        data['word'] = np.load(word_save_path)
        logger.warning(f"# ---- file found cache for Word {f_name} ---- #")
        return data

    tgrid = tg.TextGrid.fromFile(word_file)
    word_data = []
    
    if args.t_pre_encoder == "bert":
        word_data = process_bert_encoding(tgrid, f_name, args)
    else:
        word_data = process_basic_encoding(tgrid, data, args)

    data['word'] = np.array(word_data)
    os.makedirs(os.path.dirname(word_save_path), exist_ok=True)
    np.save(word_save_path, data['word'])
    return data

def process_bert_encoding(tgrid, f_name, args):
    """Process text data using BERT encoding."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.data_path_1 + "hub/bert-base-uncased", 
        local_files_only=True
    )
    model = BertModel.from_pretrained(
        args.data_path_1 + "hub/bert-base-uncased", 
        local_files_only=True
    ).eval()
    
    list_word = []
    all_hidden = []
    word_token_mapping = []
    max_len = 400
    global_len = 0
    
    for i, word in enumerate(tgrid[0]):
        if i % max_len == 0 and i > 0:
            # Process current batch
            encoded_data = process_bert_batch(
                list_word, tokenizer, model, word_token_mapping, global_len
            )
            all_hidden.append(encoded_data['hidden_states'])
            global_len = encoded_data['global_len']
            list_word = []
            
        list_word.append("." if word.mark == "" else word.mark)
    
    # Process remaining words
    if list_word:
        encoded_data = process_bert_batch(
            list_word, tokenizer, model, word_token_mapping, global_len
        )
        all_hidden.append(encoded_data['hidden_states'])
    
    return np.concatenate(all_hidden, axis=0) if all_hidden else np.array([])

def process_bert_batch(word_list, tokenizer, model, word_token_mapping, global_len):
    """Process a batch of words through BERT."""
    str_word = ' '.join(word_list)
    
    # Get token mappings
    token_offsets = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
    word_offsets = get_word_offsets(word_list)
    
    # Map words to tokens
    for start, end in word_offsets:
        sub_mapping = []
        for i, (start_t, end_t) in enumerate(token_offsets[1:-1]):
            if int(start) <= int(start_t) and int(end_t) <= int(end):
                sub_mapping.append(i + global_len)
        word_token_mapping.append(sub_mapping)
    
    # Get BERT embeddings
    with torch.no_grad():
        inputs = tokenizer(str_word, return_tensors="pt")
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
    
    return {
        'hidden_states': hidden_states,
        'global_len': word_token_mapping[-1][-1] + 1 if word_token_mapping else global_len
    }

def get_word_offsets(word_list):
    """Calculate character offsets for each word in the list."""
    offsets = []
    current_pos = 0
    
    for word in word_list:
        start = current_pos
        end = start + len(word)
        offsets.append((start, end))
        current_pos = end + 1  # +1 for the space
        
    return offsets

def process_basic_encoding(tgrid, data, args):
    """Process basic word encoding."""
    word_data = []
    for i in range(data['pose'].shape[0]):
        current_time = i/args.pose_fps
        found_word = False
        
        for word in tgrid[0]:
            if word.minTime <= current_time <= word.maxTime:
                if word.mark == " ":
                    word_data.append(args.lang_model.PAD_token)
                else:
                    word_data.append(args.lang_model.get_word_index(word.mark))
                found_word = True
                break
                
        if not found_word:
            word_data.append(args.lang_model.UNK_token)
            
    return word_data