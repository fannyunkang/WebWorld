def load_vocabulary(file_path, num_words=None):
    try:
        with open(file_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
            
        if num_words and len(words) >= num_words:
            return words[:num_words]
        return words
    except Exception as e:
        print(f"Loading Failure: {str(e)}")
        return []

def evaluate_response(response, word_list):
    response_words = set(response.lower().split())
    target_words = set(word.lower() for word in word_list)
    matched_words = response_words.intersection(target_words)
    matched_count = len(matched_words)
    total_count = len(target_words)
    ratio = matched_count / total_count if total_count > 0 else 0
    return matched_count, total_count, ratio
