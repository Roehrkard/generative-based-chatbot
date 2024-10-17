import json
import logging
from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the SQuAD v2.0 dataset
with open('train-v2.0.json', 'r') as f:
    data = json.load(f)

# Extract contexts from the dataset
contexts = []
for article in data['data']:
    for paragraph in article['paragraphs']:
        contexts.append(paragraph['context'])

# Load pre-trained BERT model and tokenizer for QA
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Load SBERT model for improved context matching
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all contexts using SBERT for fast similarity search
context_embeddings = sbert_model.encode(contexts, convert_to_tensor=True)

def find_best_context(question):
    """Find the most relevant context using SBERT-based cosine similarity."""
    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings).flatten()
    best_index = similarities.argmax().item()
    return contexts[best_index], similarities[best_index].item()  # Return context and its similarity score

def generate_answer(question, context):
    """Generate an answer using the BERT QA model."""
    inputs = tokenizer(question, context, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]

    # Get the model outputs (start and end logits)
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Find the most likely start and end of the answer span
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Confidence threshold
    confidence_threshold = 0.4  # Lowered to allow more responses

    # Check if the answer is valid
    start_score = answer_start_scores[0, answer_start].item()
    if start_score < confidence_threshold:
        return "I'm not confident about the answer. Can you please ask differently?"

    # Ensure that the end index is greater than the start index
    if answer_end <= answer_start:
        return "I'm sorry, but I couldn't find an answer in the context provided."

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    
    # Check for coherence of the answer
    if not answer.strip():  # If the answer is empty
        return "I'm sorry, but I couldn't find an answer in the context provided."
    
    return answer

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    user_input = request.json.get('message', '')
    if not user_input.strip():
        return jsonify({'response': "Please ask a valid question."})

    # Find the most relevant context and its similarity score
    context, similarity_score = find_best_context(user_input)
    logging.info(f"User question: {user_input} | Best context similarity: {similarity_score:.2f}")

    # Generate the answer
    answer = generate_answer(user_input, context)
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True)
