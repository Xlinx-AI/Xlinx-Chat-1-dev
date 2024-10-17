# ==========================================
# Install Necessary Libraries
# ==========================================

# Uncomment and run these lines if you're setting up a new environment.
# They ensure the correct versions of PyTorch and other dependencies are installed.

# !pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# !pip install transformers datasets pillow fastapi uvicorn tiktoken einops tensorboard
# !pip install faiss-cpu slowapi

# ==========================================
# Imports and Device Initialization
# ==========================================

import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from PIL import Image
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader, Dataset
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid

# Import the model and tokenizer from the training script or ensure they are accessible
from training_script import OmniModalLLM, LiquidFoundationTokenizer, device

# ==========================================
# API Models
# ==========================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # To manage conversation sessions
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage

# ==========================================
# FastAPI Setup
# ==========================================

app = FastAPI()

# Initialize Limiter for Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================
# Initialize Model and Tokenizer
# ==========================================

def initialize_model_and_tokenizer(device: torch.device):
    """
    Initializes the OmniModalLLM model and the LiquidFoundationTokenizer.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        model (OmniModalLLM): The multimodal model.
        tokenizer (LiquidFoundationTokenizer): The tokenizer for processing text and images.
    """
    token_dim = 512
    channel_dim = 512
    expert_dim = 256    # Adjust as per training
    adapt_dim = 128     # Adjust as per training
    num_experts = 4     # Adjust as per training
    num_layers = 3      # Adjust as per training
    hidden_dim = 64
    num_heads = 8

    # Initialize the tokenizer
    tokenizer = LiquidFoundationTokenizer(device=device, adapt_dim=adapt_dim)

    # Initialize the model
    model = OmniModalLLM(
        token_dim=token_dim,
        channel_dim=channel_dim,
        expert_dim=expert_dim,
        adapt_dim=adapt_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=0.1,
        max_drop_prob=0.1,
        layerdrop_prob=0.1,
        dropblock_block_size=7,
        dropblock_prob=0.1,
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.5
    ).to(device)
    
    # Load trained weights
    model.load_model('final_model.pth.tar')
    
    return model, tokenizer

model, tokenizer = initialize_model_and_tokenizer(device=device)

# ==========================================
# Conversation History Storage
# ==========================================

# Simple in-memory storage for session histories
from collections import defaultdict
import threading

conversation_history = defaultdict(list)
history_lock = threading.Lock()

# ==========================================
# Inference Function with Generation Loop
# ==========================================

def generate_response_api(
    model: OmniModalLLM,
    tokenizer: LiquidFoundationTokenizer,
    user_text: str,
    session_id: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> str:
    """
    Generates a response from the assistant based on user input and conversation history.

    Args:
        model (OmniModalLLM): The trained multimodal model.
        tokenizer (LiquidFoundationTokenizer): Tokenizer for processing text and images.
        user_text (str): The latest message from the user.
        session_id (str): Identifier for the conversation session.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        top_p (float): Nucleus (top-p) sampling.

    Returns:
        response_text (str): The assistant's generated response.
    """
    model.eval()  # Set model to evaluation mode
    generated_tokens = []
    with torch.no_grad():
        # Retrieve conversation history for the session
        with history_lock:
            history = conversation_history.get(session_id, []).copy()
        
        # Concatenate all user and assistant messages
        conversation = ""
        for msg in history:
            if msg.role == 'user':
                conversation += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                conversation += f"Assistant: {msg.content}\n"
        
        # Append the latest user message
        conversation += f"User: {user_text}\nAssistant:"
        
        for _ in range(max_new_tokens):
            # Tokenize the current conversation
            tokenized = tokenizer.text_tokenizer.tokenize(conversation)
            tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq]
            
            # Forward pass through the model
            outputs = model(tokens, image_embeddings=None)  # Assuming text-only for chat
            token_logits = outputs["token_logits"]  # [batch, vocab_size]
            
            # Apply temperature scaling
            token_logits = token_logits / temperature
            
            # Apply top-k and top-p filtering
            sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            token_logits.scatter_(1, indices_to_remove, float('-inf'))

            # Sample the next token
            probabilities = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # [batch, 1]

            # Detokenize to get the token string
            token_id = next_token.squeeze(0).cpu().numpy()
            token_str = tokenizer.text_tokenizer.detokenize(next_token.squeeze(0))
            
            # Append the token to the conversation
            conversation += token_str
            generated_tokens.append(token_str)
            
            # Check for end-of-sequence token
            if tokenizer.encoder.eos_token_id and next_token.item() == tokenizer.encoder.eos_token_id:
                break

    response_text = ''.join(generated_tokens).strip()
    return response_text

# ==========================================
# API Endpoint
# ==========================================

@app.post("/chat/", response_model=ChatResponse)
@limiter.limit("20/minute")  # Limit to 20 requests per minute per IP
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    API endpoint to handle chat messages and generate responses.

    Args:
        request (ChatRequest): Incoming chat request containing messages and optional session_id.
        req (Request): The incoming HTTP request (used for rate limiting).

    Returns:
        ChatResponse: The assistant's response along with the session_id.
    """
    # Generate a new session_id if not provided
    if not request.session_id:
        session_id = str(uuid.uuid4())
        with history_lock:
            conversation_history[session_id] = []
    else:
        session_id = request.session_id
        with history_lock:
            if session_id not in conversation_history:
                conversation_history[session_id] = []
    
    # Append incoming messages to the conversation history
    with history_lock:
        conversation_history[session_id].extend(request.messages)
    
    # Extract the latest user message
    user_message = next((msg for msg in request.messages if msg.role == 'user'), None)
    if not user_message:
        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content="I didn't receive any user message.")
        )
    
    # Generate assistant's response with a generation loop
    assistant_reply = generate_response_api(
        model, 
        tokenizer, 
        user_message.content, 
        session_id=session_id,
        max_new_tokens=100,  # Adjust as needed
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    
    # Append assistant's reply to the conversation history
    with history_lock:
        conversation_history[session_id].append(
            ChatMessage(role="assistant", content=assistant_reply)
        )
    
    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(role="assistant", content=assistant_reply)
    )

# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    # Launch the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
