import streamlit as st
import torch
from model import LanguageModel

# Load the pre-trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LanguageModel()
model.load_state_dict(torch.load('model_state.pth', map_location=device))
model.to(device)
model.eval()


with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
            
# taking all the unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating a mapping for integer to string and vice-versa
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes an string and returns list of integers
decode = lambda l: "".join([itos[i] for i in l]) # decoder: takes an list of integers and returns a string


# Function to generate text
def generate_text(context, max_new_tokens):
    # Encode the input context into numerical tokens
    context_ids = encode(context)

    # Create a tensor from the encoded context
    context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)

    # Generate text based on the context tensor
    generated_ids = model.generate(context_tensor, max_new_tokens=max_new_tokens)
    generated_text = decode(generated_ids.squeeze().tolist())
    return generated_text


# Streamlit UI
def main():
    st.title("Text Generation with QuillNet")

    # Input for starting text
    context = st.text_input("Enter starting text:", "")

    # Button to generate text
    if st.button("Generate"):
        st.write("Generated Text:")
        generated_text = generate_text(context, 1000)
        st.write(generated_text)

if __name__ == "__main__":
    main()
