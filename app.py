import os
import io
import torch
from PIL import Image
import streamlit as st
from diffusers import StableDiffusionPipeline

st.title("🎨 Генерация изображений")
st.markdown("Введите запрос для генерации изображения")

@st.cache_resource
def load_model():
    try:
        hf_token = st.secrets["HF_TOKEN"]
            
        model_id = "prompthero/openjourney"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=hf_token
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None