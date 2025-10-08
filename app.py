import os
import io
import torch
from PIL import Image
import streamlit as st
from diffusers import StableDiffusionPipeline

st.title("🎨 AI Image Generator with OpenJourney")