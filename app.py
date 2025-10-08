import os
import torch
from PIL import Image
import streamlit as st
from diffusers import StableDiffusionPipeline

st.title("Генерация изображений")
st.markdown("Введите запрос, чтобы сгенерировать изображение!")

@st.cache_resource
def load_model():
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)

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
        st.error(f"Ошибка при загрузке модели: {e}")
        return None


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Введите запрос")
    prompt = st.text_area(
        "Опишите изображение:",
        placeholder="Красивый закат над горами...",
        height=100
    )
    generate_btn = st.button("Сгенерировать", type="primary", use_container_width=True)

with col2:
    st.subheader("Сгенерированное изображение")
    image_placeholder = st.empty()
    image_placeholder.info("Сгенерированное изображение появится здесь...")

with st.spinner("Загружаем модель..."):
    pipe = load_model()

if generate_btn:
    if not prompt:
        st.warning("Введите запрос!")
    elif pipe is None:
        st.error("Ошибка при загрузке модели")
    else:
        try:
            with st.spinner("Генерируем изображение..."):
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
                image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]

                image_placeholder.image(image, caption=f"'{prompt}'", use_column_width=True)
                st.success("Изображение успешно сгенерировано!")
        except Exception as e:
            st.error(f"Ошибка при генерации: {e}")
