import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Настройка страницы
st.set_page_config(
    page_title="Генератор историй",
    page_icon="📖",
    layout="centered"
)

# Заголовок приложения
st.title("📖 Генератор историй")
st.markdown("Введите запрос и получите уникальную историю!")

# Инициализация модели (кэшируем, чтобы не загружать каждый раз)
@st.cache_resource
def load_story_generator():
    """Загрузка модели для генерации текста"""
    try:
        # Используем модель для генерации текста на русском языке
        model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Создаем пайплайн для генерации текста
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1  # Используем GPU если доступен
        )
        return generator
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

# Загрузка модели
with st.spinner("Загружаем модель для генерации историй..."):
    generator = load_story_generator()

# Интерфейс пользователя
user_input = st.text_area(
    "Введите запрос для истории:",
    placeholder="Например: 'История о рыцаре, который нашёл магический меч в древнем лесу...'",
    height=100
)

max_length = st.slider("Максимальная длина текста", 100, 500, 250)

# Кнопка генерации
generate_button = st.button("🎭 Сгенерировать историю", type="primary")

# Обработка генерации
if generate_button:
    if not user_input.strip():
        st.warning("Пожалуйста, введите запрос для генерации истории.")
    elif generator is None:
        st.error("Модель не загружена. Пожалуйста, проверьте подключение к интернету.")
    else:
        with st.spinner("Генерируем вашу историю... Это может занять несколько секунд."):
            try:
                # Генерация текста
                generated_text = generator(
                    user_input,
                    max_length=max_length,
                    temperature=1,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                # Извлекаем сгенерированный текст
                story = generated_text[0]['generated_text']
                
                # Отображаем результат
                st.success("✅ История сгенерирована!")
                st.subheader("Ваша история:")
                st.write(story)
                
                # Кнопка для копирования текста
                st.code(story, language="text")
                
            except Exception as e:
                st.error(f"Произошла ошибка при генерации: {e}")