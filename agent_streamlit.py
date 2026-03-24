# day6_agent_streamlit.py
# Установка: pip install streamlit openai python-dotenv

import os
import json
from datetime import datetime
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# КЛАСС АГЕНТА (тот же, что и в CLI-версии)
# ============================================================
class SimpleAgent:
    def __init__(self, model="stepfun/step-3.5-flash:free", system_prompt=None, max_history=20):
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history = []
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://your-domain.ru"),
                "X-Title": "AI Challenge Day 6 - Web Agent"
            }
        )
        
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
    
    def think(self, user_input):
        if not user_input or not user_input.strip():
            return "❌ Пустой запрос. Напиши что-нибудь!"
        
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history.copy(),
                temperature=0.7,
                max_tokens=500
            )
            agent_response = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": agent_response})
            return agent_response
        except Exception as e:
            return f"❌ Ошибка API: {str(e)}"
    
    def _trim_history(self):
        if len(self.history) > self.max_history:
            if self.system_prompt and self.history[0]["role"] == "system":
                system_msg = self.history[0]
                self.history = [system_msg] + self.history[-(self.max_history - 1):]
            else:
                self.history = self.history[-self.max_history:]
    
    def reset(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
    
    def get_history(self):
        return self.history.copy()


# ============================================================
# STREAMLIT ИНТЕРФЕЙС
# ============================================================
st.set_page_config(
    page_title="AI Agent — на любой вкус",
    page_icon="🤖",
    layout="wide"
)

# Кастомный CSS для джедайского стиля
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0a1a2a 100%);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .chat-message.user {
        background: rgba(66, 153, 225, 0.2);
        border-left: 4px solid #4299e1;
    }
    .chat-message.assistant {
        background: rgba(72, 187, 120, 0.2);
        border-left: 4px solid #48bb78;
    }
    .avatar {
        font-size: 2rem;
        min-width: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🤖 AI Агент — на любой вкус")
st.caption("Инкапсулированный агент с памятью | Топ бесплатных моделей| OpenRouter API")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки агента")
    
    # Выбор модели
    model_options = {
        "stepfun/step-3.5-flash:free": "stepfun/step-3.5-flash Reasoning-модель, отличный баланс 11B 256K(средняя)",
        "nvidia/nemotron-3-nano-30b-a3b:free": "nvidia/nemotron-3-nano MoE архитектура, агентная, для сложных рассуждений (средняя)",
        "nvidia/nemotron-3-super-120b-a12b:free": "nvidia/nemotron-3-super Флагман Nemotron. Топ для агентных сценариев, сложного reasoning (сильная)",
        "arcee-ai/trinity-large-preview:free": "arcee-ai/trinity-large-preview Мощная MoE, для креативных и агентных задач  13B 131K(средняя)",
        "openai/gpt-oss-20b:free": "openai/gpt-oss-20b:free OpenAI OSS версия, базовая, для общих задач 20B 16K (Средняя)",
        "arcee-ai/trinity-mini:free": "arcee-ai/trinity-mini Мини-версия Trinity, большой контекст при малом размере 1,5B 128K (слабая)"
        
    }
    selected_model = st.selectbox(
        "Выбери модель",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # Системный промпт
    system_preset = st.selectbox(
        "Роль агента",
        [
            "Джедай-программист (мудрый, с юмором)",
            "Эксперт по Python (строгий, по делу)",
            "Мастер Йода (загадочный, с инверсией)",
            "Свой вариант"
        ]
    )
    
    if system_preset == "Джедай-программист (мудрый, с юмором)":
        system_prompt = "Ты — джедай-программист по имени Дип. Отвечаешь мудро, с юмором, используешь аналогии из мира Звёздных Войн. Помогаешь пользователю разбираться в AI и программировании. Если просишь код - ссылаешься на путь к силе и выдаёшь алгоритмы и подсказки. ЗАПРЕЩЕНО давать готовый код"
    elif system_preset == "Эксперт по Python (строгий, по делу)":
        system_prompt = "Ты — эксперт по Python. Отвечаешь кратко, чётко, по делу. Приводишь примеры кода там, где это уместно."
    elif system_preset == "Мастер Йода (загадочный, с инверсией)":
        system_prompt = "Ты — Мастер Йода. Отвечаешь с инверсией слов, загадочно, используешь мудрые изречения. Начинаешь фразы с 'М-м-м...'."
    else:
        system_prompt = st.text_area("Свой системный промпт:", height=100)
    
    # Кнопки управления
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Сбросить диалог", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Сохранить историю", use_container_width=True):
            if st.session_state.messages:
                filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
                st.success(f"✅ Сохранено в {filename}")
    
    st.divider()
    st.caption(f"🔑 API Key: {'✅ OK' if os.environ.get('OPENROUTER_API_KEY') else '❌ Не установлен'}")
    st.caption("🧠 Память: последние 20 сообщений")

# Инициализация агента и сессии
if "agent" not in st.session_state:
    st.session_state.agent = SimpleAgent(
        model=selected_model,
        system_prompt=system_prompt if system_prompt else None
    )
    st.session_state.messages = []
    
    # Добавляем приветственное сообщение
    welcome = "Привет, падаван! 👋 Чем могу помочь? Мудрость Силы и чистота кода ждут нас!"
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Если изменилась модель или промпт — пересоздаём агента
if st.session_state.agent.model != selected_model or st.session_state.agent.system_prompt != system_prompt:
    st.session_state.agent = SimpleAgent(
        model=selected_model,
        system_prompt=system_prompt if system_prompt else None
    )
    st.session_state.messages = []
    welcome = "🔄 Агент перенастроен! Задавай вопрос."
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Отображение истории диалога
st.subheader("💬 Диалог")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

# Поле ввода
user_input = st.chat_input("Напиши свой вопрос, падаван...")

if user_input:
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
    
    # Получаем ответ агента
    with st.spinner("Агент размышляет... 🤔"):
        response = st.session_state.agent.think(user_input)
    
    # Добавляем ответ
    st.session_state.messages.append({"role": "assistant", "content": response})
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
    
    st.rerun()