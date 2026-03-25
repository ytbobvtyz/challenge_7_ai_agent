# day7_agent_persistent.py
# Установка: pip install streamlit openai python-dotenv

import os
import json
import glob
import hashlib
from datetime import datetime
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# КЛАСС АГЕНТА С ПОСТОЯННОЙ ПАМЯТЬЮ
# ============================================================
class PersistentAgent:
    def __init__(self, model="stepfun/step-3.5-flash:free", system_prompt=None, 
                 max_history=20, session_id=None, persist_dir="chat_history"):
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.persist_dir = persist_dir
        self.session_id = session_id or self._generate_session_id()
        self.history = []
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://your-domain.ru"),
                "X-Title": "AI Challenge Day 7 - Persistent Agent"
            }
        )
        
        self._load_history()
        
        if not self.history and self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
            self._save_history()
    
    def _generate_session_id(self):
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    
    def _get_history_path(self):
        return os.path.join(self.persist_dir, f"session_{self.session_id}.json")
    
    def _save_history(self):
        try:
            # Получаем первые 3 слова первого сообщения пользователя
            first_user_msg = ""
            for msg in self.history:
                if msg.get("role") == "user":
                    words = msg["content"].split()[:3]
                    first_user_msg = " ".join(words)
                    break
            
            history_to_save = {
                "session_id": self.session_id,
                "model": self.model,
                "system_prompt": self.system_prompt,
                "messages": self.history,
                "last_updated": datetime.now().isoformat(),
                "message_count": len(self.history),
                "preview": first_user_msg[:50] if first_user_msg else "Новый диалог"
            }
            with open(self._get_history_path(), 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения истории: {e}")
            return False
    
    def _load_history(self):
        history_path = self._get_history_path()
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Загружаем историю даже если модель/промпт не совпадают
                    # (но предупредим об этом позже в интерфейсе)
                    self.history = data.get("messages", [])
                    return True
            except Exception as e:
                print(f"❌ Ошибка загрузки истории: {e}")
                self.history = []
                return False
        else:
            self.history = []
            return False
    
    def load_session_by_id(self, session_id):
        """Загружает конкретную сессию по ID"""
        old_session_id = self.session_id
        self.session_id = session_id
        success = self._load_history()
        
        if not success or not self.history:
            # Если не загрузилось — возвращаем старую сессию
            self.session_id = old_session_id
            self._load_history()
            return False
        
        # Сохраняем, чтобы обновить preview
        self._save_history()
        return True
    
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
                max_tokens=2000
            )
            agent_response = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": agent_response})
            self._save_history()
            return agent_response
        except Exception as e:
            return f"❌ Ошибка API: {str(e)}"
    
    def _trim_history(self):
        if len(self.history) > self.max_history:
            if self.system_prompt and self.history and self.history[0]["role"] == "system":
                system_msg = self.history[0]
                self.history = [system_msg] + self.history[-(self.max_history - 1):]
            else:
                self.history = self.history[-self.max_history:]
    
    def reset(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
        self._save_history()
        try:
            os.remove(self._get_history_path())
        except:
            pass
    
    def get_history(self):
        return self.history.copy()
    
    def get_session_info(self):
        return {
            "session_id": self.session_id,
            "message_count": len(self.history),
            "history_file": self._get_history_path(),
            "model": self.model
        }


# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С СЕССИЯМИ
# ============================================================
def get_all_sessions(persist_dir="chat_history"):
    """Возвращает список всех сессий с метаданными"""
    sessions = []
    if os.path.exists(persist_dir):
        for f in glob.glob(os.path.join(persist_dir, "session_*.json")):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    session_id = data.get("session_id", os.path.basename(f).replace("session_", "").replace(".json", ""))
                    
                    # Получаем preview
                    preview = data.get("preview", "")
                    if not preview:
                        for msg in data.get("messages", []):
                            if msg.get("role") == "user":
                                words = msg["content"].split()[:3]
                                preview = " ".join(words)
                                break
                        if not preview:
                            preview = "Новый диалог"
                    
                    last_updated = data.get("last_updated", "")
                    if last_updated:
                        try:
                            dt = datetime.fromisoformat(last_updated)
                            date_str = dt.strftime("%d.%m.%Y %H:%M")
                        except:
                            date_str = last_updated[:16]
                    else:
                        date_str = "неизвестно"
                    
                    sessions.append({
                        "id": session_id,
                        "name": f"{preview[:40]} — {date_str}",
                        "message_count": data.get("message_count", 0),
                        "file": f,
                        "messages": data.get("messages", [])
                    })
            except Exception as e:
                print(f"Ошибка чтения сессии {f}: {e}")
    return sorted(sessions, key=lambda x: x.get("file", ""), reverse=True)


# ============================================================
# STREAMLIT ИНТЕРФЕЙС
# ============================================================
st.set_page_config(
    page_title="AI Agent — Постоянная память",
    page_icon="🧠",
    layout="wide"
)


# Заголовок
st.title("🧠 AI Агент — Постоянная память")
st.caption("История диалога сохраняется в JSON | При перезапуске агент помнит всё | День 7")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    model_options = {
        "stepfun/step-3.5-flash:free": "Step 3.5 Flash (средняя, reasoning)",
        "nvidia/nemotron-3-nano-30b-a3b:free": "Nemotron 3 Nano (средняя, MoE)",
        "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super (сильная, агентная)"
    }
    selected_model = st.selectbox(
        "Модель",
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
        system_prompt = """Ты — джедай-программист по имени Дип. Отвечаешь мудро, с юмором, 
        используешь аналогии из мира Звёздных Войн. Помогаешь пользователю разбираться в AI и программировании. 
        Если просишь код - ссылаешься на путь к силе и выдаёшь алгоритмы и подсказки. ЗАПРЕЩЕНО давать готовый код."""
    elif system_preset == "Эксперт по Python (строгий, по делу)":
        system_prompt = "Ты — эксперт по Python. Отвечаешь кратко, чётко, по делу. Приводишь примеры кода там, где это уместно."
    elif system_preset == "Мастер Йода (загадочный, с инверсией)":
        system_prompt = "Ты — Мастер Йода. Отвечаешь с инверсией слов, загадочно, используешь мудрые изречения. Начинаешь фразы с 'М-м-м...'."
    else:
        system_prompt = st.text_area("Свой системный промпт:", height=100)
    
    st.divider()
    
    # Управление сессиями
    st.subheader("📁 История диалогов")
    
    # Получаем список всех сессий
    all_sessions = get_all_sessions()
    
    if all_sessions:
        # Создаём словарь для выбора
        session_options = {s["id"]: s["name"] for s in all_sessions}
        current_session_id = st.session_state.get("session_id", None)
        
        # Находим индекс текущей сессии
        current_index = 0
        for i, s in enumerate(all_sessions):
            if s["id"] == current_session_id:
                current_index = i
                break
        
        selected_session_id = st.selectbox(
            "Загрузить диалог",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=current_index,
            key="session_selector"
        )
        
        # Кнопка переключения сессии
        if st.button("🔄 Переключить сессию", use_container_width=True, key="switch_session_btn"):
            if selected_session_id != st.session_state.get("session_id"):
                with st.spinner("Загрузка диалога..."):
                    # Создаём нового агента с выбранной сессией
                    new_agent = PersistentAgent(
                        model=selected_model,
                        system_prompt=system_prompt if system_prompt else None,
                        session_id=selected_session_id
                    )
                    # Загружаем историю
                    new_agent._load_history()
                    
                    # Обновляем session_state
                    st.session_state.agent = new_agent
                    st.session_state.session_id = selected_session_id
                    st.session_state.messages = new_agent.get_history()
                    
                    st.success(f"✅ Загружен диалог: {session_options[selected_session_id]}")
                    st.rerun()
    else:
        st.info("Нет сохранённых диалогов. Начни новый разговор!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Сбросить диалог", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.session_id = st.session_state.agent.session_id
            st.success("✅ Диалог сброшен")
            st.rerun()
    with col2:
        if st.button("💾 Сохранить сейчас", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent._save_history()
                st.success("✅ Сохранено!")
    
    st.divider()
    st.caption(f"🔑 API Key: {'✅ OK' if os.environ.get('OPENROUTER_API_KEY') else '❌ Не установлен'}")
    st.caption("🧠 Постоянная память: JSON-файлы в chat_history/")
    st.caption("📝 Максимум токенов: 2000")

# Инициализация агента
if "agent" not in st.session_state:
    session_id = st.session_state.get("session_id", None)
    st.session_state.agent = PersistentAgent(
        model=selected_model,
        system_prompt=system_prompt if system_prompt else None,
        session_id=session_id
    )
    st.session_state.session_id = st.session_state.agent.session_id
    st.session_state.messages = st.session_state.agent.get_history()

# Если изменилась модель или промпт — пересоздаём агента (но сохраняем текущую сессию)
if (st.session_state.agent.model != selected_model or 
    st.session_state.agent.system_prompt != system_prompt):
    
    old_session_id = st.session_state.agent.session_id
    
    st.session_state.agent = PersistentAgent(
        model=selected_model,
        system_prompt=system_prompt if system_prompt else None,
        session_id=old_session_id
    )
    st.session_state.session_id = st.session_state.agent.session_id
    st.session_state.messages = st.session_state.agent.get_history()
    
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "🔄 Агент перенастроен! Продолжаем диалог."}]

# Отображение информации о сессии
session_info = st.session_state.agent.get_session_info()
with st.sidebar:
    with st.expander("ℹ️ Текущая сессия", expanded=False):
        st.write(f"**ID:** `{session_info['session_id']}`")
        st.write(f"**Сообщений:** {session_info['message_count']}")
        st.write(f"**Модель:** {session_info['model']}")

# Отображение истории диалога
st.subheader("💬 Диалог")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
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