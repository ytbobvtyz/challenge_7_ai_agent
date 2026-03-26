# day8_agent_token_counter.py (исправленная версия)
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
# НАСТРОЙКИ АУТЕНТИФИКАЦИИ
# ============================================================
AUTH_QUESTION = "Введите фамилию автора челленджа на русском:"
AUTH_ANSWER = "Гладков"

def check_auth():
    return st.session_state.get("authenticated", False)

def authenticate():
    st.markdown("""
    <style>
        .auth-container {
            max-width: 500px;
            margin: 100px auto;
            padding: 2rem;
            background: rgba(26, 42, 58, 0.8);
            border-radius: 1rem;
            border: 1px solid #4299e1;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<h2>🔐 Доступ ограничен</h2>', unsafe_allow_html=True)
    
    user_answer = st.text_input(AUTH_QUESTION, type="password", key="auth_input")
    
    if st.button("🚪 Войти"):
        if user_answer.strip() == AUTH_ANSWER:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Неверный ответ")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return False


# ============================================================
# КЛАСС АГЕНТА
# ============================================================
class TokenAwareAgent:
    def __init__(self, model="stepfun/step-3.5-flash:free", system_prompt=None, 
                 max_history=20, session_id=None, persist_dir="chat_history",
                 model_max_tokens=256000):
        self.model = model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.model_max_tokens = model_max_tokens
        self.persist_dir = persist_dir
        self.session_id = session_id or self._generate_session_id()
        self.history = []
        
        # Единое хранилище статистики токенов
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None  # Храним последний usage для отображения
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://your-domain.ru"),
                "X-Title": "AI Challenge Day 8 - Token Counter"
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
    
    def get_token_stats(self):
        """Возвращает актуальную статистику токенов"""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        usage_percent = (total_tokens / self.model_max_tokens) * 100 if self.model_max_tokens else 0
        usage_percent = min(usage_percent, 100)
        
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "max_tokens": self.model_max_tokens,
            "usage_percent": round(usage_percent, 1),
            "is_critical": usage_percent > 90,
            "is_warning": usage_percent > 70
        }
    
    def _save_history(self):
        try:
            first_user_msg = ""
            for msg in self.history:
                if msg.get("role") == "user":
                    words = msg["content"].split()[:3]
                    first_user_msg = " ".join(words)
                    break
            
            token_stats = self.get_token_stats()
            
            history_to_save = {
                "session_id": self.session_id,
                "model": self.model,
                "system_prompt": self.system_prompt,
                "messages": self.history,
                "last_updated": datetime.now().isoformat(),
                "message_count": len(self.history),
                "preview": first_user_msg[:50] if first_user_msg else "Новый диалог",
                "token_stats": token_stats,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens
            }
            with open(self._get_history_path(), 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def _load_history(self):
        history_path = self._get_history_path()
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get("messages", [])
                    self.total_prompt_tokens = data.get("total_prompt_tokens", 0)
                    self.total_completion_tokens = data.get("total_completion_tokens", 0)
                    return True
            except Exception as e:
                print(f"❌ Ошибка загрузки: {e}")
                self.history = []
                return False
        else:
            self.history = []
            return False
    
    def delete_session(self, session_id):
        """Удаляет файл сессии"""
        file_path = os.path.join(self.persist_dir, f"session_{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def estimate_tokens(self, text):
        """Грубая оценка токенов для предупреждения"""
        return len(text) // 4
    
    def think(self, user_input):
        if not user_input or not user_input.strip():
            return "❌ Пустой запрос. Напиши что-нибудь!", None
        
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
            
            # Получаем данные из API
            api_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Обновляем общую статистику ТОЛЬКО здесь, один раз
            self.total_prompt_tokens = api_usage["prompt_tokens"]
            self.total_completion_tokens = api_usage["completion_tokens"]
            self.last_usage = api_usage
            
            self.history.append({"role": "assistant", "content": agent_response})
            self._save_history()
            
            return agent_response, api_usage
            
        except Exception as e:
            return f"❌ Ошибка API: {str(e)}", None
    
    def _trim_history(self):
        if len(self.history) > self.max_history:
            if self.system_prompt and self.history and self.history[0]["role"] == "system":
                system_msg = self.history[0]
                self.history = [system_msg] + self.history[-(self.max_history - 1):]
            else:
                self.history = self.history[-self.max_history:]
    
    def _trim_aggressive(self):
        """Агрессивное обрезание при переполнении"""
        if self.system_prompt and self.history and self.history[0]["role"] == "system":
            system_msg = self.history[0]
            self.history = [system_msg] + self.history[-5:]
        else:
            self.history = self.history[-5:]
        self._save_history()
    
    def reset(self):
        self.history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
        self._save_history()
    
    def get_history(self):
        return self.history.copy()
    
    def get_session_info(self):
        return {
            "session_id": self.session_id,
            "message_count": len(self.history),
            "model": self.model,
            "token_stats": self.get_token_stats()
        }


# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С СЕССИЯМИ
# ============================================================
def get_all_sessions(persist_dir="chat_history"):
    sessions = []
    if os.path.exists(persist_dir):
        for f in glob.glob(os.path.join(persist_dir, "session_*.json")):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    session_id = data.get("session_id", "")
                    
                    preview = data.get("preview", "Новый диалог")
                    last_updated = data.get("last_updated", "")
                    if last_updated:
                        try:
                            dt = datetime.fromisoformat(last_updated)
                            date_str = dt.strftime("%d.%m.%Y %H:%M")
                        except:
                            date_str = last_updated[:16]
                    else:
                        date_str = "неизвестно"
                    
                    token_stats = data.get("token_stats", {})
                    token_info = f"{token_stats.get('total_tokens', 0)} токенов" if token_stats else ""
                    
                    sessions.append({
                        "id": session_id,
                        "name": f"{preview[:40]} — {date_str}",
                        "token_info": token_info,
                        "message_count": data.get("message_count", 0),
                        "file": f
                    })
            except Exception as e:
                print(f"Ошибка чтения {f}: {e}")
    return sorted(sessions, key=lambda x: x.get("file", ""), reverse=True)


def delete_session_file(session_id, persist_dir="chat_history"):
    file_path = os.path.join(persist_dir, f"session_{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


# ============================================================
# STREAMLIT ИНТЕРФЕЙС
# ============================================================
st.set_page_config(
    page_title="AI Agent — Токен-менеджер",
    page_icon="📊",
    layout="wide"
)

if not check_auth():
    authenticate()
    st.stop()

st.title("📊 AI Агент — Токен-менеджер")
st.caption("Подсчёт токенов через API | День 8")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    model_options = {
        "stepfun/step-3.5-flash:free": "Step 3.5 Flash (256K токенов)",
        "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super (256K токенов)",
        "arcee-ai/trinity-mini:free": "Trinity-mini (128K токенов)",
        "z-ai/glm-4.5-air:free": "z-ai/glm-4.5-air:free (32K токенов)"
    }
    selected_model = st.selectbox(
        "Модель",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    model_limits = {
        "stepfun/step-3.5-flash:free": 256000,
        "nvidia/nemotron-3-super-120b-a12b:free": 256000,
        "arcee-ai/trinity-mini:free": 128000,
        "z-ai/glm-4.5-air:free": 32000
    }
    model_max_tokens = model_limits.get(selected_model, 256000)
    
    system_preset = st.selectbox(
        "Роль агента",
        [
            "Джедай-программист (мудрый, с юмором)",
            "Эксперт по Python (строгий, по делу)",
            "Мастер Йода (загадочный, с инверсией)"
        ]
    )
    
    if system_preset == "Джедай-программист (мудрый, с юмором)":
        system_prompt = "Ты — джедай-программист по имени Дип. Отвечаешь мудро, с юмором, используешь аналогии из мира Звёздных Войн."
    elif system_preset == "Эксперт по Python (строгий, по делу)":
        system_prompt = "Ты — эксперт по Python. Отвечаешь кратко, чётко, по делу."
    else:
        system_prompt = "Ты — Мастер Йода. Отвечаешь с инверсией слов, загадочно. Начинаешь фразы с 'М-м-м...'."
    
    st.divider()
    
    # Токен-статистика (только из агента, единый источник правды)
    if "agent" in st.session_state and st.session_state.agent:
        token_stats = st.session_state.agent.get_token_stats()
        
        st.subheader("📊 Токен-статистика")
        
        # Прогресс-бар - теперь точно будет отображаться
        progress_value = token_stats["usage_percent"] / 100
        st.progress(progress_value, text=f"{token_stats['usage_percent']}% использовано")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📥 Токенов запроса", f"{token_stats['prompt_tokens']:,}")
        with col2:
            st.metric("📤 Токенов ответа", f"{token_stats['completion_tokens']:,}")
        
        st.metric("📚 Всего токенов", f"{token_stats['total_tokens']:,}")
        st.caption(f"🎯 Лимит модели: {token_stats['max_tokens']:,} токенов")
        
        if token_stats["is_critical"]:
            st.warning("⚠️ **КРИТИЧЕСКИЙ УРОВЕНЬ!** Контекст почти заполнен.")
        elif token_stats["is_warning"]:
            st.info("⚠️ Контекст заполнен более чем на 70%.")
        
        # Отображение последнего запроса
        if st.session_state.agent.last_usage:
            st.divider()
            st.caption("📊 Последний запрос:")
            last = st.session_state.agent.last_usage
            st.caption(f"• Запрос: {last['prompt_tokens']:,} токенов")
            st.caption(f"• Ответ: {last['completion_tokens']:,} токенов")
            st.caption(f"• Всего: {last['total_tokens']:,} токенов")
    
    st.divider()
    
    # История диалогов с кнопками удаления
    st.subheader("📁 История диалогов")
    
    all_sessions = get_all_sessions()
    
    if all_sessions:
        for sess in all_sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{sess['name']}**")
                st.caption(f"{sess['message_count']} сообщений | {sess['token_info']}")
            with col2:
                if st.button("🗑️", key=f"del_{sess['id']}", help="Удалить этот диалог"):
                    if delete_session_file(sess['id']):
                        st.success(f"✅ Диалог удалён")
                        if "agent" in st.session_state and st.session_state.agent.session_id == sess['id']:
                            st.session_state.agent = None
                        st.rerun()
            st.divider()
        
        session_options = {s["id"]: s["name"] for s in all_sessions}
        current_session_id = st.session_state.get("session_id", None)
        
        selected_session_id = st.selectbox(
            "Загрузить диалог",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=0 if current_session_id in session_options else 0
        )
        
        if st.button("🔄 Переключить сессию", use_container_width=True):
            if selected_session_id != st.session_state.get("session_id"):
                with st.spinner("Загрузка диалога..."):
                    new_agent = TokenAwareAgent(
                        model=selected_model,
                        system_prompt=system_prompt,
                        session_id=selected_session_id,
                        model_max_tokens=model_max_tokens
                    )
                    new_agent._load_history()
                    
                    st.session_state.agent = new_agent
                    st.session_state.session_id = selected_session_id
                    st.rerun()
    else:
        st.info("Нет сохранённых диалогов")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Сбросить диалог", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent.reset()
            st.rerun()
    with col2:
        if st.button("✂️ Обрезать историю", use_container_width=True, help="Оставляет системный промпт + 5 последних сообщений"):
            if "agent" in st.session_state:
                agent = st.session_state.agent
                if agent.system_prompt:
                    system_msg = {"role": "system", "content": agent.system_prompt}
                    agent.history = [system_msg] + agent.history[-5:]
                else:
                    agent.history = agent.history[-5:]
                agent._save_history()
                st.success("✅ История обрезана")
                st.rerun()
    
    st.divider()
    st.caption("📊 Токены считаются из ответа API")
    st.caption("🧠 Точность: 100%")

# Инициализация агента
if "agent" not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = TokenAwareAgent(
        model=selected_model,
        system_prompt=system_prompt,
        session_id=st.session_state.get("session_id", None),
        model_max_tokens=model_max_tokens
    )
    st.session_state.session_id = st.session_state.agent.session_id

# Если изменилась модель
if st.session_state.agent.model != selected_model:
    old_session_id = st.session_state.agent.session_id
    old_history = st.session_state.agent.history
    old_total_prompt = st.session_state.agent.total_prompt_tokens
    old_total_completion = st.session_state.agent.total_completion_tokens
    
    st.session_state.agent = TokenAwareAgent(
        model=selected_model,
        system_prompt=system_prompt,
        session_id=old_session_id,
        model_max_tokens=model_max_tokens
    )
    # Сохраняем старую статистику
    st.session_state.agent.history = old_history
    st.session_state.agent.total_prompt_tokens = old_total_prompt
    st.session_state.agent.total_completion_tokens = old_total_completion
    st.session_state.agent.model_max_tokens = model_max_tokens
    st.session_state.session_id = st.session_state.agent.session_id

# Отображение диалога
st.subheader("💬 Диалог")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.agent.get_history():
        if msg["role"] == "system":
            continue
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

# Поле ввода
user_input = st.chat_input("Напиши свой вопрос...")

if user_input:
    with st.spinner("Агент размышляет... 🤔"):
        response, token_usage = st.session_state.agent.think(user_input)
    
    # Обновляем интерфейс
    st.rerun()