# day9_agent_context_compression.py
# Установка: pip install streamlit openai python-dotenv

import os
import json
import glob
import hashlib
import time
from datetime import datetime
from collections import deque
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
# КЛАСС АГЕНТА С КОМПРЕССИЕЙ КОНТЕКСТА
# ============================================================
class ContextCompressionAgent:
    def __init__(self, model="stepfun/step-3.5-flash:free", system_prompt=None, 
                 session_id=None, persist_dir="chat_history",
                 model_max_tokens=256000, window_size=6, compress_after=8):
        
        self.model = model
        self.system_prompt = system_prompt
        self.model_max_tokens = model_max_tokens
        self.persist_dir = persist_dir
        self.session_id = session_id or self._generate_session_id()
        
        # Настройки компрессии
        self.window_size = window_size  # Сколько последних сообщений храним без сжатия
        self.compress_after = compress_after  # Через сколько сообщений сжимаем
        
        # Хранилище
        self.full_history = []  # Полная история (для сохранения и отладки)
        self.recent_messages = []  # Последние сообщения без сжатия
        self.summaries = []  # Список summary старых частей
        
        # Статистика токенов
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None
        
        # Статистика компрессии
        self.compression_stats = {
            "compressions_count": 0,
            "total_tokens_saved": 0,
            "original_tokens": 0,
            "compressed_tokens": 0
        }
        
        # Защита от rate limit
        self.request_timestamps = deque(maxlen=10)
        self.min_interval = 2
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://your-domain.ru"),
                "X-Title": "AI Challenge Day 9 - Context Compression"
            }
        )
        
        self._load_history()
        
        # Инициализируем с системным промптом
        if not self.full_history and self.system_prompt:
            system_msg = {"role": "system", "content": self.system_prompt}
            self.full_history.append(system_msg)
            self._save_history()
    
    def _generate_session_id(self):
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    
    def _get_history_path(self):
        return os.path.join(self.persist_dir, f"session_{self.session_id}.json")
    
    def _estimate_tokens(self, messages):
        """Грубая оценка токенов для проверки порогов"""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if content is not None:  # Защита от None
                total_chars += len(content)
        return total_chars // 4

    def _clean_messages(self, messages):
        """Очищает список сообщений от None и пустых строк"""
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if content and isinstance(content, str) and content.strip():
                    cleaned.append(msg)
        return cleaned
    
    def _check_rate_limit(self):
        """Проверяет, не слишком ли часто мы шлем запросы"""
        now = time.time()
        
        while self.request_timestamps and now - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        if len(self.request_timestamps) >= 5:
            wait_time = 60 - (now - self.request_timestamps[0])
            if wait_time > 0:
                return False, f"Слишком много запросов! Подожди {wait_time:.1f} секунд."
        
        if self.request_timestamps and now - self.request_timestamps[-1] < self.min_interval:
            return False, f"Слишком быстро! Подожди {self.min_interval} секунды."
        
        return True, "OK"
    
    def _create_summary(self, messages_to_compress):
        """Создает summary с увеличенным max_tokens"""
        
        formatted_messages = []
        for msg in messages_to_compress:
            role = "👤 Пользователь" if msg["role"] == "user" else "🤖 Агент"
            content = msg.get("content", "")
            if content and isinstance(content, str) and content.strip():
                # Обрезаем слишком длинные сообщения для промпта
                if len(content) > 800:
                    content = content[:800] + "..."
                formatted_messages.append(f"{role}: {content}")
        
        if not formatted_messages:
            return "[Нет сообщений для сжатия]"
        
        dialog_text = "\n".join(formatted_messages)
        
        # Улучшенный промпт с четкой структурой
        summary_prompt = f"""Создай КРАТКУЮ выжимку этого диалога в СТРУКТУРИРОВАННОМ виде.

    ПРАВИЛА:
    1. Используй маркированный список
    2. Сохрани ключевые факты, код, решения
    3. Не теряй последовательность
    4. Объем: до 500 символов

    ДИАЛОГ:
    {dialog_text}

    ВЫЖИМКА:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=500  # Увеличил с 300 до 500
            )
            
            summary = response.choices[0].message.content
            
            if not summary or len(summary) < 10:
                return self._create_fallback_summary(messages_to_compress)
            
            # Учитываем токены
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            
            return summary.strip()
            
        except Exception as e:
            print(f"❌ Ошибка суммаризации: {e}")
            return self._create_fallback_summary(messages_to_compress)

    def _create_fallback_summary(self, messages):
        """Качественный fallback summary"""
        # Извлекаем ключевые темы
        topics = []
        for msg in messages[-3:]:  # Берем последние 3 сообщения
            content = msg.get("content", "")
            if content:
                # Простой экстракт ключевых слов
                words = content.split()[:5]
                if words:
                    topics.extend(words[:2])
        
        unique_topics = list(set(topics))[:3]
        if unique_topics:
            return f"[Диалог о {', '.join(unique_topics)}...]"
        return f"[Диалог из {len(messages)} сообщений]"
 
    def _compress_old_messages(self):
        """Сжимает старые сообщения, которые выходят за окно"""
        # задержка для повышения вероятности успешной компрессии.
        time.sleep(4)
        # Очищаем recent_messages от пустых сообщений
        self.recent_messages = self._clean_messages(self.recent_messages)
        
        if len(self.recent_messages) <= self.window_size:
            return
        
        # Сообщения для сжатия (все кроме последних window_size)
        to_compress = self.recent_messages[:-self.window_size]
        
        # Фильтруем пустые сообщения
        to_compress = [msg for msg in to_compress if msg.get("content")]
        
        if not to_compress:
            # Если нет нормальных сообщений, просто обрезаем
            self.recent_messages = self.recent_messages[-self.window_size:]
            return
        
        # Считаем токены до сжатия
        original_tokens = self._estimate_tokens(to_compress)
        
        print(f"\n{'='*50}")
        print(f"🗜️ ЗАПУСК КОМПРЕССИИ")
        print(f"   Сообщений для сжатия: {len(to_compress)}")
        print(f"   Токенов до сжатия: {original_tokens}")
        print(f"{'='*50}")
        
        # Создаем summary с retry
        summary = self._create_summary(to_compress)
        
        # Считаем токены после сжатия
        compressed_tokens = self._estimate_tokens([{"content": summary}])
        
        # Сохраняем summary
        self.summaries.append({
            "text": summary,
            "timestamp": datetime.now().isoformat(),
            "original_messages": len(to_compress),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens
        })
        
        # Обновляем статистику компрессии
        self.compression_stats["compressions_count"] += 1
        self.compression_stats["total_tokens_saved"] += (original_tokens - compressed_tokens)
        self.compression_stats["original_tokens"] += original_tokens
        self.compression_stats["compressed_tokens"] += compressed_tokens
        
        # Обновляем recent messages (оставляем только последние)
        self.recent_messages = self.recent_messages[-self.window_size:]
        
        print(f"✅ КОМПРЕССИЯ ЗАВЕРШЕНА")
        print(f"   Токены: {original_tokens} → {compressed_tokens}")
        print(f"   Экономия: {original_tokens - compressed_tokens} токенов")
        print(f"   Сжатие: {(1 - compressed_tokens/original_tokens)*100:.1f}%")
        print(f"{'='*50}\n")


    def _build_context(self):
        """Собирает контекст для отправки в API"""
        context = []
        
        # 1. Системный промпт
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        
        # 2. Все summary (объединяем в один блок)
        if self.summaries:
            combined_summaries = []
            for i, summary_data in enumerate(self.summaries, 1):
                summary_text = summary_data.get('text', '')
                # Фильтруем фолбэки и обрезанные summary
                if summary_text and summary_text != "None" and not summary_text.startswith("[Диалог из"):
                    # Обрезаем слишком длинные summary для контекста
                    if len(summary_text) > 500:
                        summary_text = summary_text[:500] + "..."
                    combined_summaries.append(f"[Часть {i}]\n{summary_text}")
            
            if combined_summaries:
                summary_text = "\n\n".join(combined_summaries)
                context.append({
                    "role": "system",
                    "content": f"📚 ИСТОРИЯ ДИАЛОГА (сжатая):\n\n{summary_text}"
                })
        
        # 3. Последние сообщения (без сжатия)
        clean_messages = []
        for msg in self.recent_messages[-self.window_size:]:  # Берем только последние window_size
            content = msg.get("content")
            if content and isinstance(content, str) and content.strip():
                clean_messages.append(msg)
        
        context.extend(clean_messages)
        
        # Отладка
        print(f"\n📊 КОНТЕКСТ ДЛЯ API:")
        print(f"   - System prompt: {len(self.system_prompt) if self.system_prompt else 0} символов")
        print(f"   - Summaries: {len(combined_summaries)} блоков" if self.summaries else "   - Summaries: нет")
        print(f"   - Recent messages: {len(clean_messages)} сообщений")
        print(f"   - Всего сообщений в контексте: {len(context)}")
        
        return context


    def get_token_stats(self):
        """Возвращает актуальную статистику токенов"""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        usage_percent = (total_tokens / self.model_max_tokens) * 100 if self.model_max_tokens else 0
        usage_percent = min(usage_percent, 100)
        
        # Добавляем статистику компрессии
        compression_info = {
            "compressions": self.compression_stats["compressions_count"],
            "tokens_saved": self.compression_stats["total_tokens_saved"],
            "efficiency": 0
        }
        
        if self.compression_stats["original_tokens"] > 0:
            compression_info["efficiency"] = round(
                (1 - self.compression_stats["compressed_tokens"] / self.compression_stats["original_tokens"]) * 100,
                1
            )
        
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "max_tokens": self.model_max_tokens,
            "usage_percent": round(usage_percent, 1),
            "is_critical": usage_percent > 90,
            "is_warning": usage_percent > 70,
            "compression": compression_info
        }
    
    def _save_history(self):
        """Сохраняет всю историю (включая summaries)"""
        try:
            first_user_msg = ""
            for msg in self.full_history:
                if msg.get("role") == "user":
                    words = msg["content"].split()[:3]
                    first_user_msg = " ".join(words)
                    break
            
            token_stats = self.get_token_stats()
            
            history_to_save = {
                "session_id": self.session_id,
                "model": self.model,
                "system_prompt": self.system_prompt,
                "full_history": self.full_history,
                "recent_messages": self.recent_messages,
                "summaries": self.summaries,
                "last_updated": datetime.now().isoformat(),
                "message_count": len(self.full_history),
                "preview": first_user_msg[:50] if first_user_msg else "Новый диалог",
                "token_stats": token_stats,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "compression_stats": self.compression_stats
            }
            with open(self._get_history_path(), 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def _load_history(self):
        """Загружает историю из файла"""
        history_path = self._get_history_path()
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.full_history = data.get("full_history", [])
                    self.recent_messages = data.get("recent_messages", [])
                    self.summaries = data.get("summaries", [])
                    self.total_prompt_tokens = data.get("total_prompt_tokens", 0)
                    self.total_completion_tokens = data.get("total_completion_tokens", 0)
                    self.compression_stats = data.get("compression_stats", {
                        "compressions_count": 0,
                        "total_tokens_saved": 0,
                        "original_tokens": 0,
                        "compressed_tokens": 0
                    })
                    return True
            except Exception as e:
                print(f"❌ Ошибка загрузки: {e}")
                return False
        return False
    
    def think(self, user_input):
        """Основной метод для отправки сообщения"""
        if not user_input or not user_input.strip():
            return "❌ Пустой запрос. Напиши что-нибудь!", None
        
        # Проверяем rate limit
        ok, message = self._check_rate_limit()
        if not ok:
            return f"⏸️ {message}", None
        
        # Добавляем сообщение пользователя
        user_msg = {"role": "user", "content": user_input}
        self.full_history.append(user_msg)
        self.recent_messages.append(user_msg)
        
        # Проверяем, нужно ли сжать
        if len(self.recent_messages) > self.compress_after:
            self._compress_old_messages()
        
        # Строим контекст для API
        context = self._build_context()
        
        # Добавляем метку времени
        self.request_timestamps.append(time.time())
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context,
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
            
            # Обновляем статистику
            self.total_prompt_tokens += api_usage["prompt_tokens"]
            self.total_completion_tokens += api_usage["completion_tokens"]
            self.last_usage = api_usage
            
            # Добавляем ответ
            assistant_msg = {"role": "assistant", "content": agent_response}
            self.full_history.append(assistant_msg)
            self.recent_messages.append(assistant_msg)
            
            self._save_history()
            
            return agent_response, api_usage
            
        except Exception as e:
            return f"❌ Ошибка API: {str(e)}", None
    
    def reset(self):
        """Сбрасывает диалог"""
        self.full_history = []
        self.recent_messages = []
        self.summaries = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None
        self.compression_stats = {
            "compressions_count": 0,
            "total_tokens_saved": 0,
            "original_tokens": 0,
            "compressed_tokens": 0
        }
        
        if self.system_prompt:
            system_msg = {"role": "system", "content": self.system_prompt}
            self.full_history.append(system_msg)
        
        self._save_history()
    
    def get_history(self):
        """Возвращает полную историю для отображения"""
        return self.full_history.copy()
    
    def get_compression_info(self):
        """Возвращает информацию о компрессии"""
        return self.compression_stats


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
                    token_info = f"{token_stats.get('total_tokens', 0)} токенов"
                    
                    # Добавляем информацию о компрессии
                    compression_stats = data.get("compression_stats", {})
                    comp_info = f" | 🗜️ {compression_stats.get('compressions_count', 0)} сжатий"
                    
                    sessions.append({
                        "id": session_id,
                        "name": f"{preview[:40]} — {date_str}",
                        "token_info": token_info + comp_info,
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
    page_title="AI Agent — Контекст-менеджер",
    page_icon="🗜️",
    layout="wide"
)

if not check_auth():
    authenticate()
    st.stop()

st.title("🗜️ AI Агент — Управление контекстом с компрессией")
st.caption("День 9: Сжатие истории для экономии токенов")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    model_options = {
        "stepfun/step-3.5-flash:free": "Step 3.5 Flash (256K токенов)",
        "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super (256K токенов)",
        "arcee-ai/trinity-mini:free": "Trinity-mini (128K токенов)",
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
    
    # Настройки компрессии
    st.divider()
    st.subheader("🗜️ Настройки компрессии")
    window_size = st.slider("Размер окна (сообщений без сжатия)", 4, 12, 6)
    compress_after = st.slider("Сжимать после (сообщений)", 6, 20, 8)
    
    st.divider()
    
    # Статистика
    if "agent" in st.session_state and st.session_state.agent:
        token_stats = st.session_state.agent.get_token_stats()
        
        st.subheader("📊 Токен-статистика")
        progress_value = token_stats["usage_percent"] / 100
        st.progress(progress_value, text=f"{token_stats['usage_percent']}% использовано")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📥 Токенов запроса", f"{token_stats['prompt_tokens']:,}")
        with col2:
            st.metric("📤 Токенов ответа", f"{token_stats['completion_tokens']:,}")
        
        st.metric("📚 Всего токенов", f"{token_stats['total_tokens']:,}")
        
        # Статистика компрессии
        if token_stats.get("compression"):
            comp = token_stats["compression"]
            st.divider()
            st.subheader("🗜️ Эффективность компрессии")
            st.metric("Количество сжатий", comp["compressions"])
            st.metric("Сэкономлено токенов", f"{comp['tokens_saved']:,}")
            st.metric("Эффективность", f"{comp['efficiency']}%")
    
    st.divider()
    
    # История диалогов
    st.subheader("📁 История диалогов")
    all_sessions = get_all_sessions()
    
    if all_sessions:
        for sess in all_sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{sess['name']}**")
                st.caption(f"{sess['message_count']} сообщений | {sess['token_info']}")
            with col2:
                if st.button("🗑️", key=f"del_{sess['id']}"):
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
                    new_agent = ContextCompressionAgent(
                        model=selected_model,
                        system_prompt=system_prompt,
                        session_id=selected_session_id,
                        model_max_tokens=model_max_tokens,
                        window_size=window_size,
                        compress_after=compress_after
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
        if st.button("🔄 Принудительно сжать", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent._compress_old_messages()
                st.session_state.agent._save_history()
                st.success("✅ Принудительное сжатие выполнено")
                st.rerun()
    
    st.divider()
    st.caption("🗜️ Старые сообщения сжимаются в summary")
    st.caption(f"📦 Окно: {window_size} сообщений без сжатия")
    st.caption(f"⚙️ Сжатие каждые {compress_after} сообщений")

# Инициализация агента
if "agent" not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = ContextCompressionAgent(
        model=selected_model,
        system_prompt=system_prompt,
        session_id=st.session_state.get("session_id", None),
        model_max_tokens=model_max_tokens,
        window_size=window_size,
        compress_after=compress_after
    )
    st.session_state.session_id = st.session_state.agent.session_id

# Если изменилась модель
if st.session_state.agent.model != selected_model:
    old_session_id = st.session_state.agent.session_id
    st.session_state.agent = ContextCompressionAgent(
        model=selected_model,
        system_prompt=system_prompt,
        session_id=old_session_id,
        model_max_tokens=model_max_tokens,
        window_size=window_size,
        compress_after=compress_after
    )
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