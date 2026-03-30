# day10_agent_context_strategies.py
# Установка: pip install streamlit openai python-dotenv

import os
import json
import glob
import hashlib
import time
from datetime import datetime
from collections import deque
from abc import ABC, abstractmethod
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
# СТРАТЕГИИ ПАМЯТИ
# ============================================================

class MemoryStrategy(ABC):
    @abstractmethod
    def add_message(self, message):
        pass
    
    @abstractmethod
    def get_context(self, system_prompt):
        pass
    
    @abstractmethod
    def get_stats(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class SlidingWindowMemory(MemoryStrategy):
    """Стратегия 1: Скользящее окно — храним только последние N сообщений"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size)
        self.total_messages = 0
    
    def add_message(self, message):
        self.messages.append(message)
        self.total_messages += 1
    
    def get_context(self, system_prompt=None):
        context = []
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})
        context.extend(list(self.messages))
        return context
    
    def get_stats(self):
        return {
            "type": "Sliding Window",
            "window_size": self.window_size,
            "current_messages": len(self.messages),
            "total_messages": self.total_messages
        }
    
    def reset(self):
        self.messages.clear()
        self.total_messages = 0


class FactMemory(MemoryStrategy):
    """Стратегия 2: Key-Value Memory — факты + последние сообщения"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size)
        self.facts = {}
        self.fact_history = []
        self.total_messages = 0
    
    def add_message(self, message):
        self.messages.append(message)
        self.total_messages += 1
        
        if message["role"] == "assistant" and "<<<FACTS>>>" in message["content"]:
            self._extract_facts(message["content"])
    
    def _extract_facts(self, content):
        """Извлекает факты из ответа модели"""
        try:
            if "<<<FACTS>>>" in content:
                parts = content.split("<<<FACTS>>>")
                facts_part = parts[1].split("<<<END_FACTS>>>")[0]
                
                for line in facts_part.strip().split('\n'):
                    line = line.strip().lstrip('- ')
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key in self.facts and self.facts[key] != value:
                            self.fact_history.append({
                                "key": key,
                                "old": self.facts[key],
                                "new": value,
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        self.facts[key] = value
        except Exception as e:
            print(f"⚠️ Ошибка парсинга фактов: {e}")
    
    def get_context(self, system_prompt=None):
        context = []
        
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})
        
        if self.facts:
            facts_text = "\n".join([f"- {k}: {v}" for k, v in self.facts.items()])
            context.append({
                "role": "system",
                "content": f"📌 ИЗВЕСТНЫЕ ФАКТЫ ИЗ ДИАЛОГА:\n{facts_text}"
            })
        
        context.extend(list(self.messages))
        
        return context
    
    def get_stats(self):
        return {
            "type": "Key-Value Memory (Facts)",
            "window_size": self.window_size,
            "current_messages": len(self.messages),
            "total_messages": self.total_messages,
            "facts_count": len(self.facts),
            "fact_changes": len(self.fact_history)
        }
    
    def reset(self):
        self.messages.clear()
        self.facts.clear()
        self.fact_history.clear()
        self.total_messages = 0


class BranchingMemory(MemoryStrategy):
    """Стратегия 3: Branching — ветки диалога с чекапоинтами"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.branches = {"main": deque(maxlen=window_size)}
        self.active_branch = "main"
        self.checkpoints = []
        self.total_messages = 0
    
    def add_message(self, message):
        if self.active_branch not in self.branches:
            self.branches[self.active_branch] = deque(maxlen=self.window_size)
        self.branches[self.active_branch].append(message)
        self.total_messages += 1
    
    def get_messages(self):
        """Возвращает сообщения активной ветки для отображения"""
        if self.active_branch in self.branches:
            return list(self.branches[self.active_branch])
        return []
    
    def get_context(self, system_prompt=None):
        context = []
        if system_prompt:
            context.append({"role": "system", "content": system_prompt})
        
        context.append({
            "role": "system",
            "content": f"🌿 Текущая ветка: {self.active_branch}"
        })
        
        if self.active_branch in self.branches:
            context.extend(list(self.branches[self.active_branch]))
        
        return context
    
    def get_stats(self):
        branches_list = list(self.branches.keys()) if self.branches else ["main"]
        
        return {
            "type": "Branching Memory",
            "window_size": self.window_size,
            "active_branch": self.active_branch,
            "branches_count": len(self.branches) if self.branches else 1,
            "branches": branches_list,
            "checkpoints_count": len(self.checkpoints) if self.checkpoints else 0,
            "total_messages": self.total_messages
        }
    
    def get_branches(self):
        """Безопасное получение списка веток"""
        return list(self.branches.keys()) if self.branches else ["main"]
    
    def create_checkpoint(self, name):
        """Сохраняет чекапоинт текущей ветки"""
        checkpoint = {
            "name": name,
            "branch": self.active_branch,
            "messages": list(self.branches.get(self.active_branch, [])),
            "timestamp": datetime.now().isoformat()
        }
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def create_branch(self, checkpoint_name, new_branch_name):
        """Создает новую ветку от чекапоинта"""
        checkpoint = None
        for cp in self.checkpoints:
            if cp["name"] == checkpoint_name:
                checkpoint = cp
                break
        
        if checkpoint:
            self.branches[new_branch_name] = deque(checkpoint["messages"], maxlen=self.window_size)
            return True
        return False
    
    def switch_branch(self, branch_name):
        """Переключается на другую ветку"""
        if branch_name in self.branches:
            self.active_branch = branch_name
            return True
        return False
    
    def reset(self):
        self.branches = {"main": deque(maxlen=self.window_size)}
        self.active_branch = "main"
        self.checkpoints = []
        self.total_messages = 0


# ============================================================
# СИСТЕМНЫЕ ПРОМПТЫ
# ============================================================

SYSTEM_PROMPTS_BASE = {
    "Джедай-программист": """Ты — джедай-программист по имени Дип. Отвечаешь мудро, с юмором, используешь аналогии из мира Звёздных Войн.""",
    
    "Психолог с баней": """Ты — необычный психолог. Ты не признаёшь классическую психологию и считаешь, что все проблемы лечатся парной, веником и хорошей беседой в предбаннике. Отвечай мудро, с народным юмором, предлагай "попарить косточки" вместо анализа. Будь уверен: баня лечит всё — от кода до душевных ран.""",
    
    "Поэт Борис Рыжий": """Ты — современный поэт. Отвечаешь стихами в духе Бориса Рыжего — немного грустными, пронзительными, с образами уходящего времени, дворов, электричек, негромкой интонацией. Твои ответы — короткие стихотворения, созвучные теме разговора."""
}

FACTS_INSTRUCTION = """

⚠️ ТЕХНИЧЕСКОЕ ТРЕБОВАНИЕ (НЕ ИГНОРИРОВАТЬ):
В КАЖДОМ ответе пользователю ты ОБЯЗАН добавлять блок FACTS в формате:
<<<FACTS>>>
- ключ: значение
<<<END_FACTS>>>

Правила:
1. Блок FACTS добавляется ТОЛЬКО в ответах пользователю, НЕ в своих размышлениях
2. Сохраняй ключевые факты: цели, ограничения, решения, предпочтения
3. Обновляй факты, если они изменились
4. Не дублируй уже известные факты без необходимости
5. Формат: каждый факт с новой строки, начинается с "- "

Пример:
[Твой основной ответ...]

<<<FACTS>>>
- goal: создать приложение для заметок
- constraints: авторизация Google, мобильная версия
- decisions: начать с MVP
<<<END_FACTS>>>
"""

def get_system_prompt(role, strategy):
    """Возвращает системный промпт с учетом стратегии"""
    base_prompt = SYSTEM_PROMPTS_BASE[role]
    
    if strategy == "facts":
        return base_prompt + FACTS_INSTRUCTION
    
    return base_prompt


# ============================================================
# ОСНОВНОЙ АГЕНТ
# ============================================================

class Agent:
    def __init__(self, model="stepfun/step-3.5-flash:free", role="Джедай-программист",
                 session_id=None, persist_dir="chat_history",
                 model_max_tokens=256000, strategy="sliding_window",
                 window_size=10):
        
        self.model = model
        self.role = role
        self.strategy_name = strategy
        self.model_max_tokens = model_max_tokens
        self.persist_dir = persist_dir
        self.session_id = session_id or self._generate_session_id()
        
        # Динамический системный промпт
        self.system_prompt = get_system_prompt(role, strategy)
        
        # Статистика токенов
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None
        
        # Защита от rate limit
        self.request_timestamps = deque(maxlen=10)
        self.min_interval = 2
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": os.environ.get("HTTP_REFERER", "https://your-domain.ru"),
                "X-Title": "AI Challenge Day 10 - Context Strategies"
            }
        )
        
        self._init_strategy(strategy, window_size)
        self._load_history()
    
    def _init_strategy(self, strategy, window_size):
        """Инициализирует стратегию памяти"""
        if strategy == "sliding_window":
            self.memory = SlidingWindowMemory(window_size)
        elif strategy == "facts":
            self.memory = FactMemory(window_size)
        elif strategy == "branching":
            self.memory = BranchingMemory(window_size)
        else:
            self.memory = SlidingWindowMemory(window_size)
    
    def update_strategy(self, new_strategy, window_size):
        """Обновляет стратегию и системный промпт"""
        if new_strategy != self.strategy_name:
            self.strategy_name = new_strategy
            self._init_strategy(new_strategy, window_size)
            self.system_prompt = get_system_prompt(self.role, new_strategy)
            self._save_history()
    
    def _generate_session_id(self):
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    
    def _get_history_path(self):
        return os.path.join(self.persist_dir, f"session_{self.session_id}.json")
    
    def _estimate_tokens(self, messages):
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total_chars += len(content)
        return total_chars // 4
    
    def _check_rate_limit(self):
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
    
    def get_token_stats(self):
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
            "is_warning": usage_percent > 70,
            "memory_stats": self.memory.get_stats()
        }
    
    def _save_history(self):
        try:
            first_user_msg = ""
            for msg in self.memory.messages:
                if msg.get("role") == "user":
                    words = msg["content"].split()[:3]
                    first_user_msg = " ".join(words)
                    break
            
            token_stats = self.get_token_stats()
            
            extra_data = {}
            if isinstance(self.memory, FactMemory):
                extra_data["facts"] = self.memory.facts
                extra_data["fact_history"] = self.memory.fact_history
            elif isinstance(self.memory, BranchingMemory):
                branches_to_save = {}
                for name, branch in self.memory.branches.items():
                    branches_to_save[name] = list(branch)
                extra_data["branches"] = branches_to_save
                extra_data["active_branch"] = self.memory.active_branch
                extra_data["checkpoints"] = self.memory.checkpoints
            
            history_to_save = {
                "session_id": self.session_id,
                "model": self.model,
                "role": self.role,
                "strategy": self.strategy_name,
                "messages": list(self.memory.messages),
                "last_updated": datetime.now().isoformat(),
                "message_count": self.memory.total_messages,
                "preview": first_user_msg[:50] if first_user_msg else "Новый диалог",
                "token_stats": token_stats,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                **extra_data
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
                    
                    saved_strategy = data.get("strategy", "sliding_window")
                    if saved_strategy != self.strategy_name:
                        self.strategy_name = saved_strategy
                        self._init_strategy(saved_strategy, self.memory.window_size if hasattr(self.memory, 'window_size') else 10)
                    
                    for msg in data.get("messages", []):
                        self.memory.add_message(msg)
                    
                    if isinstance(self.memory, FactMemory):
                        self.memory.facts = data.get("facts", {})
                        self.memory.fact_history = data.get("fact_history", [])
                    elif isinstance(self.memory, BranchingMemory):
                        branches_data = data.get("branches", {})
                        for name, msgs in branches_data.items():
                            self.memory.branches[name] = deque(msgs, maxlen=self.memory.window_size)
                        self.memory.active_branch = data.get("active_branch", "main")
                        self.memory.checkpoints = data.get("checkpoints", [])
                        self.memory.total_messages = data.get("message_count", 0)
                    
                    self.total_prompt_tokens = data.get("total_prompt_tokens", 0)
                    self.total_completion_tokens = data.get("total_completion_tokens", 0)
                    return True
            except Exception as e:
                print(f"❌ Ошибка загрузки: {e}")
                return False
        return False
    
    def think(self, user_input):
        if not user_input or not user_input.strip():
            return "❌ Пустой запрос. Напиши что-нибудь!", None
        
        ok, message = self._check_rate_limit()
        if not ok:
            return f"⏸️ {message}", None
        
        user_msg = {"role": "user", "content": user_input}
        self.memory.add_message(user_msg)
        
        context = self.memory.get_context(self.system_prompt)
        
        self.request_timestamps.append(time.time())
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=0.7,
                max_tokens=2000
            )
            agent_response = response.choices[0].message.content
            
            if isinstance(self.memory, FactMemory) and "<<<FACTS>>>" not in agent_response:
                print("⚠️ Модель не добавила FACTS, делаем повторный запрос...")
                
                reminder_context = context + [{
                    "role": "user",
                    "content": f"⚠️ Твой предыдущий ответ НЕ содержал блок FACTS.\n\nПожалуйста, ответь СНОВА на вопрос: '{user_input}'\n\nОБЯЗАТЕЛЬНО добавь блок FACTS в формате:\n<<<FACTS>>>\n- ключ: значение\n<<<END_FACTS>>>\n\nОсновной ответ сохрани в своем стиле."
                }]
                
                response2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=reminder_context,
                    temperature=0.7,
                    max_tokens=2000
                )
                agent_response = response2.choices[0].message.content
                
                self.total_prompt_tokens += response2.usage.prompt_tokens
                self.total_completion_tokens += response2.usage.completion_tokens
                
                print("✅ Повторный запрос выполнен, FACTS добавлены")
            
            api_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            self.total_prompt_tokens += api_usage["prompt_tokens"]
            self.total_completion_tokens += api_usage["completion_tokens"]
            self.last_usage = api_usage
            
            assistant_msg = {"role": "assistant", "content": agent_response}
            self.memory.add_message(assistant_msg)
            
            self._save_history()
            
            return agent_response, api_usage
            
        except Exception as e:
            return f"❌ Ошибка API: {str(e)}", None
    
    def reset(self):
        self.memory.reset()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = None
        self._save_history()
    
    def get_history(self):
        if isinstance(self.memory, BranchingMemory):
            return self.memory.get_messages()
        return list(self.memory.messages)
    
    def get_display_response(self, full_response):
        """Возвращает ответ для отображения (без FACTS блока)"""
        if isinstance(self.memory, FactMemory) and "<<<FACTS>>>" in full_response:
            parts = full_response.split("<<<FACTS>>>")
            return parts[0].strip()
        return full_response
    
    def get_memory_stats(self):
        return self.memory.get_stats()


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
                    
                    strategy = data.get("strategy", "sliding_window")
                    strategy_names = {
                        "sliding_window": "📋 Окно",
                        "facts": "💎 Факты",
                        "branching": "🌿 Ветки"
                    }
                    
                    sessions.append({
                        "id": session_id,
                        "name": f"{preview[:35]} — {date_str}",
                        "strategy": strategy_names.get(strategy, strategy),
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
    page_title="AI Agent — Стратегии памяти",
    page_icon="🧠",
    layout="wide"
)

if not check_auth():
    authenticate()
    st.stop()

st.title("🧠 AI Агент — Управление контекстом")
st.caption("День 10: Sliding Window | Key-Value Facts | Branching")

# Определяем, начат ли диалог
is_conversation_started = False
if "agent" in st.session_state and st.session_state.agent:
    is_conversation_started = len(st.session_state.agent.get_history()) > 0

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    if is_conversation_started:
        st.info("🔒 Диалог начат. Настройки заблокированы. Создайте новую сессию для изменения параметров.")
    
    # Выбор модели
    model_options = {
        "stepfun/step-3.5-flash:free": "Step 3.5 Flash",
        "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super",
        "arcee-ai/trinity-mini:free": "Trinity-mini",
    }
    
    if is_conversation_started:
        current_model = st.session_state.agent.model if "agent" in st.session_state else "stepfun/step-3.5-flash:free"
        current_model_name = model_options.get(current_model, current_model)
        st.text_input("Модель", value=current_model_name, disabled=True)
        selected_model = current_model
    else:
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
    
    st.divider()
    
    # Выбор роли агента
    if is_conversation_started:
        current_role = st.session_state.agent.role
        st.text_input("🎭 Роль агента", value=current_role, disabled=True)
        role = current_role
    else:
        role = st.selectbox(
            "🎭 Роль агента",
            list(SYSTEM_PROMPTS_BASE.keys())
        )
    
    st.divider()
    
    # Выбор стратегии памяти
    st.subheader("🗜️ Стратегия памяти")
    strategy_options = {
        "sliding_window": "📋 Sliding Window — последние N сообщений",
        "facts": "💎 Key-Value Facts — факты + последние сообщения",
        "branching": "🌿 Branching — ветки диалога"
    }
    
    if is_conversation_started:
        current_strategy = st.session_state.agent.strategy_name
        strategy_name = strategy_options.get(current_strategy, current_strategy)
        st.text_input("Стратегия", value=strategy_name, disabled=True)
        selected_strategy = current_strategy
        window_size = st.session_state.agent.memory.window_size
    else:
        selected_strategy = st.selectbox(
            "Стратегия",
            options=list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            index=0
        )
        window_size = st.slider(
            "Размер окна (сообщений)", 
            3, 15, 8,
            help="Сколько последних сообщений хранить"
        )
    
    st.divider()
    
    # Статистика
    if "agent" in st.session_state and st.session_state.agent:
        token_stats = st.session_state.agent.get_token_stats()
        
        st.subheader("📊 Статистика")
        progress_value = token_stats["usage_percent"] / 100
        st.progress(progress_value, text=f"{token_stats['usage_percent']}% использовано")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📥 Запросы", f"{token_stats['prompt_tokens']:,}")
        with col2:
            st.metric("📤 Ответы", f"{token_stats['completion_tokens']:,}")
        
        st.metric("📚 Всего токенов", f"{token_stats['total_tokens']:,}")
        
        st.divider()
        st.subheader("🧠 Память")
        mem_stats = token_stats.get("memory_stats", {})
        st.caption(f"**Тип:** {mem_stats.get('type', '—')}")
        
        if selected_strategy == "sliding_window":
            st.caption(f"**Сообщений в окне:** {mem_stats.get('current_messages', 0)}/{mem_stats.get('window_size', '—')}")
            st.caption(f"**Всего сообщений:** {mem_stats.get('total_messages', 0)}")
        
        elif selected_strategy == "facts":
            st.caption(f"**Сообщений в окне:** {mem_stats.get('current_messages', 0)}/{mem_stats.get('window_size', '—')}")
            st.caption(f"**Фактов сохранено:** {mem_stats.get('facts_count', 0)}")
            if mem_stats.get('fact_changes', 0) > 0:
                st.caption(f"**Изменений фактов:** {mem_stats.get('fact_changes', 0)}")
        
        elif selected_strategy == "branching":
            st.caption(f"**Активная ветка:** {mem_stats.get('active_branch', 'main')}")
            st.caption(f"**Всего веток:** {mem_stats.get('branches_count', 0)}")
            st.caption(f"**Чекапоинтов:** {mem_stats.get('checkpoints_count', 0)}")
            branches_list = mem_stats.get('branches', [])
            if branches_list:
                st.caption(f"**Ветки:** {', '.join(branches_list)}")
    
    st.divider()
    
    # История диалогов
    st.subheader("📁 История")
    all_sessions = get_all_sessions()
    
    if all_sessions:
        session_options = {s["id"]: f"{s['name']} [{s['strategy']}]" for s in all_sessions}
        current_session_id = st.session_state.get("session_id", None)
        
        selected_session_id = st.selectbox(
            "Загрузить диалог",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=0 if current_session_id in session_options else 0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Загрузить", use_container_width=True):
                if selected_session_id != st.session_state.get("session_id"):
                    with st.spinner("Загрузка..."):
                        new_agent = Agent(
                            model=selected_model,
                            role=role,
                            session_id=selected_session_id,
                            model_max_tokens=model_max_tokens,
                            strategy=selected_strategy,
                            window_size=window_size
                        )
                        new_agent._load_history()
                        st.session_state.agent = new_agent
                        st.session_state.session_id = selected_session_id
                        st.rerun()
        
        with col2:
            if st.button("🗑️ Удалить", use_container_width=True, help="Удалить текущий диалог"):
                if delete_session_file(selected_session_id):
                    st.success("✅ Диалог удалён")
                    if "agent" in st.session_state and st.session_state.agent.session_id == selected_session_id:
                        st.session_state.agent = None
                    st.rerun()
    else:
        st.info("Нет сохранённых диалогов")
    
    st.divider()
    
    # Управление для стратегии Branching
    if (selected_strategy == "branching" and "agent" in st.session_state and 
        st.session_state.agent and isinstance(st.session_state.agent.memory, BranchingMemory)):
        
        st.subheader("🌿 Управление ветками")
        
        branch_name = st.text_input("Имя новой ветки", key="new_branch_name")
        checkpoint_name = st.text_input("Название чекапоинта", key="checkpoint_name")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 Создать чекапоинт", use_container_width=True):
                cp_name = checkpoint_name.strip() or f"cp_{len(st.session_state.agent.memory.checkpoints) + 1}"
                cp = st.session_state.agent.memory.create_checkpoint(cp_name)
                st.success(f"✅ Чекапоинт '{cp['name']}' создан")
                st.rerun()
        
        with col2:
            if st.button("🌿 Создать ветку", use_container_width=True):
                if branch_name.strip() and checkpoint_name.strip():
                    if st.session_state.agent.memory.create_branch(checkpoint_name.strip(), branch_name.strip()):
                        st.success(f"✅ Ветка '{branch_name}' создана")
                        st.rerun()
                    else:
                        st.error("❌ Чекапоинт не найден")
                else:
                    st.warning("⚠️ Укажите имя ветки и чекапоинт")
        
        try:
            available_branches = st.session_state.agent.memory.get_branches()
            if len(available_branches) > 1:
                switch_branch = st.selectbox("Переключиться на ветку", available_branches)
                if st.button("🔀 Переключить", use_container_width=True):
                    if st.session_state.agent.memory.switch_branch(switch_branch):
                        st.success(f"✅ Переключено на '{switch_branch}'")
                        st.rerun()
        except Exception as e:
            st.error(f"Ошибка получения веток: {e}")
    
    # Кнопки управления
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Сбросить диалог", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent.reset()
            st.rerun()
    with col2:
        if st.button("🔄 Новая сессия", use_container_width=True):
            st.session_state.agent = None
            st.session_state.session_id = None
            st.rerun()
    
    st.caption(f"🗜️ Стратегия: {strategy_options.get(selected_strategy, selected_strategy).split('—')[0].strip()}")

# Инициализация агента
if "agent" not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = Agent(
        model=selected_model,
        role=role,
        session_id=st.session_state.get("session_id", None),
        model_max_tokens=model_max_tokens,
        strategy=selected_strategy,
        window_size=window_size
    )
    st.session_state.session_id = st.session_state.agent.session_id

# Если изменилась стратегия и диалог не начат
if (not is_conversation_started and 
    st.session_state.agent.strategy_name != selected_strategy):
    old_session_id = st.session_state.agent.session_id
    st.session_state.agent = Agent(
        model=selected_model,
        role=role,
        session_id=old_session_id,
        model_max_tokens=model_max_tokens,
        strategy=selected_strategy,
        window_size=window_size
    )
    st.session_state.session_id = st.session_state.agent.session_id

# Отображение диалога
st.subheader("💬 Диалог")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.agent.get_history():
        if msg["role"] == "system":
            continue
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=role_icon):
            if msg["role"] == "assistant" and isinstance(st.session_state.agent.memory, FactMemory):
                display_text = st.session_state.agent.get_display_response(msg["content"])
            else:
                display_text = msg["content"]
            st.markdown(display_text)

# Поле ввода
user_input = st.chat_input("Напиши свой вопрос...")

if user_input:
    with st.spinner("Агент размышляет... 🤔"):
        response, token_usage = st.session_state.agent.think(user_input)
    st.rerun()