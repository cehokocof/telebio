# 🤖 TeleBio

> Автоматическая смена описания профиля (Bio) в Telegram — по таймеру, из списка фраз, через YandexGPT или по контексту ваших недавних сообщений.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)
![Telethon](https://img.shields.io/badge/Telethon-MTProto-0088cc?logo=telegram)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-100%20passed-brightgreen?logo=pytest)

---

## 📋 Оглавление

- [Возможности](#-возможности)
- [Архитектура](#-архитектура)
- [Структура проекта](#-структура-проекта)
- [Быстрый старт](#-быстрый-старт)
- [Конфигурация](#-конфигурация)
- [Провайдеры био](#-провайдеры-био)
- [Telegram-бот](#-telegram-бот-управление)
- [Docker](#-docker)
- [Тестирование](#-тестирование)
- [Расширение](#-расширение)

---

## ✨ Возможности

| Фича | Описание |
|---|---|
| 🔄 **Автосмена био** | Меняет "О себе" в Telegram по заданному интервалу (по умолчанию — каждые 60 минут) |
| 📝 **Список фраз** | Берёт фразы из JSON-файла, перебирает их последовательно с зацикливанием |
| 🧠 **YandexGPT** | Генерирует абсурдные/смешные фразы через API YandexGPT с few-shot примерами |
| 🧩 **Контекст сообщений** | Генерирует био по вашим недавним исходящим сообщениям, чтобы отражать текущее состояние |
| 🔌 **Модульность** | Провайдеры реализуют `Protocol` — можно добавить свой за 5 минут |
| 🤖 **Telegram-бот** | Управление через бота: `/status`, `/history`, `/set_mode`, `/context`, `/new`, `/pause` |
| 🛡️ **Обработка ошибок** | `FloodWaitError` — автоматический sleep + retry; остальные ошибки не крашат процесс |
| ⚡ **Graceful shutdown** | Корректное завершение по `SIGINT`/`SIGTERM` |
| 🐳 **Docker-ready** | Dockerfile + docker-compose для деплоя на сервер |
| ✅ **100 тестов** | Полное покрытие: провайдеры, сервис Telegram, bot-команды, state, планировщик, конфиг, фабрика |

---

## 🚀 Быстрый старт

### Предварительные требования

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (рекомендуемый менеджер пакетов) или pip
- **Telegram API credentials** — получить на [my.telegram.org/apps](https://my.telegram.org/apps)

### 1. Клонирование и установка

```bash
git clone https://github.com/your-username/telebio.git
cd telebio
uv sync
```

### 2. Создание `.env`

```bash
cp .env.example .env
```

Заполните обязательные поля:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
```

### 3. Первый запуск

```bash
uv run python main.py
```

При первом запуске Telethon попросит ввести номер телефона и код подтверждения. После этого будет создан файл `telebio.session` — повторный логин не потребуется.

---

## ⚙️ Конфигурация

Все настройки задаются через переменные окружения (файл `.env`):

| Переменная | Обязательная | По умолчанию | Описание |
|---|:---:|---|---|
| `TELEGRAM_API_ID` | ✅ | — | API ID из my.telegram.org |
| `TELEGRAM_API_HASH` | ✅ | — | API Hash из my.telegram.org |
| `BIO_PROVIDER` | | `list` | Провайдер: `list`, `llm` или `context` |
| `UPDATE_INTERVAL_MINUTES` | | `60` | Интервал смены био (минуты) |
| `PHRASES_FILE` | | `data/phrases.json` | Путь к файлу фраз |
| `EXAMPLES_FILE` | | `data/examples.json` | Путь к few-shot примерам для LLM |
| `SESSION_NAME` | | `telebio` | Имя файла сессии Telethon |
| `LOG_LEVEL` | | `INFO` | Уровень логирования |
| `YANDEX_API_KEY` | ⚠️ | — | API-ключ Yandex Cloud (нужен при `llm`/`context`) |
| `YANDEX_FOLDER_ID` | ⚠️ | — | Folder ID в Yandex Cloud (нужен при `llm`/`context`) |
| `YANDEX_MODEL` | | `yandexgpt-lite/latest` | Модель YandexGPT |
| `YANDEX_TEMPERATURE` | | `0.9` | Температура генерации (0.0–1.0) |
| `CONTEXT_DAYS` | | `14` | Стартовый период сообщений для `context` |
| `CONTEXT_LIMIT` | | `500` | Стартовый лимит сообщений для `context` |
| `CONTEXT_DIALOG_SCAN_LIMIT` | | `10` | Сколько последних диалогов сканировать |
| `CONTEXT_PER_DIALOG_LIMIT` | | `50` | Сколько сообщений смотреть в каждом диалоге |
| `CONTEXT_TOP_K` | | `15` | Сколько scored сообщений отдавать в prompt |
| `CONTEXT_MIN_SCORE` | | `0.55` | Минимальный relevance score для сообщения |
| `CONTEXT_EXCLUDED_DIALOGS` | | `telebio` | Чаты, исключённые из context scoring |
| `CONTEXT_ENABLE_NLI` | | `true` | Включить локальный NLI relevance слой |
| `CONTEXT_SEMANTIC_SCORER` | | `nli` | Semantic scorer: `nli` или `embedding` |
| `CONTEXT_NLI_MODEL` | | `cointegrated/rubert-base-cased-nli-threeway` | Hugging Face NLI модель |
| `CONTEXT_EMBEDDING_MODEL` | | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face embedding модель |
| `STATE_FILE` | | `telebio_state.json` | JSON-файл для bot-controlled mode/context settings |
| `BOT_TOKEN` | | — | Токен Telegram-бота от @BotFather (для управления) |

---

## 🎭 Провайдеры био

### 📝 `list` — Список фраз

Берёт фразы из `data/phrases.json` и перебирает по кругу:

```json
[
    "☕ Код, кофе, повтор",
    "🐍 Python — мой второй язык",
    "⚡ async def life(): await adventure()"
]
```

- Фразы длиннее 70 символов автоматически обрезаются
- Пустой файл или невалидный JSON — ошибка при старте

### 🧠 `llm` — YandexGPT

Генерирует абсурдные фразы через [YandexGPT Foundation Models API](https://yandex.cloud/ru/docs/foundation-models/):

**Системный промпт:**
> Ты — генератор случайных абсурдных фактов и сюрреалистичного юмора.
> Придумай странную, смешную фразу для био.
> Длина: до 60 символов. Тон: хаотичный, непредсказуемый, абсурдный.

**Few-shot примеры** загружаются из `data/examples.json` и включаются в запрос как пары `user → assistant`:

```json
[
    "Борщ — это UI-фреймворк для бабушек",
    "Выгуливаю массив на поводке",
    "Обучаю нейросеть варить пельмени"
]
```

Чем точнее примеры отражают желаемый стиль — тем лучше результат.

**Для включения:**
```env
BIO_PROVIDER=llm
YANDEX_API_KEY=ваш_ключ
YANDEX_FOLDER_ID=ваш_folder_id
```

### 🧩 `context` — YandexGPT по вашим сообщениям

Собирает ваши недавние исходящие текстовые сообщения в Telegram и просит YandexGPT сжать их в короткое био, которое отражает текущие темы, настроение и фокус внимания.

- Источник: исходящие текстовые сообщения аккаунта Telethon
- Окно: гибрид `CONTEXT_DAYS` + `CONTEXT_LIMIT`
- Фильтрация: heuristic rules + semantic scorer (`nli` или `embedding`), с JSON-логом решений
- Skip: если выбранный context не изменился, YandexGPT не вызывается
- Порядок в промпте: от старых сообщений к новым
- Настройки окна можно менять через Telegram-бота: `/context 7 300`
- Выбранный режим и context-настройки сохраняются в `STATE_FILE`

**Для включения:**
```env
BIO_PROVIDER=context
YANDEX_API_KEY=ваш_ключ
YANDEX_FOLDER_ID=ваш_folder_id
CONTEXT_DAYS=14
CONTEXT_LIMIT=500
```

**Экспорт истории для экспериментов:**
```bash
PYTHONPATH=src:. uv run python -m labeling.cli.context_export \
  --days 30 \
  --limit 1000 \
  --dialogs 50 \
  --per-dialog 200 \
  --output tests/fixtures/context_messages_live.json
```

После этого можно сравнивать фильтры на собранном JSON:
```bash
PYTHONPATH=src:. uv run python -m labeling.cli.context_relevance_report \
  tests/fixtures/context_messages_live.json \
  --scorer nli
```

Benchmark scoring pipeline на 100 сообщениях:
```bash
PYTHONPATH=src:. uv run python -m labeling.benchmarks.benchmark_100_messages
```

---

## 🧪 Тестирование

### Запуск тестов

```bash
uv sync --group dev
uv run pytest -v
```

### Покрытие

| Модуль | Тестов | Что проверяется |
|---|:---:|---|
| `test_config.py` | 12 | `_get_env`, `Settings` (пути, frozen, defaults), `load_settings` |
| `test_list_provider.py` | 10 | Загрузка JSON, ошибки валидации, truncation, sequential cycling |
| `test_llm_provider.py` | 15 | Init, `_build_request_body`, `_extract_text`, `get_bio` через mocked HTTP |
| `test_context_provider.py` | 5 | ContextBioProvider, prompt body, truncation, mocked HTTP |
| `test_telegram_service.py` | 9 | Lifecycle, `update_bio`, `collect_recent_outgoing_texts`, FloodWait/RPCError |
| `test_scheduler.py` | 3 | Цикл обновления, устойчивость к ошибкам provider/telegram |
| `test_bot_handlers.py` | 29 | Все команды бота: /status, /history, /set_mode, /context, /new, /pause |
| `test_main.py` | 7 | Фабрика: list, llm, context, missing credentials, unknown provider |
| `test_state.py` | 8 | Runtime state load/save, fallback, context validation |
| `test_protocol.py` | 2 | Оба провайдера реализуют `BioProvider` Protocol |
| **Итого** | **100** | |

### Стек тестирования

- **pytest** + **pytest-asyncio** — запуск async-тестов
- **respx** — мок HTTP-запросов к YandexGPT (поверх httpx)
- **unittest.mock** — мок Telethon клиента

---

## 🤖 Telegram-бот (управление)

Если задана переменная `BOT_TOKEN`, запускается управляющий бот. Все команды доступны только владельцу аккаунта.

| Команда | Описание |
|---|---|
| `/status` | Текущий режим, состояние (активно / на паузе), последнее био |
| `/history` | Последние 10 обновлений био с таймстампами |
| `/set_mode list\|llm\|context` | Переключение провайдера на лету |
| `/context` | Показать окно сообщений для context-режима |
| `/context <days> <limit>` | Изменить и сохранить окно сообщений, например `/context 14 500` |
| `/new` | Немедленно сгенерировать и применить новое био |
| `/pause` | Приостановить / возобновить автообновление (текущее био сохраняется) |

Хендлеры вынесены в отдельные модули (`services/handlers/`), что упрощает добавление новых команд — достаточно создать файл с функцией-обработчиком и зарегистрировать её в `handlers/__init__.py`.

---

## 📜 Лицензия

MIT — делайте что хотите.
