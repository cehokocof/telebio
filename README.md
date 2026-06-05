# TeleBio

Автосмена Telegram-bio по расписанию. Берёт текст для bio у выбранного провайдера и
обновляет поле «О себе» через MTProto (Telethon). Управление — опционально через
Telegram-бота.

---

## Главная фича: provider `context_prod`

Bio собирается не из готового списка и не «из головы» LLM, а из вашего собственного
стиля общения за последние дни. Pipeline:

```
outgoing messages (Telethon)
    │
    ▼
склейка соседних реплик одного диалога        (CONTEXT_PROD_MERGE_GAP_SECONDS)
    │  закрытие группы — входящее от собеседника
    │  или gap > N секунд
    ▼
pre-filter по длине → drop                    (CONTEXT_PROD_MAX_MESSAGE_LENGTH)
    │  отсечение «портянок» до эмбеддингов
    │  (защита от OOM и пустой работы модели)
    ▼
mix0035 classifier
    ├─ stage1: rubert-tiny2 + numeric features  → drop / passthrough
    ├─ stage2: distiluse-multilingual + CatBoost → drop / maybe / keep
    └─ (опц.) NLI score                         (CONTEXT_PROD_ENABLE_NLI_SCORE)
    │
    ▼
SQLite-очередь pending (maybe + keep)
    │
    ▼
ready_batch?
    ├─ accumulated ≥ CONTEXT_PROD_MIN_BATCH                       → fire
    └─ accumulated ≥ FALLBACK_MIN_BATCH &&
       oldest msg ≥ FALLBACK_MAX_AGE_DAYS                         → fire
    │
    ▼
prompt = top-N keep + ≤ M maybe → YandexGPT → bio
    │
    ▼
update_bio() → commit_used (messages помечены как использованные,
              в следующий батч не попадут)
```

Двухстадийный классификатор на эмбеддингах вместо чистого ключевого матчинга
позволяет отделять content-bearing реплики от мусора («ок», «понял», ссылок,
команд бота). Батч не запускается, пока не накопится материал — это страхует от
bio, сгенерированного из 2-3 случайных «ага».

Веса mix0035 (`stage1_catboost.cbm`, `stage2_nearest_centroid.pkl`) лежат в
`data/prod_models/mix0035/` и закоммичены — они небольшие (~150 KB).
Stage-1/2 трансформеры подтягиваются с HuggingFace по именам из `.env` при
первом запуске.

---

## Архитектура

```
Scheduler (asyncio)
    ├── collect_context() — periodic Telethon poll + классификация (только context_prod)
    ├── BioProvider.get_bio() — list | llm | context_prod
    └── TelegramService.update_bio() — Telethon + FloodWait retry

BotService (опц.)
    └── /status /history /set_mode /new /pause
```

`BioProvider` — `typing.Protocol` (single async-метод `get_bio()`), реализации
подменяются переменной окружения, наследования нет.

---

## Запуск

Требования: Python 3.13+, [uv](https://docs.astral.sh/uv/), Telegram API creds
с [my.telegram.org/apps](https://my.telegram.org/apps).

```bash
git clone https://github.com/your-username/telebio.git
cd telebio
uv sync
cp .env.example .env      # заполнить TELEGRAM_API_ID / TELEGRAM_API_HASH
uv run python main.py
```

Первый запуск — интерактивный логин Telethon (телефон + код), создаст
`telebio.session`. Дальше — non-interactive.

---

## Провайдеры

| `BIO_PROVIDER` | Что делает |
|---|---|
| `list`         | Циклически перебирает фразы из `data/phrases.json`. Простой fallback. |
| `llm`          | Прямой вызов YandexGPT с few-shot из `data/examples.json`. Без вашего контекста. |
| `context_prod` | Pipeline сверху: outgoing → склейка → классификатор → батч → YandexGPT. |

Свой провайдер — файл в `src/telebio/providers/` с `async def get_bio(self) -> str` и
ветка в фабрике `main.py`. Protocol structural-typing, никакого ABC.

---

## Конфигурация

Все настройки — в `.env`. Каждая переменная в `.env.example` снабжена комментарием
о назначении, дефолте и побочных эффектах — отдельную таблицу здесь не
дублирую, чтобы не разъезжалась со временем. Смотреть `.env.example`.

Минимум для запуска: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`.
Для `BIO_PROVIDER=llm` или `context_prod`: `YANDEX_API_KEY` + `YANDEX_FOLDER_ID`.

---

## Управляющий бот

Если задан `BOT_TOKEN`, поднимается локальный бот; команды доступны только владельцу
аккаунта.

| Команда | Назначение |
|---|---|
| `/status`            | Текущий провайдер, состояние, последнее bio. |
| `/history`           | Последние 10 обновлений с таймстампами. |
| `/set_mode list\|llm\|context_prod` | Переключение провайдера на лету. |
| `/new`               | Сгенерировать и применить bio немедленно (для `context_prod` — с `force=True`). |
| `/pause`             | Пауза / возобновление автообновления; текущее bio остаётся. |

Хендлеры — отдельные модули в `services/handlers/`, регистрация через
`register_all` в `handlers/__init__.py`.

---

## Обработка ошибок

| Ситуация | Поведение |
|---|---|
| `FloodWaitError` | `sleep(N)` + одна повторная попытка. |
| `RPCError`       | Логирование, re-raise. |
| Ошибка провайдера / сервиса | Логирование, переход к следующему тику, процесс жив. |
| `ContextBatchNotReady` | Лог-info, `update_bio` пропускается, ждём следующего тика. |
| Bio > 70 символов | Авто-обрезка + warning. |
| `BIO_PROVIDER=llm`/`context_prod` без Yandex-ключей | Фатально при старте, понятное сообщение. |

---

## Docker

```bash
docker compose up --build
docker attach telebio       # для первого интерактивного логина Telethon
```

После создания `telebio.session` можно убрать `stdin_open`/`tty` из compose.

Volumes: `./telebio.session`, `./data` (ro). Restart policy: `unless-stopped`.

Для `context_prod` веса mix0035 уже в репозитории
(`data/prod_models/mix0035/`). Перетренировать — `labeling/train_catboost.py`
(см. также вспомогательные скрипты в `labeling/` и `scripts/`).

---

## Тесты

```bash
uv sync --group dev
uv run pytest -v
```

Стек: `pytest`, `pytest-asyncio`, `respx` (мок HTTP к YandexGPT),
`unittest.mock` (Telethon). Покрытие: конфиг, list/llm-провайдеры,
TelegramService с FloodWait, scheduler, бот-команды, фабрика, Protocol-conformance.

---

## Лицензия

MIT.
