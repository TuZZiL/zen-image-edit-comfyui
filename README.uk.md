# zen-image-edit для ComfyUI

🇬🇧 [English version →](README.md)

Ноди для ComfyUI, які запускають **Qwen-Image-2.1** на текстовому енкодері **Qwen3.5-0.8B** плюс
адаптер [zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit) — замість нативного
Qwen3-VL-8B на 17.5 ГБ. Один вузол покриває і генерацію з тексту, і редагування (до 16 референсних
зображень); DiT, VAE, семплер і префіксний KV-кеш лишаються стоковими з ComfyUI.

| | |
|---|---|
| текстовий енкодер | Qwen3.5-0.8B (1.7 ГБ, тягнеться за id) + адаптер (0.64 ГБ) |
| кондиціювання | cos **1.0000** проти diffusers-збірки тієї самої теки; **0.95** (текст) / **0.97** (зір, edit-промпти) проти нативного Qwen3-VL-8B |
| DiT / VAE | стоковий Qwen-Image-2.1, без патчів |
| обмеження | тільки англійська; цифри на вивісках можуть вийти неправильними |

## Встановлення

1. **Онови ComfyUI** — потрібна підтримка Qwen-Image-2.1 (`comfy/ldm/qwen_image21/model.py` має існувати).
2. Скопіюй цю теку в `ComfyUI/custom_nodes/zen-image-edit-comfyui/` (або `git clone` туди).
3. Завантаж **`adapter_v12.safetensors`** (0.64 ГБ) у `ComfyUI/models/`:

   ```bash
   curl -L -o ComfyUI/models/adapter_v12.safetensors \
     https://github.com/recoilme/zen-image-edit-comfyui/releases/download/v2/adapter_v12.safetensors
   ```

   v12 — поточна ревізія: таблиця позицій у її attention-гілці покриває 2304 слоти, і її донавчили
   на **реальній геометрії едиту 1024 px**, тому довгі послідовності референсів більше не гублять
   свої позиції (cos по зору проти нативного енкодера 0.93 → 0.97). `adapter_v11.safetensors`
   лишається на [релізі v1](https://github.com/recoilme/zen-image-edit-comfyui/releases/tag/v1) і
   досі завантажується — вузол читає архітектуру, включно з шириною таблиці, з метаданих файлу.
4. Поклади стокові файли моделі з [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1):
   `qwen_image_2.1_bf16.safetensors` → `models/diffusion_models/`,
   `qwen_image_2.1_vae_bf16.safetensors` → `models/vae/`.
5. Завантаж текстовий енкодер (1.7 ГБ) — воркфлоу з репо вказує саме на цю теку:

   ```bash
   hf download Qwen/Qwen3.5-0.8B --local-dir ComfyUI/models/text_encoders/qwen3.5_0.8b
   ```
6. Перезапусти ComfyUI. Потрібен `transformers >= 5.17` (підтримка Qwen3.5) — ComfyUI ставить сумісний сам.

## Приклади

Два референси персонажа на вході — три сцени на виході, усе однією нодою, у ComfyUI, 768×1280:

![examples](examples/melim_collage.jpg)

Референсні зображення — власний персонаж автора моделі; кожна сцена використала ті самі два
референси з іншим промптом і сідом. `examples/t2i_api.json` і `examples/edit_api.json` — це самі
 графи (формат API, обидва перевірені наскрізно).

## Ноди

**Zen Image Edit — Adapter Loader** — завантажує енкодер і адаптер один раз.

| вхід | значення |
|---|---|
| `adapter_file` | шлях до `adapter_v12.safetensors` (архітектура, включно з шириною таблиці позицій, лежить у його метаданих) |
| `text_encoder` | id або шлях енкодера, типово `Qwen/Qwen3.5-0.8B` |
| `model_folder` | необов'язково: чекаут zen-image-edit, тоді енкодер і адаптер беруться звідти |
| `dtype` | `bf16` (збігається з DiT) або `fp16` (відтворює diffusers-збірку) |

**Zen Image Edit — Text Encode** — `adapter`, `prompt`, `negative_prompt`, `vae` (необов'язково),
`resolution`, `image_1 … image_16`. Видає `positive`, `negative` і порожній `latent`.
Ті самі входи й виходи, що в стокової `TextEncodeQwenImage21`, тому замінює її один-в-один.

## Граф

```
UNETLoader qwen_image_2.1_bf16.safetensors ─→ ModelSamplingAuraFlow shift=5.0 ─┐
VAELoader   qwen_image_2.1_vae_bf16.safetensors ───────────────────────────────┤
ZenImage21AdapterLoader ─→ ZenImage21TextEncode ── positive/negative/latent ──→ KSampler
                                                                                │
                                        VAEDecode (vae) ←───────────────────────┘
                                             │
                                          SaveImage
```

Готові графи, обидва перевірені наскрізно через API ComfyUI:
`examples/t2i_api.json`, `examples/edit_api.json`.

* **Генерація з тексту** — лиши `image_*` порожніми, `cfg = 1`, `euler`, 25–40 кроків.
* **Редагування** — під'єднай `vae` до ноди текстового енкоду, заведи `image_1` (ціль редагування)
  і `image_2 …` (референси), та згадай їх у промпті як `<image1>`, `<image2>`. VAE **обов'язковий**:
  DiT вставляє референси як латенти.

  **Порядок має значення, і він перевертається між рантаймами.** Тут — як і в стоковій
  `TextEncodeQwenImage21` — ціль це `image_1`, і полотно береться з неї. Diffusers-пайплайн цієї
  моделі бере ціль з **останнього** зображення. Якщо перенести промпт з одного боку в інший, не
  помінявши порядок картинок, отримаєш гібрид замість заміни: вихідне зображення стає полотном, і
  модель лише вмішує обличчя (типовий симптом «нічого не замінюється»).

### Планувальник

Власний shift ComfyUI для 2.1 — **0.69** (його mu на 1024²). Diffusers-збірка цієї моделі везе
простий статичний shift **5.0**, який автор моделі визнав явно кращим — встав
`ModelSamplingAuraFlow` зі `shift = 5.0`, щоб це повторити (саме так роблять приклади графів).

## Траблшутинг

Кожне повідомлення нижче замінює сире виключення, яке не називало ні причини, ні виправлення.
Усі вони були зловлені на стоковій інсталяції (ComfyUI 0.37.0, RTX 3060, Windows) під час
налаштування цієї ноди.

### `adapter file not found: 'adapter_v12.safetensors'`

`adapter_file` резолвиться **відносно робочої директорії ComfyUI**, а не відносно `models/`, тому
звичайна причина — відносний шлях. Передай абсолютний:

```
<ComfyUI>/models/adapter_v12.safetensors
```

Завантажувач тепер ще й обшукує зареєстровані теки моделей ComfyUI (`folder_paths`), перш ніж
здатися, тож якщо покласти файл у `models/` і передати голе ім'я — зазвичай теж працює.

### `text_encoder must be a DIRECTORY, but got a file: ...`

`from_pretrained()` завантажує **теку** (config.json + ваги + токенізатор). Якщо вказати шлях до
`.safetensors`, він сприйме рядок як repo id з Hugging Face і впаде з `HFValidationError`; а тека,
що існує, але без ваг, дасть `OSError: Error no file named model.safetensors, or pytorch_model.bin`.
Передай директорію — або repo id `Qwen/Qwen3.5-0.8B`.

### `... has weight shards but no model.safetensors.index.json`

`Qwen/Qwen3.5-0.8B` називає свої ваги `model.safetensors-00001-of-00001.safetensors`. Без
`model.safetensors.index.json` `from_pretrained` їх **не бачить, хоча ваги на місці** — тобто
завантаження просто не завершене. Дотягни його:

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir <ComfyUI>/models/text_encoders/qwen3.5_0.8b
```

У теці мають бути `config.json`, `tokenizer_config.json`, шард ваг **і** `model.safetensors.index.json`
(усього 13 файлів).

### Сам `hf` падає: `TypeError: Typer.__init__() got an unexpected keyword argument 'suggest_commands'`

Версії `typer` і `huggingface_hub` у venv ComfyUI не згодні між собою — це ламає CLI `hf` **до**
будь-якого завантаження, тоді як **бібліотека** `huggingface_hub` продовжує працювати. Два виходи:

1. Передай repo id `Qwen/Qwen3.5-0.8B` як `text_encoder` — і хай `from_pretrained` його витягне.
2. Дотягни потрібні файли напряму, обминаючи CLI:

```powershell
$b = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main"
$d = "<ComfyUI>\models\text_encoders\qwen3.5_0.8b"
@("model.safetensors.index.json","tokenizer.json","vocab.json","merges.txt",
  "preprocessor_config.json","video_preprocessor_config.json","chat_template.jinja") |
  ForEach-Object { Invoke-WebRequest "$b/$_" -OutFile "$d\$_" }
```

### Перемикання конфігурацій перезавантажує енкодер

Кеш адаптерів має ключ `(model_folder, text_encoder, adapter_file, dtype)` і тримає щонайбільше
**два** записи (~2.3 ГБ VRAM кожен). Старіші записи зсуваються на CPU і звільняються при витісненні,
тож перемикання більш ніж двох конфігурацій за сесію почне швидше гальмувати, ніж падати в OOM.

## Запуск цього форку поряд із upstream-нодою

ComfyUI реєструє ноди за `node_id`, і upstream-репо займає
`ZenImage21AdapterLoader` / `ZenImage21TextEncode`. Якщо поставити обидві копії без змін, **друга
завантажена тихо замінить першу** в меню нод — і ти не скажеш, яку саме реалізацію використовує
воркфлоу.

Тому цей форк за замовчуванням додає суфікс до своїх id (`NODE_SUFFIX = "Plus"`):

```
ZenImage21AdapterLoaderPlus      Zen Image Edit Plus — Adapter Loader
ZenImage21TextEncodePlus         Zen Image Edit Plus — Text Encode
```

Постав обидві — і отримаєш чотири різні ноди, чітко підписані.

Якщо запускаєш **тільки цю копію**, очисти суфікс, щоб зберегти upstream-сумісні id (і не чіпати
наявні воркфлоу):

```bash
# Linux / macOS
export ZEN_NODE_SUFFIX=""

# Windows (cmd)
set ZEN_NODE_SUFFIX=
```

Порожній суфікс відновлює `ZenImage21AdapterLoader` / `ZenImage21TextEncode` точно як в upstream.
Постав будь-який рядок, щоб підписати другий чи третій чекаут.

## Тести

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

`tests/conftest.py` підставляє заглушки `comfy*`, тож набір тестів біжить **без ComfyUI, без GPU і
без ваг моделей** — він перевіряє чисту логіку (парсинг метаданих, резолв шляхів, валідацію енкодера).

## Примітки

* **Тільки англійська** — адаптер навчали й тестували на англійських підписах та інструкціях.
* **Цифри на вивісках** можуть вийти неправильними («OPEN 24 HOURS» → «OPEN 26 HOURS» на кожному
  перевіреному сіді).
* Кондиціювання збігається з diffusers-збіркою біт-у-біт (cos 1.0000); проти нативного енкодера
  Qwen3-VL-8B — 0.94, тобто та сама якість, що й у diffusers-пайплайні.
* Адаптер передбачає позиції **тільки для тексту** — слоти зображень заповнює DiT латентами, саме
  тому для редагування потрібен VAE.

## Ліцензія

Код нод: Apache-2.0. Ваги адаптера і все похідне від Qwen-Image-2.1 підпадають під
Qwen Research License — див. NOTICE у [AiArtLab/zen-image-edit](https://huggingface.co/AiArtLab/zen-image-edit).
Qwen3.5-0.8B — Apache-2.0.
