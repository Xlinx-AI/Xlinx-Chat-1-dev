# LFByteTransformer

## Overview / Обзор

**English**

LFByteTransformer is a custom Language Model (LLM) architecture that combines adaptive linear layers, token mixing, channel mixing, and a Mixture of Experts (MoE) module. It operates on byte-level input, allowing it to handle any type of data that can be represented as bytes, such as text, images, audio, and more. This repository provides the implementation of LFByteTransformer, training scripts, and an inference script using Gradio for easy interaction.

**Русский**

LFByteTransformer — это пользовательская архитектура языковой модели (LLM), которая сочетает в себе адаптивные линейные слои, перемешивание токенов, перемешивание каналов и модуль Mixture of Experts (MoE). Она работает на уровне байтов, что позволяет обрабатывать любые данные, представимые в виде байтов, такие как текст, изображения, аудио и многое другое. Этот репозиторий предоставляет реализацию LFByteTransformer, скрипты для обучения и скрипт инференса с использованием Gradio для удобного взаимодействия.

---

## Table of Contents / Содержание

- [Features / Особенности](#features--особенности)
- [Usage / Использование](#usage--использование)
  - [Training / Обучение](#training--обучение)
  - [Inference / Инференс](#inference--инференс)
- [Model Architecture / Архитектура модели](#model-architecture--архитектура-модели)
- [Examples / Примеры](#examples--примеры)
- [Contributing / Вклад](#contributing--вклад)
- [License / Лицензия](#license--лицензия)

---

## Features / Особенности

**English**

- Byte-level language model capable of handling diverse data types.
- Adaptive Linear layers that adjust weights based on the input.
- Token Mixing and Channel Mixing layers for enhanced feature interactions.
- Mixture of Experts module for dynamic expert selection.
- Supports distributed training with mixed precision for performance optimization.
- Gradio-based inference script for easy interaction and testing.

**Русский**

- Языковая модель на уровне байтов, способная обрабатывать различные типы данных.
- Адаптивные линейные слои, настраивающие веса на основе входных данных.
- Слои перемешивания токенов и каналов для улучшения взаимодействия признаков.
- Модуль Mixture of Experts для динамического выбора экспертов.
- Поддержка распределенного обучения со смешанной точностью для оптимизации производительности.
- Скрипт инференса на базе Gradio для удобного взаимодействия и тестирования.

---

## Usage / Использование

### Training / Обучение

**English**

To train the LFByteTransformer model, you can use the provided `train.py` script. The training script supports distributed training with mixed precision and gradient accumulation.

**Example Command:**

```bash
torchrun --nproc_per_node=2 train.py \
    --distributed \
    --epochs 10 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --seq_len 256 \
    --embed_dim 512 \
    --num_layers 8 \
    --adapt_dim 512 \
    --num_experts 4 \
    --lr 3e-4 \
    --max_grad_norm 1.0 \
    --dropout 0.1 \
    --save_path "./checkpoints"
```

**Русский**

Для обучения модели LFByteTransformer вы можете использовать предоставленный скрипт `train.py`. Скрипт обучения поддерживает распределенное обучение со смешанной точностью и накоплением градиентов.

**Пример команды:**

```bash
torchrun --nproc_per_node=2 train.py \
    --distributed \
    --epochs 10 \
    --batch_size 8 \
    --accumulation_steps 4 \
    --seq_len 256 \
    --embed_dim 512 \
    --num_layers 8 \
    --adapt_dim 512 \
    --num_experts 4 \
    --lr 3e-4 \
    --max_grad_norm 1.0 \
    --dropout 0.1 \
    --save_path "./checkpoints"
```

#### Command Arguments / Аргументы команды

- `--distributed`: Enable DistributedDataParallel (DDP) training.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size per GPU.
- `--accumulation_steps`: Number of gradient accumulation steps.
- `--seq_len`: Sequence length for input data.
- `--embed_dim`: Embedding dimension.
- `--num_layers`: Number of LFModel layers.
- `--adapt_dim`: Adaptation dimension.
- `--num_experts`: Number of experts in MoE module.
- `--lr`: Learning rate.
- `--max_grad_norm`: Max gradient norm for gradient clipping.
- `--dropout`: Dropout rate.
- `--save_path`: Path to save model checkpoints.

**Note:** Adjust the parameters based on your hardware capabilities.

**Русский**

#### Описание аргументов

- `--distributed`: Включает распределенное обучение с использованием DDP.
- `--epochs`: Количество эпох обучения.
- `--batch_size`: Размер батча на один GPU.
- `--accumulation_steps`: Количество шагов накопления градиентов.
- `--seq_len`: Длина входной последовательности.
- `--embed_dim`: Размерность эмбеддингов.
- `--num_layers`: Количество слоев LFModel.
- `--adapt_dim`: Размерность адаптации.
- `--num_experts`: Число экспертов в модуле MoE.
- `--lr`: Скорость обучения.
- `--max_grad_norm`: Максимальная норма градиента для клиппинга.
- `--dropout`: Вероятность дропаут.
- `--save_path`: Путь для сохранения контрольных точек модели.

**Примечание:** Настройте параметры в соответствии с возможностями вашего оборудования.

### Inference / Инференс

**English**

An inference script using Gradio is provided for easy interaction with the trained model. You can generate text by providing a starting sequence.

**Run the inference script:**

```bash
python app.py
```

**Русский**

Для удобного взаимодействия с обученной моделью предоставлен скрипт инференса с использованием Gradio. Вы можете генерировать текст, предоставляя начальную последовательность.

**Запустите скрипт инференса:**

```bash
python inference.py
```

#### Using the Gradio Interface / Использование интерфейса Gradio

1. After running the script, a Gradio web interface will open in your browser.
2. Enter the starting text in the input box.
3. Adjust the `Max Length` and `Temperature` sliders as needed.
4. Click the "Submit" button to generate text.

**Русский**

1. После запуска скрипта в вашем браузере откроется веб-интерфейс Gradio.
2. Введите начальный текст в поле ввода.
3. При необходимости настройте ползунки `Max Length` и `Temperature`.
4. Нажмите кнопку "Submit" для генерации текста.

---

## Model Architecture / Архитектура модели

**English**

The LFByteTransformer model integrates several advanced components:

- **AdaptiveLinear**: Linear layers that adapt their weights based on the input.
- **TokenMixing**: Mixes token representations to capture interactions across the sequence.
- **ChannelMixing**: Mixes channel (feature) representations to enhance feature interactions.
- **MixtureOfExperts (MoE)**: Dynamically selects expert networks based on the input.
- **PositionalEncoding**: Adds positional information to the byte embeddings.

**Русский**

Модель LFByteTransformer включает несколько продвинутых компонентов:

- **AdaptiveLinear**: Линейные слои, адаптирующие свои веса на основе входных данных.
- **TokenMixing**: Перемешивает представления токенов для захвата взаимодействий в последовательности.
- **ChannelMixing**: Перемешивает представления каналов (признаков) для улучшения взаимодействия признаков.
- **MixtureOfExperts (MoE)**: Динамически выбирает экспертные сети на основе входных данных.
- **PositionalEncoding**: Добавляет позиционную информацию к эмбеддингам байтов.

---

## Examples / Примеры

**English**

Here's how you can generate text using the model:

```python
from train import LFByteTransformer
import torch

# Load the trained model
model = LFByteTransformer()
checkpoint = torch.load('checkpoints/model_epoch_9.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
start_sequence = "Once upon a time"
generated_text = generate_text(model, start_sequence, max_length=100)
print("Generated text:", generated_text)
```

**Русский**

Вот как вы можете сгенерировать текст с помощью модели:

```python
from train import LFByteTransformer
import torch

# Загрузите обученную модель
model = LFByteTransformer()
checkpoint = torch.load('checkpoints/model_epoch_9.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Генерация текста
start_sequence = "Жили-были"
generated_text = generate_text(model, start_sequence, max_length=100)
print("Сгенерированный текст:", generated_text)
```

---

## Contributing / Вклад

**English**

Contributions are welcome! Please open issues or pull requests for any improvements or bug fixes.

**Русский**

Мы приветствуем вклад сообщества! Пожалуйста, открывайте issues или pull requests для любых улучшений или исправлений.

---

## License / Лицензия

This project is licensed under the MIT License.

**Русский**

Этот проект лицензирован под лицензией MIT.

---
