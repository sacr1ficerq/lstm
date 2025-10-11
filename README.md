# LSTM multi-label classification with BPE tokenizer written in Rust

Реализовал по формулкам и сравнил разные модели:
- RNN
- LSMT
- LSTM + Attention
- LSTM deep

получилось выбить f1-score 40%.

в качестве эксперимента написал на Rust BPE токенизатор и обернул его в питоновский модуль, чтобы можно было вызывать внутри проекта.
