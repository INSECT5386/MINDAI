import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


# 데이터 로드
file_path = "filtered_merged_data.csv"
data = pd.read_csv(file_path, encoding='utf-8', header=None, names=["Questions", "Answers"], on_bad_lines="skip")

questions = data["Questions"].astype(str).tolist()
answers = data["Answers"].astype(str).tolist()

# ✅ 데이터 전처리
questions = ["<start> " + q + " <end>" for q in questions]
answers = ["<start> " + a + " <end>" for a in answers]

# ✅ <start> 바로 다음에 <end>가 오는 데이터 제거
filtered_questions = []
filtered_answers = []
for q, a in zip(questions, answers):
    if not (a == "<start> <end>"):
        filtered_questions.append(q)
        filtered_answers.append(a)
questions, answers = filtered_questions, filtered_answers

# ✅ 토크나이저 생성 및 저장
special_tokens = ["<start>", "<end>"]
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>")
tokenizer.fit_on_texts(special_tokens + questions + answers)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# ✅ 단어 개수
vocab_size = len(tokenizer.word_index) + 1

# ✅ 하이퍼파라미터
max_len = 40
latent_dim = 136
embedding_dim = 68

# ✅ 데이터 토큰화 및 패딩
input_sequences = tokenizer.texts_to_sequences(questions)
output_sequences = tokenizer.texts_to_sequences(answers)

input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='post')
output_data = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_len, padding='post')

# ✅ 인코더 모델 정의
encoder_inputs = tf.keras.Input(shape=(max_len,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_gru = tf.keras.layers.GRU(latent_dim, return_state=True, dropout=0.2)
encoder_outputs, state_h = encoder_gru(encoder_embedding)
encoder_states = [state_h]

# ✅ 디코더 모델 정의
decoder_inputs = tf.keras.Input(shape=(max_len,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.2)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# ✅ 모델 생성 및 컴파일
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ 데이터셋 준비
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(output_data[:, :-1], maxlen=max_len, padding='post')
decoder_target_data = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(output_data[:, 1:], maxlen=max_len, padding='post'), -1)

# 데이터셋을 점차적으로 증가시키는 학습 로직
initial_size = 5000
increment_size = 5000
max_size = len(input_data)  # 최대 데이터셋 크기 (전체 데이터)

for size in range(initial_size, max_size + 1, increment_size):
    print(f"학습 데이터 크기: {size}")
    
    # 부분 데이터셋을 선택
    input_data_batch = input_data[:size]
    decoder_input_data_batch = decoder_input_data[:size]
    decoder_target_data_batch = decoder_target_data[:size]
    
    # 모델 학습 (단계별로 학습)
    history = model.fit(
        [input_data_batch, decoder_input_data_batch], decoder_target_data_batch,
        epochs=5,
        batch_size=12,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('seq2seq_model_best.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
        ],
        verbose=1
    )

    # 모델 저장 (매 단계마다 모델을 저장할 수 있음)
    model.save(f"seq2seq_model_{size}.h5")
    print(f"모델 {size} 저장 완료")
