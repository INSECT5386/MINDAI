import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 로드 (예시: 데이터는 'data.csv' 파일에서 로드)
data = pd.read_csv("data.csv", encoding='utf-8', header=None, names=["data"], on_bad_lines="skip")

# 예시 텍스트 데이터 확인
sample_data = data['data'][:5]
print("Sample data:", sample_data)

# start_token과 end_token 설정
start_token = "<start>"
end_token = "<end>"

# 훈련 데이터를 수정할 때
train_data = [start_token + " " + text + " " + end_token for text in data['data']]

# 토큰화
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)  # 전체 데이터로 학습

# 데이터의 토큰화된 샘플
tokens = tokenizer.texts_to_sequences(train_data)
print("Tokens for the sample text:", tokens[:5])

# 패딩 추가
max_length = max([len(seq) for seq in tokens])  # 가장 긴 시퀀스 길이
padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post', maxlen=max_length)

print("Padded sequence sample:", padded_tokens[:5])

# 데이터 분리
train_data, val_data = train_test_split(padded_tokens, test_size=0.1, random_state=42)

# 파라미터 설정
vocab_size = len(tokenizer.word_index) + 1  # 토큰화된 단어 수
num_heads = 8
num_layers = 4
dropout_rate = 0.1
epochs = 10
batch_size = 12

# 파라미터 설정
embedding_dim = 256  # embedding_dim과 hidden_dim을 동일하게 설정
hidden_dim = 256

# Transformer 모델 정의
def transformer_model(input_shape, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout_rate):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # 임베딩
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = embedding

    # 여러 층의 Transformer 블록
    for _ in range(num_layers):
        # 멀티헤드 어텐션
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + x)  # 잔차 연결과 LayerNorm
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # 피드포워드 네트워크
        ff = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
        ff = tf.keras.layers.Dropout(dropout_rate)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff + x)  # 잔차 연결과 LayerNorm

    # 출력
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)

    # Model 클래스는 tf.keras.Model을 사용
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# 모델 컴파일
model = transformer_model((max_length,), vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout_rate)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 입력과 타깃 시퀀스 준비 함수
def prepare_sequences(data, tokenizer, max_length):
    input_sequences = []
    target_sequences = []

    for seq in data:
        input_seq = seq[:-1]  # 입력 시퀀스는 마지막 토큰을 제외한 부분
        target_seq = seq[1:]  # 타깃 시퀀스는 첫 번째 토큰을 제외한 부분

        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post', maxlen=max_length)
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, padding='post', maxlen=max_length)

    return np.array(input_sequences), np.array(target_sequences)

# 학습 데이터와 검증 데이터 준비
train_input, train_target = prepare_sequences(train_data, tokenizer, max_length)
val_input, val_target = prepare_sequences(val_data, tokenizer, max_length)

# 학습
model.fit(train_input, train_target, epochs=epochs, batch_size=batch_size, validation_data=(val_input, val_target))


def predict_text(model, tokenizer, start_token, end_token, max_length, temperature=1.0):
    input_seq = [tokenizer.texts_to_sequences([start_token])[0]]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_length, padding='post')

    generated_text = start_token

    for _ in range(max_length - 1):
        predictions = model.predict(input_seq)[0, -1, :]
        predictions = predictions / temperature
        probabilities = tf.nn.softmax(predictions).numpy()
        predicted_id = np.random.choice(len(probabilities), p=probabilities)

        predicted_word = tokenizer.index_word.get(predicted_id, "<unk>")
        generated_text += " " + predicted_word

        if predicted_word == end_token:  # 종료 토큰이면 루프 탈출
            break

        input_seq = np.append(input_seq[:, 1:], [[predicted_id]], axis=1)

    return generated_text

# 예측 (start_token을 입력하여 텍스트 생성)
generated_text = predict_text(model, tokenizer, start_token, end_token, max_length)
print("Generated text:", generated_text)
model.save(f'Transformer_model_10000.h5')
