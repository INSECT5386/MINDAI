import tensorflow as tf
import numpy as np
import pandas as pd

# 모델 불러오기 (모델 파일 경로를 지정해 주세요)
model = tf.keras.models.load_model('seq2seq_model_50000.h5')

# 모델의 모든 레이어를 동결(freeze)하여 기존 가중치 유지
for layer in model.layers:
    layer.trainable = False  # 모든 레이어를 학습하지 않음

# 복습을 위해 마지막 레이어만 학습 가능하도록 설정
model.layers[-1].trainable = True  # 마지막 레이어만 학습

# 모델 컴파일 (최적화기, 손실 함수, 지표 지정)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 데이터 불러오기
file_path = "filtered_merged_data.csv"
data = pd.read_csv(file_path, encoding='utf-8', header=None, names=["Questions", "Answers"], on_bad_lines="skip")

questions = data["Questions"].astype(str).tolist()
answers = data["Answers"].astype(str).tolist()

# 데이터 전처리
questions = ["<start> " + q + " <end>" for q in questions]
answers = ["<start> " + a + " <end>" for a in answers]

filtered_questions = []
filtered_answers = []
for q, a in zip(questions, answers):
    if not (a == "<start> <end>"):
        filtered_questions.append(q)
        filtered_answers.append(a)
questions, answers = filtered_questions, filtered_answers

# 토크나이저 생성
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>")
tokenizer.fit_on_texts(questions + answers)

# 시퀀스 변환
max_len = 40
input_sequences = tokenizer.texts_to_sequences(questions)
output_sequences = tokenizer.texts_to_sequences(answers)

# 패딩 추가
input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='post')
output_data = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_len, padding='post')

# 디코더 입력 및 목표 데이터 생성
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(output_data[:, :-1], maxlen=max_len, padding='post')
decoder_target_data = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(output_data[:, 1:], maxlen=max_len, padding='post'), -1)

# 학습 설정
epochs = 5
batch_size = 12
validation_split = 0.1

# 학습 데이터 배치 설정
initial_size = 15000
increment_size = 15000
max_size = len(input_data)

for size in range(initial_size, max_size + 1, increment_size):
    print(f"학습 데이터 크기: {size}")
    
    # 부분 데이터셋을 선택
    input_data_batch = input_data[:size]
    decoder_input_data_batch = decoder_input_data[:size]
    decoder_target_data_batch = decoder_target_data[:size]

    # 모델 학습
    history = model.fit(
    [input_data_batch, decoder_input_data_batch], decoder_target_data_batch,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=[ 
        tf.keras.callbacks.ModelCheckpoint(f'seq2seq_model_best_epoch{epochs}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=True, verbose=1)
    ],
    verbose=1
)


    # 학습된 모델 저장
    model.save(f'seq2seq_model_50000_{size}_reviewed.h5')
    print(f"모델 학습 및 저장 완료: seq2seq_model_50000_{size}_reviewed.h5")
