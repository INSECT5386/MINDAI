import sys
import re
import random
import pickle
import numpy as np
import tensorflow as tf
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QSlider, QComboBox, QLabel
from PySide6.QtCore import Qt
import os
import webbrowser  # webbrowser 라이브러리 추가

# 인사 패턴 및 응답
greetings = [r"\b안녕\b", r"\b안녕하세요\b", r"\b반가워\b", r"\b하이\b", r"\b잘 지내\b"]
greeting_responses = ["안녕하세요! 😊", "반갑습니다!", "안녕! 좋은 하루 보내!", "하이~ 뭐 도와줄까?"]

name_questions = [r"\b이름이 뭐야\b", r"\b너 누구야\b", r"\b너의 이름은\b", r"\b너 뭐야\b"]
name_responses = ["내 이름은 MIND AI이야!", "난 챗봇 MIND이야, 반가워!", "MIND AI이라고 불러줘! 😊"]

def load_model(model_name):
    global tokenizer

    # PyInstaller로 빌드된 경우에는 _MEIPASS 사용
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath('.')  # 개발 환경에서는 현재 디렉토리 경로 사용

    # 모델 파일 경로 설정
    if model_name == "98000 모델":
        model_path = os.path.join(base_path, "seq2seq_model_98000.h5")
    elif model_name == "90000 모델":
        model_path = os.path.join(base_path, "seq2seq_model_90000.h5")
    elif model_name == "50000 모델":
        model_path = os.path.join(base_path, "seq2seq_model_50000.h5")
    else:
        raise ValueError(f"{model_name}는 지원되지 않는 모델입니다.")

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 토크나이저 로드
    tokenizer_path = os.path.join(base_path, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_model("98000 모델")

start_token = "<start>"
end_token = "<end>"

# 인코더 모델 생성
encoder_inputs = model.input[0]
encoder_embedding = model.layers[2]
encoder_gru = model.layers[4]
encoder_outputs, state_h = encoder_gru(encoder_embedding(encoder_inputs))
encoder_model = tf.keras.Model(encoder_inputs, [encoder_outputs, state_h])

# 디코더 모델 생성
decoder_inputs = model.input[1]
decoder_embedding = model.layers[3]
decoder_state_input_h = tf.keras.Input(shape=(136,))
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_gru = model.layers[5]
decoder_outputs, decoder_state_h = decoder_gru(decoder_embedded, initial_state=decoder_state_input_h)
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model([decoder_inputs, decoder_state_input_h], [decoder_outputs, decoder_state_h])

# 인사 체크 함수
def is_greeting(text):
    return any(re.search(pattern, text.lower()) for pattern in greetings)

def is_name_question(text):
    return any(re.search(pattern, text.lower()) for pattern in name_questions)

# 구글 검색 함수
def google_search(query):
    search_url = f"https://www.google.com/search?q={query}"
    webbrowser.open(search_url)  # 기본 웹 브라우저로 구글 검색을 엽니다.

def chatbot_response(user_input, temperature=0.7):
    if "검색" in user_input:  # '검색'이라는 단어가 입력되면 구글 검색 실행
        query = user_input.replace("검색", "").strip()
        google_search(query)
        return "구글에서 검색 중입니다..."
    elif is_greeting(user_input):
        return random.choice(greeting_responses)
    elif is_name_question(user_input):
        return random.choice(name_responses)

    response = chat_with_model(user_input, temperature)
    return response

# Seq2Seq 모델을 사용한 채팅 응답 함수
def chat_with_model(input_text, temperature):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=40, padding="post")

    encoder_output, state_h = encoder_model.predict(input_seq)
    target_seq = np.array([[tokenizer.word_index[start_token]]])
    stop_condition = False
    decoded_sentence = ""
    prev_words = []

    max_output_length = 58
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, state_h])
        preds = np.asarray(output_tokens[0, -1, :]).astype("float64")
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        sampled_token_index = np.random.choice(len(preds), p=preds)
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")

        if not sampled_word or sampled_word in prev_words:
            continue

        prev_words.append(sampled_word)
        if len(prev_words) > 3:
            prev_words.pop(0)

        if sampled_word == end_token or len(decoded_sentence.split()) >= max_output_length:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word

        target_seq = np.array([[sampled_token_index]])
        state_h = h

    return decoded_sentence.strip()

# PySide6 GUI 클래스
class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot Shell")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()

        self.model_selector = QComboBox(self)
        self.model_selector.addItem("98000 모델")
        self.model_selector.addItem("90000 모델")
        self.model_selector.addItem("50000 모델")
        self.model_selector.currentTextChanged.connect(self.change_model)

        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("background-color: #1e1e1e; color: white; padding: 10px; border-radius: 5px;")

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("메시지를 입력하세요...")
        self.text_input.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("보내기", self)
        self.send_button.clicked.connect(self.send_message)

        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setMinimum(10)
        self.temperature_slider.setMaximum(100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        self.temperature_label = QLabel(f"Temperature: {self.temperature_slider.value() / 100:.2f}", self)

        layout.addWidget(QLabel("모델 선택:"))
        layout.addWidget(self.model_selector)
        layout.addWidget(self.temperature_label)
        layout.addWidget(self.temperature_slider)
        layout.addWidget(self.chat_area)
        layout.addWidget(self.text_input)
        layout.addWidget(self.send_button)

        self.setLayout(layout)
        self.apply_dark_mode()

    def send_message(self):
        user_message = self.text_input.text().strip()
        if not user_message:
            return

        self.display_message(f"You: {user_message}\n", "user")
        response = chatbot_response(user_message, self.temperature_slider.value() / 100)
        self.display_message(f"마음이: {response}\n", "bot")
        self.text_input.clear()

    def display_message(self, message, sender):
        self.chat_area.append(message)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def update_temperature(self):
        temp = self.temperature_slider.value() / 100
        self.temperature_label.setText(f"Temperature: {temp:.2f}")

    def change_model(self, model_name):
        global model, tokenizer
        model, tokenizer = load_model(model_name)

    def apply_dark_mode(self):
        self.setStyleSheet(""" 
            QWidget { background-color: #2b2b2b; color: white; }
            QLineEdit, QPushButton { background-color: #444444; color: white; border: 1px solid #888888; }
            QLabel { color: white; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
