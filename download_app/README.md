import sys
import re
import random
import pickle
import numpy as np
import tensorflow as tf
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QSlider, QComboBox, QLabel
from PySide6.QtCore import Qt
import os
import webbrowser

greetings = [r"\b안녕\b", r"\b안녕하세요\b", r"\b반가워\b", r"\b하이\b", r"\b잘 지내\b"]
greeting_responses = ["안녕하세요! 😊", "반갑습니다!", "안녕! 좋은 하루 보내!", "하이~ 뭐 도와줄까?"]

name_questions = [r"\b이름이 뭐야\b", r"\b너 누구야\b", r"\b너의 이름은\b", r"\b너 뭐야\b"]
name_responses = ["내 이름은 마음이야!", "난 챗봇 마음이야, 반가워!", "마음이라고 불러줘! 😊"]



def load_model(model_name):
    global tokenizer

   
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  
    else:
        base_path = os.path.abspath('.')  

    
    if model_name == "기본 모델":
        model_path = os.path.join(base_path, "seq2seq_model_50000.h5")
    elif model_name == "고급 모델":
        model_path = os.path.join(base_path, "seq2seq_model_90000.h5")
    elif model_name == "빠른 모델":
        model_path = os.path.join(base_path, "seq2seq_model_98000.h5")
    else:
        raise ValueError("모델 이름이 올바르지 않습니다.")

    
    model = tf.keras.models.load_model(model_path)

   
    tokenizer_path = os.path.join(base_path, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer



model, tokenizer = load_model("기본 모델")

start_token = "<start>"
end_token = "<end>"


encoder_inputs = model.input[0]
encoder_embedding = model.layers[2]
encoder_gru = model.layers[4]
encoder_outputs, state_h = encoder_gru(encoder_embedding(encoder_inputs))
encoder_model = tf.keras.Model(encoder_inputs, [encoder_outputs, state_h])


decoder_inputs = model.input[1]
decoder_embedding = model.layers[3]
decoder_state_input_h = tf.keras.Input(shape=(136,))
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_gru = model.layers[5]
decoder_outputs, decoder_state_h = decoder_gru(decoder_embedded, initial_state=decoder_state_input_h)
decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model([decoder_inputs, decoder_state_input_h], [decoder_outputs, decoder_state_h])


def is_greeting(text):
    return any(re.search(pattern, text.lower()) for pattern in greetings)

def is_name_question(text):
    return any(re.search(pattern, text.lower()) for pattern in name_questions)

def chatbot_response(user_input, temperature=0.7):
    if is_greeting(user_input):
        return random.choice(greeting_responses)
    elif is_name_question(user_input):
        return random.choice(name_responses)
    
    
    response = chat_with_model(user_input, temperature)  
    return response


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


class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot Shell")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()

        
        self.model_selector = QComboBox(self)
        self.model_selector.addItem("기본 모델")
        self.model_selector.addItem("고급 모델")
        self.model_selector.addItem("빠른 모델")
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

        self.model_name = "기본 모델"
        self.model, self.tokenizer = load_model(self.model_name)

    def change_model(self, model_name):
        global model, tokenizer
        model, tokenizer = load_model(model_name) 
        self.model_name = model_name  
        self.display_message(f"모델이 '{model_name}'으로 변경되었습니다.\n", "bot")

    

    def send_message(self):
        user_message = self.text_input.text().strip()
        if not user_message:
            return

        if user_message.lower() == "clear":
            self.chat_area.clear() 
            self.display_message("대화 내용이 초기화되었습니다.\n", "bot")
            self.text_input.clear()
            return
    
        if user_message.lower() in ["/?", "/help"]:
            help_message = (
                "사용법:\n"
                "/? 또는 /help: 사용법을 표시합니다.\n"
                "검색 [검색어]: 구글에서 검색합니다.\n"
                "clear: 대화 내용 초기화\n"
                "일반 메시지를 입력하면 챗봇과 대화할 수 있습니다."
        )
            self.display_message(help_message, "bot")
            self.text_input.clear()
            return

        if "검색" in user_message: 
            search_query = user_message.replace("검색", "").strip() 
            if search_query:  
                search_url = f"https://www.google.com/search?q={search_query}"
                webbrowser.open(search_url)  
                self.display_message(f"구글에서 '{search_query}' 검색 중...\n", "bot")
            else:
                self.display_message("검색어를 입력해주세요.\n", "bot")
            self.text_input.clear()
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
