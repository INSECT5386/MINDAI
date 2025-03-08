### 사용법
##### exe 파일을 다운로드 후, /? 를 입력하면 도움말이 출력 됨
+     if user_message.lower() in ["/?", "/help"]:
        help_message = (
            "사용법:\n"
            "/? 또는 /help: 사용법을 표시합니다.\n"
            "검색 [검색어]: 구글에서 검색합니다.\n"
            "clear: 대화 내용 초기화\n"
            "일반 메시지를 입력하면 챗봇과 대화할 수 있습니다."
### 모델/앱 소개 및 다운로드                                                                                
+ LICENSE를 꼭 읽어 주세요.                                                                                                                                                                                                 
+ 완전한 로컬 모델/앱으로 안심하고 사용하셔도 좋습니다. 만약 문제가 발생하면 바로 삭제하고 저에게 알려주세요.
+ 완전한 로컬이므로 광고가 나오지 않습니다.
+ 앱은 exe 파일입니다. 다른 확장자로는 제작하지 않습니다.
+ 앱은 python 코드를 Pyinstaller를 이용해 exe로 빌드하였습니다.
+ 본 모델은 한국어 모델입니다.
+ 상업적 사용을 제외하면 뭐든 괜찮습니다. 얼마든지 사용해 주세요

#### 모델 다운로드:                                                                                                   
+ [seq2seq_98000.h5](https://drive.google.com/file/d/13jwVJKOXsGiRwoMHI9dASlYEUsulsxs_/view?usp=drive_link)       
+ [seq2seq_90000.h5](https://drive.google.com/file/d/1eCgpFWnyJSX-JgShrnCTm4-LDit_cB0T/view?usp=drive_link)       
+ [seq2seq_50000.h5](https://drive.google.com/file/d/19tm0EH82sRCQQUbho6bLicw80kc1TI0s/view?usp=drive_link)    
+ [Transformer_model_10000.h5](https://drive.google.com/file/d/1befy6rcPYrYmz29tjaV8uk_fTFm-Pm7N/view?usp=drive_link)

#### 토크나이저 다운로드:                                                                                        
+ [Seq2Seq_tokenizer](https://drive.google.com/file/d/1n5vWALf3qYSynzdLu5RtAgZEj9RnaCB8/view?usp=drive_link) 

+ [Transformer_tokenizer](https://drive.google.com/file/d/1e-nNJjgAwC4HSFPEw8JCjsLviaGPgvcG/view?usp=drive_link)        

#### 앱 다운로드:                                                                                                     



### 부탁과 밑밥

+ 독학으로 만든 첫 모델들이라 성능이 낮을 수 있어요.
+ License 제발 읽어 주세요.
+ 노트북으로 딥러닝을 돌리기 때문에 걸리는 시간 대비 모델 성능이 낮을 수 있어요.
+ 모델 파일이 크다보니 구글에서 바이러스 검사를 못한다는 팝업이 나올 수 있습니다.
+ 안심하고 사용해주세요. Github 사용이 서툴어서 넣지 못한 게 있을 수 있지만 공개할 수 있는 모든 것을 공개하고 있어요.
+ Seq2Seq는 모두 같은 토크나이저를 사용합니다. 학습량의 차이가 있을 뿐 입니다.
+ Transformer의 토크나이저는 Seq2Seq와 같지 않습니다.

### 추천하는 것

+ 모델과 토크나이저를 다운로드하여 GUI를 직접 구현하는 것도 좋지만, 함께 공개중인 앱도 사용해 보는 것을 추천합니다.

### 규칙

+ 상업적 사용(돈을 벌기 위한 수단으로 사용)을 허가하지 않습니다. 개인 또는 단체의 비영리 사용만 허용합니다.
### 앱 코드
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
# 인사 패턴 및 응답
greetings = [r"\b안녕\b", r"\b안녕하세요\b", r"\b반가워\b", r"\b하이\b", r"\b잘 지내\b"]
greeting_responses = ["안녕하세요! 😊", "반갑습니다!", "안녕! 좋은 하루 보내!", "하이~ 뭐 도와줄까?"]

name_questions = [r"\b이름이 뭐야\b", r"\b너 누구야\b", r"\b너의 이름은\b", r"\b너 뭐야\b"]
name_responses = ["내 이름은 마음이야!", "난 챗봇 마음이야, 반가워!", "마음이라고 불러줘! 😊"]



def load_model(model_name):
    global tokenizer

    # PyInstaller에서 EXE가 실행될 때, 임시 경로를 사용합니다.
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  # EXE로 실행될 경우
    else:
        base_path = os.path.abspath('.')  # 개발 환경에서 실행될 경우

    # 모델 파일 경로 설정
    if model_name == "기본 모델":
        model_path = os.path.join(base_path, "seq2seq_model_50000.h5")
    elif model_name == "고급 모델":
        model_path = os.path.join(base_path, "seq2seq_model_90000.h5")
    elif model_name == "빠른 모델":
        model_path = os.path.join(base_path, "seq2seq_model_98000.h5")
    else:
        raise ValueError("모델 이름이 올바르지 않습니다.")

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    # 토크나이저 파일 경로 설정
    tokenizer_path = os.path.join(base_path, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


# 초기 모델 설정 (기본 모델)
model, tokenizer = load_model("기본 모델")

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

def chatbot_response(user_input, temperature=0.7):
    if is_greeting(user_input):
        return random.choice(greeting_responses)
    elif is_name_question(user_input):
        return random.choice(name_responses)
    
    # Seq2Seq 모델 사용
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

        # 모델 선택 콤보박스
        self.model_selector = QComboBox(self)
        self.model_selector.addItem("기본 모델")
        self.model_selector.addItem("고급 모델")
        self.model_selector.addItem("빠른 모델")
        self.model_selector.currentTextChanged.connect(self.change_model)

        # 대화창 (QTextEdit 사용)
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
        model, tokenizer = load_model(model_name)  # 모델과 토크나이저 로드
        self.model_name = model_name  # 현재 모델 이름을 저장
        self.display_message(f"모델이 '{model_name}'으로 변경되었습니다.\n", "bot")

    

    def send_message(self):
        user_message = self.text_input.text().strip()
        if not user_message:
            return

        if user_message.lower() == "clear":
            self.chat_area.clear()  # 대화 내용 초기화
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

        if "검색" in user_message:  # '검색'이 포함된 입력 처리
            search_query = user_message.replace("검색", "").strip()  # '검색' 단어 제거하고 검색어 추출
            if search_query:  # 검색어가 비어 있지 않으면
                search_url = f"https://www.google.com/search?q={search_query}"
                webbrowser.open(search_url)  # 구글에서 검색
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
