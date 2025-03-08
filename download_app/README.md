```bash
import sys
import re
import random
import pickle
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import scrolledtext
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

  
    if model_name == "기본 모델":
        model_path = os.path.join(base_path, "seq2seq_model_90000.h5")
    elif model_name == "고급 모델":
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

class ChatWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chatbot Shell")
        self.geometry("600x500")
        self.config(bg="#f4f4f4")

        # 상단 제목 표시
        self.header_label = tk.Label(self, text="마음이 챗봇", font=("Helvetica", 18, "bold"), fg="#2c3e50", bg="#f4f4f4")
        self.header_label.pack(pady=10)

        # 모델 선택
        self.model_selector = tk.StringVar(self)
        self.model_selector.set("기본 모델")
        self.model_menu = tk.OptionMenu(self, self.model_selector, "기본 모델", "고급 모델", command=self.change_model)
        self.model_menu.config(width=15, font=("Helvetica", 12), bg="#3498db", fg="white")
        self.model_menu.pack(pady=10)

        # 대화창 (스크롤텍스트)
        self.chat_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=15, width=70, state=tk.DISABLED, bg="#1e1e1e", fg="white", font=("Courier", 12))
        self.chat_area.pack(padx=20, pady=10)

        # 사용자 입력 필드
        self.text_input = tk.Entry(self, width=50, font=("Helvetica", 12))
        self.text_input.pack(pady=10)
        self.text_input.bind("<Return>", self.send_message)

        # 전송 버튼
        self.send_button = tk.Button(self, text="보내기", command=self.send_message, font=("Helvetica", 12), bg="#2ecc71", fg="white", width=10)
        self.send_button.pack(pady=5)

        # Temperature 슬라이더
        self.temperature_slider = tk.Scale(self, from_=0, to=100, orient="horizontal", length=200, label="Temperature", font=("Helvetica", 12))
        self.temperature_slider.set(70)
        self.temperature_slider.pack(pady=10)

    def change_model(self, model_name):
        global model, tokenizer
        model, tokenizer = load_model(model_name)
        self.display_message(f"모델이 '{model_name}'으로 변경되었습니다.\n", "bot")

    def send_message(self, event=None):
        user_message = self.text_input.get().strip()
        if not user_message:
            return

        if user_message.lower() == "clear":
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.delete(1.0, tk.END)
            self.chat_area.config(state=tk.DISABLED)
            self.display_message("대화 내용이 초기화되었습니다.\n", "bot")
            self.text_input.delete(0, tk.END)
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
            self.text_input.delete(0, tk.END)
            return

        if "검색" in user_message:  # '검색'이 포함된 입력 처리
            search_query = user_message.replace("검색", "").strip()  # '검색' 단어 제거하고 검색어 추출
            if search_query:  # 검색어가 비어 있지 않으면
                self.display_message(f"검색어 '{search_query}'가 입력되었습니다.\n", "user")  # 대화창에 사용자 입력 표시
                search_url = f"https://www.google.com/search?q={search_query}"
                webbrowser.open(search_url)  # 구글에서 검색
                self.display_message(f"구글에서 '{search_query}' 검색 중...\n", "bot")
            else:
                self.display_message("검색어를 입력해주세요.\n", "bot")
            self.text_input.delete(0, tk.END)
            return


        # 사용자 메시지 출력
        self.display_message(f"{user_message}\n", "user")

        # 챗봇 응답 출력
        response = chatbot_response(user_message, self.temperature_slider.get() / 100)
        self.display_message(f"{response}\n", "bot")  # '마음이:'는 이미 display_message에서 처리됨
        
        self.text_input.delete(0, tk.END)

    def display_message(self, message, sender):
        self.chat_area.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_area.insert(tk.END, f"You: {message}\n", "user")
        elif sender == "bot":
            self.chat_area.insert(tk.END, f"마음이: {message}\n", "bot")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.yview(tk.END)


if __name__ == "__main__":
    app = ChatWindow()
    app.mainloop()
