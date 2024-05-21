import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import pickle
import time
import tkinter as tk
from tkinter import messagebox

# Inicializar mediapipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Função para carregar dados do banco de dados
def load_known_faces():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, sus_card, face_encoding FROM people")
    rows = cursor.fetchall()
    conn.close()

    known_face_encodings = []
    known_face_names = []
    known_face_sus_cards = []

    for row in rows:
        name, sus_card, face_encoding = row
        known_face_names.append(name)
        known_face_sus_cards.append(sus_card)
        known_face_encodings.append(pickle.loads(face_encoding))

    return known_face_encodings, known_face_names, known_face_sus_cards

# Função para salvar novos dados no banco de dados
def save_new_face(name, sus_card, face_encoding):
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO people (name, sus_card, face_encoding) VALUES (?, ?, ?)",
                   (name, sus_card, sqlite3.Binary(pickle.dumps(face_encoding))))
    conn.commit()
    conn.close()

# Função para codificar uma face usando o mediapipe face mesh
def encode_face(face_image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_encoding = []
            for landmark in face_landmarks.landmark:
                face_encoding.extend([landmark.x, landmark.y, landmark.z])
            return np.array(face_encoding)
        return None

# Função para reconhecer uma face e retornar o nome e cartão SUS
def recognize_face(face_encoding, known_face_encodings, known_face_names, known_face_sus_cards):
    threshold = 0.7  # Limiar de confiança
    matches = [np.linalg.norm(face_encoding - known_face_encoding) < threshold for known_face_encoding in known_face_encodings]
    name = "Desconhecido"
    sus_card = None

    if any(matches):
        best_match_index = np.argmin([np.linalg.norm(face_encoding - known_face_encoding) for known_face_encoding in known_face_encodings])
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            sus_card = known_face_sus_cards[best_match_index]

    return name, sus_card

# Função para iniciar o reconhecimento facial
def start_recognition():
    # Carregar dados das faces conhecidas
    known_face_encodings, known_face_names, known_face_sus_cards = load_known_faces()

    # Iniciar a captura de vídeo
    video_capture = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            # Capturar um único frame de vídeo
            ret, frame = video_capture.read()
            
            if not ret:
                break

            # Converter a imagem para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar a imagem e detectar faces
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
                    
                    # Extraindo a caixa delimitadora
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face_image = frame[y:y+h, x:x+w]

                    # Codificar a face usando mediapipe
                    face_encoding = encode_face(face_image)

                    if face_encoding is None:
                        continue

                    # Verificar se a face é de uma pessoa conhecida
                    name, sus_card = recognize_face(face_encoding, known_face_encodings, known_face_names, known_face_sus_cards)

                    if name == "Desconhecido":
                        # Mostrar uma mensagem de aviso se a pessoa não for reconhecida
                        messagebox.showwarning("Alerta", "Pessoa não reconhecida!")

                    else:
                        # Mostrar uma mensagem de boas-vindas e os dados do paciente
                        messagebox.showinfo("Bem-vindo", f"Bem-vindo, {name}! Seu cartão SUS é: {sus_card}")

                    # Encerrar o loop quando uma pessoa reconhecida for encontrada
                    break

            # Mostrar a imagem resultante
            cv2.imshow('Video', frame)

            # Pressionar 'q' para sair do loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar a captura de vídeo apenas após pressionar 'q' e fechar as janelas
    video_capture.release()
    cv2.destroyAllWindows()

# Criar a interface Tkinter
root = tk.Tk()
root.title("Reconhecimento Facial")

# Função para iniciar o reconhecimento facial ao pressionar o botão
def start_recognition_command():
    start_recognition()

# Criar e posicionar o botão "Iniciar"
start_button = tk.Button(root, text="Iniciar Reconhecimento", command=start_recognition_command)
start_button.pack()

# Executar o loop principal do Tkinter
root.mainloop()
