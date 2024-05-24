import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import pickle
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests

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

# Função para capturar várias imagens de uma nova pessoa e calcular a codificação média
def capture_and_encode_faces(video_capture, name, sus_card, num_images=200, pause_duration=0.1):
    encodings = []
    image_count = 0

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        while image_count < num_images:
            # Capturar um único frame de vídeo
            frame = video_capture.read()[1]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face_image = frame[y:y+h, x:x+w]

                    face_encoding = encode_face(face_image)
                    if face_encoding is not None:
                        encodings.append(face_encoding)
                        image_count += 1
                        progress_bar["value"] = (image_count / num_images) * 100
                        root.update_idletasks()
                        time.sleep(pause_duration)  # Pausa para permitir mudanças de posição
                        break

            cv2.imshow('Capturando Imagens', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if encodings:
        average_encoding = np.mean(encodings, axis=0)
        save_new_face(name, sus_card, average_encoding)
        display_message("Captura de imagens concluída. Obrigado por olhar para a câmera.")
        return average_encoding
    display_message("Não foi possível capturar imagens suficientes. Por favor, tente novamente.")
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

# Função para exibir mensagens na interface gráfica
def display_message(message):
    text_area.insert(tk.END, message + "\n")
    text_area.see(tk.END)
    root.update_idletasks()

# Função para iniciar o reconhecimento facial
def start_recognition():
    known_face_encodings, known_face_names, known_face_sus_cards = load_known_faces()

    # Iniciar a captura de vídeo
    # Substitua a URL pela URL fornecida pelo IP Webcam
    url = "http://192.168.3.59:8080/video"
    video_capture = cv2.VideoCapture(url)

    with mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
        recognized = False
        while not recognized:
            # Capturar um único frame de vídeo
            frame = video_capture.read()[1]

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
                        # Solicitar o nome e cartão do SUS se a pessoa não for reconhecida
                        entry_name.delete(0, tk.END)
                        entry_sus_card.delete(0, tk.END)
                        display_message("Nome não reconhecido. Por favor, insira seu nome e cartão do SUS.")
                        root.wait_variable(user_input_var)
                        name = entry_name.get()
                        sus_card = entry_sus_card.get()

                        # Capturar várias imagens da nova pessoa e calcular a codificação média
                        average_encoding = capture_and_encode_faces(video_capture, name, sus_card, num_images=200)
                        if average_encoding is not None:
                            known_face_encodings.append(average_encoding)
                            known_face_names.append(name)
                            known_face_sus_cards.append(sus_card)
                    else:
                        # Mostrar uma mensagem de boas-vindas e os dados do paciente na área de texto
                        display_message(f"Bem-vindo, {name}! Seu cartão SUS é: {sus_card}")
                        recognized = True

                    # Mostrar o nome e cartão do SUS na imagem capturada
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"{name} - SUS: {sus_card}", (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

            # Converter a imagem do OpenCV para um formato suportado pelo Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (800, 600))  # Redimensionar para uma exibição adequada

            # Converter a imagem para o formato do Pillow e exibi-la no widget Label
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img)

            # Atualizar a imagem na label
            label_camera.img_tk = img_tk  # Mantém uma referência para evitar que o garbage collector a colete
            label_camera.config(image=img_tk)

            # Mostrar a imagem resultante
            root.update()

            # Pressionar 'q' para sair do loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar a captura de vídeo apenas após pressionar 'q' e fechar as janelas
    video_capture.release()
    cv2.destroyAllWindows()

# Função chamada quando o botão "Salvar" é pressionado
def save_user_input():
    user_input_var.set(True)

# Criar a interface gráfica
root = tk.Tk()
root.title("Reconhecimento Facial SUS")
root.geometry("900x700")  # Definindo o tamanho da janela

# Configurações de cores e estilo
bg_color = "#F0E5DA"  # Cor de fundo principal
label_color = "#6D214F"  # Cor do texto dos rótulos
entry_color = "#FFFFFF"  # Cor do fundo dos campos de entrada
button_color = "#6D214F"  # Cor de fundo dos botões
button_text_color = "#FFFFFF"  # Cor do texto dos botões
text_area_color = "#FFFFFF"  # Cor de fundo da área de texto
text_color = "#6D214F"  # Cor do texto da área de texto
progress_bar_color = "#6D214F"  # Cor da barra de progresso

root.configure(bg=bg_color)  # Define a cor de fundo da janela principal

label_name = tk.Label(root, text="Nome:", bg=bg_color, fg=label_color, font=("Helvetica", 12, "bold"))  # Configuração do rótulo "Nome"
label_name.pack(pady=(20, 5))
entry_name = tk.Entry(root, bg=entry_color, fg=label_color, font=("Helvetica", 12))  # Configuração do campo de entrada para o nome
entry_name.pack(ipady=5, padx=10)

label_sus_card = tk.Label(root, text="Cartão SUS:", bg=bg_color, fg=label_color, font=("Helvetica", 12, "bold"))  # Configuração do rótulo "Cartão SUS"
label_sus_card.pack(pady=(10, 5))
entry_sus_card = tk.Entry(root, bg=entry_color, fg=label_color, font=("Helvetica", 12))  # Configuração do campo de entrada para o cartão SUS
entry_sus_card.pack(ipady=5, padx=10)

button_save = tk.Button(root, text="Salvar", command=save_user_input, bg=button_color, fg=button_text_color, font=("Helvetica", 12, "bold"))  # Configuração
# Configuração do botão "Salvar"
button_save.pack(pady=10)

user_input_var = tk.BooleanVar()

button_start = tk.Button(root, text="Iniciar Reconhecimento Facial", command=start_recognition, bg=button_color, fg=button_text_color, font=("Helvetica", 14, "bold"))  # Configuração do botão "Iniciar Reconhecimento Facial"
button_start.pack(pady=10)

text_area = tk.Text(root, width=40, height=8, bg=text_area_color, fg=text_color, font=("Helvetica", 12))  # Configuração da área de texto
text_area.pack(pady=10)

# Barra de progresso
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", style="TProgressbar", maximum=100)  # Configuração da barra de progresso
progress_bar.pack(pady=10)

# Criar um label para exibir a câmera
label_camera = tk.Label(root)
label_camera.pack(pady=10)

root.mainloop()

