
import os
import sys
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QDateEdit, QMessageBox, QTimeEdit)
from PyQt6.QtCore import QDate, QTime
from PyQt6.QtGui import QPixmap
from fpdf import FPDF

class FirearmDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
     
    def initUI(self):
        self.setWindowTitle('Gunalert')
        self.setFixedHeight(280)
        
        # Layout principal
        main_layout = QVBoxLayout()
        
        # Layout de la información del caso
        layout_0 = QHBoxLayout()
        layout_1 = QHBoxLayout()
        layout_2 = QHBoxLayout()
        layout_3 = QHBoxLayout()
        
        # Layout horizontal para el campo de audio y el botón
        audio_layout = QHBoxLayout()
        
        # Añadir botones de análisis y exportar al layout
        btn_layout = QHBoxLayout()
        
        # Logo en la interfaz
        image_logo = QLabel()
        pixmap = QPixmap("LOGO1.png") 
        image_logo.setPixmap(pixmap)
        image_logo.setMaximumWidth(60)
        image_logo.setMaximumHeight(120)
        image_logo.setScaledContents(True)

        # Campos de entrada
        self.case_name = QLineEdit(self)
        self.date = QDateEdit()
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setMaximumDate(QDate.currentDate())
        self.date.setCalendarPopup(True)
        self.location = QLineEdit(self)
        self.criminologist = QLineEdit(self)
        self.time = QTimeEdit()
        self.time.setDisplayFormat("HH:mm")
        self.crime_type = QLineEdit(self)
        self.audio_path = QLineEdit(self)
        self.audio_path.setReadOnly(True)
        
        # Botones
        audio_btn = QPushButton('Cargar Audio', self)
        audio_btn.clicked.connect(self.open_file)

        analyze_btn = QPushButton('Análisis', self)
        analyze_btn.clicked.connect(self.run_analysis)

        export_btn = QPushButton('Exportar PDF', self)
        export_btn.clicked.connect(self.export_pdf)

        # Añadir campos de entrada a los distintos layout
        layout_0.addWidget(QLabel('Información del caso:'))
        layout_0.addWidget(image_logo)

        layout_1.addWidget(QLabel('Nombre del Caso:'))
        layout_1.addWidget(self.case_name)
        layout_1.addWidget(QLabel('Fecha:'))
        layout_1.addWidget(self.date)

        layout_2.addWidget(QLabel('Criminólogo a cargo:'))
        layout_2.addWidget(self.criminologist)
        layout_2.addWidget(QLabel('Hora:'))
        layout_2.addWidget(self.time)

        layout_3.addWidget(QLabel('Tipo de Delito:'))
        layout_3.addWidget(self.crime_type)
        layout_3.addWidget(QLabel('Lugar:'))
        layout_3.addWidget(self.location)
        
        audio_layout.addWidget(self.audio_path)
        audio_layout.addWidget(audio_btn)
        
        btn_layout.addWidget(analyze_btn)
        btn_layout.addWidget(export_btn)

        main_layout.addLayout(layout_0)
        main_layout.addLayout(layout_1)
        main_layout.addLayout(layout_2)
        main_layout.addLayout(layout_3)
        main_layout.addLayout(audio_layout)
        main_layout.addLayout(btn_layout)
        
        # Establecer layout principal
        self.setLayout(main_layout)

        # Inicializar variables para guardar resultados del análisis
        self.principal_components_existing = None
        self.audio_files_matrix1 = None
        self.audio_files_matrix2 = None
        self.principal_components_new_audio = None
        self.most_similar_index = None
        self.similarity_percentage = None
        self.pca = None

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo de Audio", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.audio_path.setText(file_path)

    def run_analysis(self):
        # Obtener la ruta del archivo de audio
        audio_path = self.audio_path.text()
        if not audio_path:
            QMessageBox.warning(self, "Advertencia", "Por favor, cargue un archivo de audio.")
            return

        # Directorios de las matrices
        matrix1_directory = '/Users/cesarponce/Desktop/CAPSTON/samples_estandarizados/9mm'
        matrix2_directory = '/Users/cesarponce/Desktop/CAPSTON/samples_estandarizados/fusil556'

        # Realizar el análisis
        self.perform_analysis(matrix1_directory, matrix2_directory, audio_path)
    
    def perform_analysis(self, matrix1_directory, matrix2_directory, new_audio_path):
        # Cargar audios de las matrices
        self.audio_files_matrix1 = load_audio_files(matrix1_directory)
        self.audio_files_matrix2 = load_audio_files(matrix2_directory)

        # Limitar a 10 audios por matriz
        self.audio_files_matrix1 = self.audio_files_matrix1[:10]
        self.audio_files_matrix2 = self.audio_files_matrix2[:10]

        # Crear matrices de características
        feature_matrix1 = create_feature_matrix(self.audio_files_matrix1)
        feature_matrix2 = create_feature_matrix(self.audio_files_matrix2)

        # Combinar las matrices de características
        combined_feature_matrix = np.vstack((feature_matrix1, feature_matrix2))

        # Extraer características del nuevo audio
        new_audio_features = extract_features(new_audio_path)
        new_audio_features = new_audio_features.reshape(1, -1)  # Reshape para una única muestra

        # Aplicar PCA a la matriz combinada
        combined_feature_matrix_with_new_audio = np.vstack((combined_feature_matrix, new_audio_features))
        principal_components, scaler, self.pca = apply_pca(combined_feature_matrix_with_new_audio)

        # Separar las componentes principales
        self.principal_components_existing = principal_components[:-1]
        self.principal_components_new_audio = principal_components[-1].reshape(1, -1)

        # Calcular la similitud
        self.most_similar_index, self.similarity_percentage = calculate_similarity(self.principal_components_existing, self.principal_components_new_audio)

        # Visualizar los resultados en 2D
        visualize_pca_2d(self.principal_components_existing, self.audio_files_matrix1, self.audio_files_matrix2, self.principal_components_new_audio, self.most_similar_index, self.similarity_percentage)
        
        # Visualizar los resultados en 3D
        visualize_pca_3d(self.principal_components_existing, self.audio_files_matrix1, self.audio_files_matrix2, self.principal_components_new_audio, self.most_similar_index, self.similarity_percentage)
        
        # Plot scree plot
        plot_scree(self.pca)

        QMessageBox.information(self, "Análisis", "Análisis del audio completado.")

    def export_pdf(self):
        # Verificar si se ha realizado un análisis
        if self.principal_components_existing is None:
            QMessageBox.warning(self, "Advertencia", "Por favor, realice un análisis antes de exportar.")
            return

        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Información del caso
        pdf.cell(200, 10, txt="Informe de Análisis de Audio", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Nombre del Caso: {self.case_name.text()}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Fecha: {self.date.text()}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Criminólogo a cargo: {self.criminologist.text()}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Hora: {self.time.text()}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Tipo de Delito: {self.crime_type.text()}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Lugar: {self.location.text()}", ln=True, align='L')

        # Resultados del análisis
        pdf.cell(200, 10, txt=f"Audio más similar: {os.path.basename((self.audio_files_matrix1 + self.audio_files_matrix2)[self.most_similar_index])}", ln=True, align='L')
        pdf.cell(200, 10, txt=f"Porcentaje de similitud: {self.similarity_percentage:.2f}%", ln=True, align='L')

        # Guardar gráficos como imágenes y añadir al PDF
        self.save_plots()

        pdf.image("pca_2d.png", x=10, y=None, w=180)
        pdf.add_page()
        pdf.image("pca_3d.png", x=10, y=None, w=180)
        pdf.add_page()
        pdf.image("scree_plot.png", x=10, y=None, w=180)

        # Guardar PDF
        pdf_output_path = QFileDialog.getSaveFileName(self, "Guardar PDF", "", "PDF Files (*.pdf)")
        if pdf_output_path[0]:
            pdf.output(pdf_output_path[0])
            QMessageBox.information(self, "Exportar", "Resultados exportados a PDF.")

    def save_plots(self):
        # Guardar gráfico PCA 2D
        plt.figure(figsize=(10, 7))
        colors = ['grey'] * len(self.audio_files_matrix1) + ['k'] * len(self.audio_files_matrix2)
        plt.scatter(self.principal_components_existing[:len(self.audio_files_matrix1), 0], self.principal_components_existing[:len(self.audio_files_matrix1), 1], label='9mm', color='grey')
        plt.scatter(self.principal_components_existing[len(self.audio_files_matrix1):, 0], self.principal_components_existing[len(self.audio_files_matrix1):, 1], label='fusil556', color='k', marker='+')
        plt.scatter(self.principal_components_new_audio[0, 0], self.principal_components_new_audio[0, 1], color='k', label='Muestra', marker='x')
        plt.annotate('Muestra', (self.principal_components_new_audio[0, 0], self.principal_components_new_audio[0, 1]), color='k')
        most_similar_file = os.path.basename((self.audio_files_matrix1 + self.audio_files_matrix2)[self.most_similar_index])
        plt.annotate(f'{self.similarity_percentage:.2f}% similar a {most_similar_file}', 
                     xy=(self.principal_components_new_audio[0, 0], self.principal_components_new_audio[0, 1]), 
                     xytext=(0.95, 0.95), 
                     textcoords='axes fraction',
                     bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.5'),
                     ha='right', 
                     va='top',
                     color='black')
        plt.title('Componentes Principales (2D)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend(loc='upper left')
        plt.savefig("pca_2d.png")
        plt.close()

        # Guardar gráfico PCA 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.principal_components_existing[:len(self.audio_files_matrix1), 0], self.principal_components_existing[:len(self.audio_files_matrix1), 1], self.principal_components_existing[:len(self.audio_files_matrix1), 2], label='9mm', color='grey')
        ax.scatter(self.principal_components_existing[len(self.audio_files_matrix1):, 0], self.principal_components_existing[len(self.audio_files_matrix1):, 1], self.principal_components_existing[len(self.audio_files_matrix1):, 2], label='fusil556', color='k', marker='+')
        ax.scatter(self.principal_components_new_audio[0, 0], self.principal_components_new_audio[0, 1], self.principal_components_new_audio[0, 2], color='k', label='Muestra', marker='x')
        ax.text(self.principal_components_new_audio[0, 0], self.principal_components_new_audio[0, 1], self.principal_components_new_audio[0, 2], 'Muestra', color='k')
        most_similar_file = os.path.basename((self.audio_files_matrix1 + self.audio_files_matrix2)[self.most_similar_index])
        ax.text2D(0.95, 0.95, f'{self.similarity_percentage:.2f}% similar a {most_similar_file}', 
                  transform=ax.transAxes,
                  bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.5'),
                  ha='right', 
                  va='top',
                  color='black')
        ax.set_title('Componentes Principal (3D)')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        plt.legend(loc='upper left')
        plt.savefig("pca_3d.png")
        plt.close()

        # Guardar scree plot
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio) * 100
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='black', alpha=0.5, align='center')
        for i, v in enumerate(explained_variance_ratio):
            ax[0].text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        ax[0].set_xlabel('Componentes Principales')
        ax[0].set_ylabel('Porcentaje de varianza explicada')
        ax[0].set_title('Porcentaje de varianza explicada por cada componente')
        ax[0].set_ylim(0, 1)
        ax[0].set_xticks(range(1, len(explained_variance_ratio) + 1))
        ax[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='black')
        for i, v in enumerate(cumulative_variance):
            ax[1].text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='black')
        ax[1].set_xlabel('Componentes Principales')
        ax[1].set_ylabel('Porcentaje de varianza acumulada')
        ax[1].set_title('Porcentaje de varianza explicada acumulada')
        ax[1].set_ylim(0, 100)
        ax[1].set_xticks(range(1, len(cumulative_variance) + 1))
        plt.tight_layout()
        plt.savefig("scree_plot.png")
        plt.close()

# Funciones de análisis (reutilizadas del código anterior)
def load_audio_files(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            audio_files.append(os.path.join(directory, filename))
    return audio_files

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def create_feature_matrix(audio_files):
    feature_list = []
    for file in audio_files:
        features = extract_features(file)
        feature_list.append(features)
    feature_matrix = np.array(feature_list)
    return feature_matrix

def apply_pca(feature_matrix, n_components=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)
    return principal_components, scaler, pca

def calculate_similarity(principal_components_existing, new_audio_principal_component):
    new_audio_principal_component = new_audio_principal_component.ravel()
    distances = [distance.euclidean(pc, new_audio_principal_component) for pc in principal_components_existing]
    min_distance = min(distances)
    most_similar_index = distances.index(min_distance)
    similarity_percentage = 100 * (1 - min_distance / max(distances))

    print("Distancias Euclidianas entre el nuevo audio y los audios existentes:")
    for i, dist in enumerate(distances):
        print(f"Audio {i + 1}: {dist:.2f}")

    return most_similar_index, similarity_percentage

def visualize_pca_2d(principal_components, audio_files_matrix1, audio_files_matrix2, new_audio_principal_component, most_similar_index, similarity_percentage):
    plt.figure(figsize=(10, 7))
    colors = ['grey'] * len(audio_files_matrix1) + ['k'] * len(audio_files_matrix2)
    plt.scatter(principal_components[:len(audio_files_matrix1), 0], principal_components[:len(audio_files_matrix1), 1], label='9mm', color='grey')
    plt.scatter(principal_components[len(audio_files_matrix1):, 0], principal_components[len(audio_files_matrix1):, 1], label='fusil556', color='k', marker='+')
    plt.scatter(new_audio_principal_component[0, 0], new_audio_principal_component[0, 1], color='k', label='Muestra', marker='x')
    plt.annotate('Muestra', (new_audio_principal_component[0, 0], new_audio_principal_component[0, 1]), color='k')
    most_similar_file = os.path.basename((audio_files_matrix1 + audio_files_matrix2)[most_similar_index])
    plt.annotate(f'{similarity_percentage:.2f}% similar a {most_similar_file}', 
                 xy=(new_audio_principal_component[0, 0], new_audio_principal_component[0, 1]), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.5'),
                 ha='right', 
                 va='top',
                 color='black')
    plt.title('Componentes Principales (2D)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(loc='upper left')
    plt.show()

def visualize_pca_3d(principal_components, audio_files_matrix1, audio_files_matrix2, new_audio_principal_component, most_similar_index, similarity_percentage):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['grey'] * len(audio_files_matrix1) + ['k'] * len(audio_files_matrix2)
    ax.scatter(principal_components[:len(audio_files_matrix1), 0], principal_components[:len(audio_files_matrix1), 1], principal_components[:len(audio_files_matrix1), 2], label='9mm', color='grey')
    ax.scatter(principal_components[len(audio_files_matrix1):, 0], principal_components[len(audio_files_matrix1):, 1], principal_components[len(audio_files_matrix1):, 2], label='fusil556', color='k', marker='+')
    ax.scatter(new_audio_principal_component[0, 0], new_audio_principal_component[0, 1], new_audio_principal_component[0, 2], color='k', label='Muestra', marker='x')
    ax.text(new_audio_principal_component[0, 0], new_audio_principal_component[0, 1], new_audio_principal_component[0, 2], 'Muestra', color='k')
    most_similar_file = os.path.basename((audio_files_matrix1 + audio_files_matrix2)[most_similar_index])
    ax.text2D(0.95, 0.95, f'{similarity_percentage:.2f}% similar a {most_similar_file}', 
              transform=ax.transAxes,
              bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.5'),
              ha='right', 
              va='top',
              color='black')
    ax.set_title('Componentes Principal (3D)')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    plt.legend(loc='upper left')
    plt.show()

def plot_scree(pca):
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color='black', alpha=0.5, align='center')
    for i, v in enumerate(explained_variance_ratio):
        ax[0].text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='black')
    ax[0].set_xlabel('Componentes Principales')
    ax[0].set_ylabel('Porcentaje de varianza explicada')
    ax[0].set_title('Porcentaje de varianza explicada por cada componente')
    ax[0].set_ylim(0, 1)
    ax[0].set_xticks(range(1, len(explained_variance_ratio) + 1))
    ax[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='black')
    for i, v in enumerate(cumulative_variance):
        ax[1].text(i + 1, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='black')
    ax[1].set_xlabel('Componentes Principales')
    ax[1].set_ylabel('Porcentaje de varianza acumulada')
    ax[1].set_title('Porcentaje de varianza explicada acumulada')
    ax[1].set_ylim(0, 100)
    ax[1].set_xticks(range(1, len(cumulative_variance) + 1))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FirearmDetectionApp()
    ex.show()
    sys.exit(app.exec())


