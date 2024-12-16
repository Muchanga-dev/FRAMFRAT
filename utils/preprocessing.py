# utils/preprocessing.py
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from streamlit_cropper import st_cropper

class ImageProcessor:
    """
    Classe para processar imagens, incluindo rotacionar, ajustar brilho/contraste, gamma,
    equalização de histograma e aplicação de CLAHE.

    Args:
        image (Image.Image): Imagem PIL a ser processada.
    """
    def __init__(self, image: Image.Image):
        self.original_image = image
        self.processed_image = image.copy()

    def rotate_image(self, angle: float):
        """
        Rotaciona a imagem em um ângulo especificado.

        Args:
            angle (float): Ângulo em graus para rotacionar a imagem.
        """
        self.processed_image = self.processed_image.rotate(angle, expand=True)

    def adjust_brightness_contrast(self, brightness: float, contrast: float):
        """
        Ajusta o brilho e o contraste da imagem.

        Args:
            brightness (float): Fator de brilho.
            contrast (float): Fator de contraste.
        """
        enhancer_brightness = ImageEnhance.Brightness(self.processed_image)
        self.processed_image = enhancer_brightness.enhance(brightness)
        enhancer_contrast = ImageEnhance.Contrast(self.processed_image)
        self.processed_image = enhancer_contrast.enhance(contrast)

    def adjust_gamma(self, gamma: float):
        """
        Ajusta a correção gamma da imagem.

        Args:
            gamma (float): Valor de gamma para ajuste.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        self.processed_image = Image.fromarray(cv2.LUT(np.array(self.processed_image), table))

    def equalize_histogram(self):
        """
        Aplica equalização de histograma à imagem para melhorar o contraste.
        """
        image_np = np.array(self.processed_image)
        img_y_cr_cb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(img_y_cr_cb)
        y_eq = cv2.equalizeHist(y)
        img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
        image_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2RGB)
        self.processed_image = Image.fromarray(image_eq)

    def apply_clahe(self, clip_limit: float, tile_grid_size: int):
        """
        Aplica o algoritmo CLAHE para melhorar o contraste localmente.

        Args:
            clip_limit (float): Limite de contraste para recorte.
            tile_grid_size (int): Tamanho da grade para o CLAHE.
        """
        image_np = np.array(self.processed_image)
        img_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l_clahe = clahe.apply(l)
        img_lab_clahe = cv2.merge((l_clahe, a, b))
        image_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
        self.processed_image = Image.fromarray(image_clahe)

    def crop_image(self):
        """
        Permite ao usuário recortar a imagem utilizando uma ferramenta interativa.
        """
        st.warning("Selecione a área de recorte utilizando o retângulo na imagem.")
        cropped_img = st_cropper(
            self.processed_image,
            realtime_update=True,
            box_color="#FF0004",
            aspect_ratio=None
        )
        if cropped_img:
            self.processed_image = cropped_img
            st.success("Imagem recortada com sucesso!")

    def get_processed_image(self) -> Image.Image:
        """
        Retorna a imagem processada.

        Returns:
            Image.Image: Imagem PIL processada.
        """
        return self.processed_image

def render_preprocessing_page():
    """
    Renderiza a página de pré-processamento, permitindo ajustes na imagem antes da detecção.
    """

    st.markdown("<h3 style='text-align: center; color: Green;'> Pré-Processamento de Imagem </h3>", unsafe_allow_html=True)
    
    col1, col2,col3=st.columns([5,1,5])
    uploaded_file = st.session_state.get('uploaded_file', None)
    if uploaded_file is None:
        st.warning("Nenhuma imagem foi carregada. Por favor, faça upload de uma imagem na página inicial.")
        st.session_state.page = 'home'
        st.rerun()
        return

    image = Image.open(uploaded_file).convert('RGB')
    col1.image(image, caption='Imagem Original', use_container_width=True)
    st.session_state['original_image'] = image.copy()

    # Instancia o processador de imagem
    processor = ImageProcessor(image)

    # Opções de processamento na barra lateral
    st.sidebar.subheader("Opções de Ajuste Manual")

    if st.sidebar.toggle("Rotacionar Imagem"):
        angle = st.sidebar.slider("Ângulo de Rotação", -360, 360, 0)
        processor.rotate_image(angle)

    if st.sidebar.toggle("Ajustar Brilho e Contraste"):
        brightness = st.sidebar.slider("Brilho", 0.1, 3.0, 1.0, 0.1)
        contrast = st.sidebar.slider("Contraste", 0.1, 3.0, 1.0, 0.1)
        processor.adjust_brightness_contrast(brightness, contrast)

    if st.sidebar.toggle("Ajustar Gamma"):
        gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
        processor.adjust_gamma(gamma)

    if st.sidebar.toggle("Aplicar Equalização de Histograma"):
        processor.equalize_histogram()

    if st.sidebar.toggle("Aplicar CLAHE"):
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 4.0, 2.0, 0.1)
        tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 16, 8, 1)
        processor.apply_clahe(clip_limit, tile_grid_size)

    if st.sidebar.toggle("Recortar Imagem"):
        processor.crop_image()

    # Exibe a imagem processada
    processed_image = processor.get_processed_image()
    col3.image(processed_image, caption='Imagem Processada', use_container_width=True)
    st.session_state['processed_image'] = processed_image

    # Botões de navegação
    st.sidebar.markdown("---")
    if st.sidebar.button("Proceder para Detecção"):
        st.session_state.page = 'detection'
        st.rerun()
    if st.sidebar.button("Desfazer Alterações"):
        st.session_state['processed_image'] = st.session_state['original_image'].copy()
        st.rerun()
    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.session_state.page = 'home'
        st.rerun()
