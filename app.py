# app.py
import streamlit as st
from utils.config import configure_app
from utils.home import render_home
from utils.preprocessing import render_preprocessing_page
from utils.detection import render_detection_page
from utils.characterization import render_characterization_page

# Configura a aplicação com o ícone especificado
configure_app("assets/logos/icon.png")

def main():
    """
    Função principal que gerencia a navegação entre as diferentes páginas da aplicação.
    """
    # Inicializa o estado da sessão com valores padrão, se não existirem
    st.session_state.setdefault('page', 'home')
    st.session_state.setdefault('uploaded_file', None)
    st.session_state.setdefault('processed_image', None)
    st.session_state.setdefault('detection_image', None)
    st.session_state.setdefault('analysis_results', None)
    st.session_state.setdefault('skeleton_image', None)
    st.session_state.setdefault('area_total_mm2', None)
    st.session_state.setdefault('image_angle', 0.0)  # Adicionado campo para ângulo da imagem
    st.session_state.setdefault('advanced_segmentation', False)  # Adicionado campo para segmentação avançada

    # Verifica se um arquivo foi carregado na página inicial e redireciona para o pré-processamento
    if st.session_state.uploaded_file is not None and st.session_state.page == 'home':
        st.session_state.page = 'preprocessing'
        st.rerun()

    # Navegação entre páginas com base no estado atual
    if st.session_state.page == 'home':
        logo_path = "assets/logos/logo.png"
        render_home(logo_path)

        st.sidebar.info("Bem-vindo! Por favor, faça upload de uma imagem para começar.")

        # Upload de imagem na barra lateral
        uploaded_file = st.sidebar.file_uploader("Faça Upload de uma Imagem", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.page = 'preprocessing'
            st.rerun()

    elif st.session_state.page == 'preprocessing':
        render_preprocessing_page()

    elif st.session_state.page == 'detection':
        render_detection_page()

    elif st.session_state.page == 'characterization':
        render_characterization_page()

    else:
        st.error("Página não encontrada.")

if __name__ == "__main__":
    main()
