# utils/detection.py

import streamlit as st
import numpy as np
import cv2
from utils.core import ImageProcessorBackend

def render_detection_page():
    """
    Renderiza a página de detecção de fraturas, permitindo ajustes nos parâmetros de detecção.
    """

    st.markdown("<h3 style='text-align: center; color: Green;'> Detecção de Fraturas </h3>", unsafe_allow_html=True)

    # Verifica se a imagem processada está disponível na sessão
    if 'processed_image' not in st.session_state or st.session_state['processed_image'] is None:
        st.warning("Por favor, vá para 'Processamento de Imagem' e salve a imagem processada antes de prosseguir.")
        st.session_state.page = 'preprocessing'
        st.rerun()
        return

    image = st.session_state['processed_image']

    # Instancia o backend de processamento
    backend = ImageProcessorBackend(image)

    # Opções de detecção na barra lateral
    st.sidebar.subheader("Ajustes Manual de Detecção")

    # Slider para ajuste de eliminação de ruído
    ruido_level = st.sidebar.slider(
        "Eliminar Ruído",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Ajuste o nível de eliminação de ruído na imagem."
    )

    # Aplicar eliminação de ruído
    if ruido_level > 0:
        with st.spinner("Eliminando ruído..."):
            backend.ruido_scanner(ruido_level)

    # Slider para ajuste do limiar de Otsu
    otsu_adjustment = st.sidebar.slider(
        "Ajuste do Limiar de Otsu",
        min_value=-100,
        max_value=100,
        value=0,
        step=1,
        help="Ajuste fino do limiar de Otsu para melhorar a binarização."
    )

    # Botão para realizar a detecção inicial de fraturas
    if st.sidebar.button("Realizar Detecção Inicial"):
       with st.spinner("Realizando detecção inicial de fraturas..."):
            resultados = backend.process_and_visualize(
                adjustment=otsu_adjustment,
                show_contours=True,
                connectivity=True,
                connectivity_alpha=0.4,
                overlay_alpha=0.3,
                overlay_color=(0, 0, 255)  # Vermelho
            )

            # Salva os resultados iniciais na sessão
            st.session_state['initial_detection_image'] = resultados.get('binary_otsu')
            st.session_state['initial_overlay_image'] = resultados.get('overlay_fraturas')
            st.session_state['initial_thresh'] = resultados.get('threshold')

            # Salva a detecção inicial também como a detecção atual
            st.session_state['detection_image'] = st.session_state['initial_detection_image']
            st.session_state['detection_overlay_image'] = st.session_state['initial_overlay_image']

    # Verifica se a detecção inicial foi feita
    if 'initial_detection_image' not in st.session_state or st.session_state['initial_detection_image'] is None:
        st.warning("Clique no botão 'Realizar Detecção Inicial' para começar.")
        return

    # Atualiza a visualização com base nos ajustes dos filtros, se necessário
    with st.spinner("Processando ajustes na detecção..."):
        resultados_ajustados = backend.process_and_visualize(
            adjustment=otsu_adjustment,
            show_contours=True,
            connectivity=True,
            connectivity_alpha=0.4,
            overlay_alpha=0.3,
            overlay_color=(0, 255, 0)  # Verde para indicar ajustes
        )
        adjusted_overlay_image = resultados_ajustados.get('overlay_fraturas')

    # Exibe a visualização inicial e ajustada lado a lado
   #st.markdown("### Comparação da Detecção Inicial e Ajustada")
    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state['initial_overlay_image'], caption='Detecção de Fraturas (Sem Ajuste Manual)',  use_container_width=True)

    with col2:
        st.image(adjusted_overlay_image, caption='Detecção de Fraturas (Com Ajuste Manual)',  use_container_width=True)

    # Atualiza a detecção ajustada na sessão para uso posterior
    st.session_state['detection_image'] = resultados_ajustados.get('binary_otsu')
    st.session_state['detection_overlay_image'] = adjusted_overlay_image

    # Exibe informações do limiar aplicado
    st.sidebar.markdown(f"**Limiar Final Ajustado:** {resultados_ajustados.get('threshold', 0):.2f}")

    # Botões de navegação
    st.sidebar.markdown("---")
    if st.sidebar.button("Proceder para Caracterização"):
        if 'detection_image' not in st.session_state or st.session_state['detection_image'] is None:
            st.warning("Nenhuma detecção disponível. Por favor, processe a imagem primeiro.")
        else:
            st.session_state.page = 'characterization'
            st.rerun()
    if st.sidebar.button("Voltar para Processamento"):
        st.session_state.page = 'preprocessing'
        st.rerun()
    if st.sidebar.button("Sair"):
        st.session_state.clear()
        st.session_state.page = 'home'
        st.rerun()
