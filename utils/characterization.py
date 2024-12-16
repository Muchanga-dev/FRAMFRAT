# utils/characterization.py

import streamlit as st
import pandas as pd
from utils.core import FractureCharacterizer
from utils.visualization import (
    plot_orientacoes,
    plot_histogram,
    plot_fracture_size_distribution,
    plot_stereogram,
    plot_boxplot_aberturas,
    plot_scatter_abertura_comprimento,
    plot_multiple_histograms_aberturas,
    plot_matriz_correlacao,
    plot_heatmap_aberturas
)
import numpy as np
import math
import cv2
from io import BytesIO 
from PIL import Image, ImageDraw, ImageFont


def render_characterization_page():
    """
    Renderiza a página de caracterização das fraturas, incluindo visualização dos resultados,
    análise estatística e opções para download das tabelas em formatos Excel e CSV.

    O fluxo principal inclui:
        - Verificação da existência da imagem de detecção na sessão.
        - Coleta de parâmetros de entrada do usuário via sidebar.
        - Instanciação e execução do caracterizador de fraturas.
        - Exibição dos resultados com base na seleção do usuário.
        - Opções de download para os resultados gerados.
        - Navegação entre diferentes páginas da aplicação.
    """

    # Verifica se a imagem de detecção está disponível na sessão
    if 'detection_image' not in st.session_state or st.session_state['detection_image'] is None:
        st.warning("Por favor, vá para 'Detecção de Fraturas' antes de prosseguir.")
        st.session_state.page = 'detection'
        st.rerun() 
        return

    # Obtém a imagem de detecção e a imagem processada da sessão
    detection_image = st.session_state['detection_image']
    image_real = st.session_state.get('processed_image', None)

    # Obter ou definir o valor padrão para a área total da imagem
    default_area_value = st.session_state.get('area_total_mm2', 0.0)
    if default_area_value is None or default_area_value == 0.0:
        default_area_value = 0.0

    # Entrada do usuário para a área total da imagem em mm² via sidebar
    area_total_mm2 = st.sidebar.number_input(
        "Insira a área total da imagem em mm²",
        min_value=0.0,
        format="%.4f",
        value=default_area_value
    )
    st.session_state['area_total_mm2'] = area_total_mm2

    # Verifica se a área total inserida é válida
    if area_total_mm2 <= 0:
        st.warning("Por favor, insira um valor válido para a área total da imagem em mm².")
        return
    
    # Define os valores máximos para os filtros de comprimento e abertura com base na área total
    max_length = math.sqrt(area_total_mm2) 
    max_opening = max_length * 0.2

    # Entradas do usuário para filtros de comprimento mínimo e abertura mínima via sidebar
    min_length = st.sidebar.slider(
        "Comprimento Mínimo (mm)", 
        min_value=5.0, 
        max_value=max_length, 
        value=10.0, 
        step=1.0
    )
    min_opening = st.sidebar.slider(
        "Abertura Mínima (mm)", 
        min_value=1.0, 
        max_value=max_opening, 
        value=2.0, 
        step=1.0
    )

    # Entrada do usuário para o ângulo de mergulho via sidebar
    dip_angle = st.sidebar.number_input(
        "Ângulo de Mergulho (graus)", 
        min_value=0.0, 
        max_value=90.0, 
        value=90.0, 
        step=1.0
    )

    # Entrada do usuário para o ângulo de orientação da imagem via sidebar
    image_angle_input = st.sidebar.text_input(
        "Insira o ângulo da imagem em relação ao norte geográfico (em graus)", 
        value="", 
        help="Deixe em branco para assumir que o topo da imagem é o norte geográfico."
    )
    if image_angle_input:
        try:
            image_angle = float(image_angle_input) % 360
            st.session_state['image_angle'] = image_angle
        except ValueError:
            st.sidebar.error("Por favor, insira um valor numérico válido para o ângulo.")
            st.session_state['image_angle'] = 0.0
    else:
        image_angle = 0.0
        st.session_state['image_angle'] = image_angle  # Assume que o topo é o norte geográfico

    # Opção de seleção para diferentes tipos de visualizações de resultados
    result = st.sidebar.selectbox(
        "Selecione O resultado",
        [
            'Fraturas Detectadas',
            'Fraturas e Segmentos',
            'Diagrama de Rose e Estereograma',
            'Distribuição e Box Plot das Aberturas',
            'Histograma de Comprimento e Aberturas',
            'Matriz de Correlação e HeatMap das Aberturas'
        ]
    )

    # Entrada do usuário para habilitar a segmentação avançada via sidebar
    segmentation_enabled = st.sidebar.checkbox(
        "Habilitar Segmentação Avançada das Fraturas",
        value=False,
        help="Ative para segmentar fraturas complexas em segmentos lineares."
    )
    
    # Título central da página
    st.markdown("<h3 style='text-align: center; color: Green;'>CARACTERIZAÇÃO DAS FRATURAS</h3>", unsafe_allow_html=True)
    
    # Exibe um spinner enquanto realiza a caracterização das fraturas
    with st.spinner("Realizando caracterização das fraturas..."):
        # Instancia o caracterizador de fraturas com os parâmetros fornecidos
        characterizer = FractureCharacterizer(
            detection_image=detection_image,
            area_total_mm2=area_total_mm2,
            image_real=image_real,
            min_length=min_length,
            min_opening=min_opening,
            image_angle=image_angle,
            segmentation_enabled=segmentation_enabled  # Passa o parâmetro de segmentação
        )
        # Executa a análise das fraturas
        analysis_results, processed_images, fractal_stats, segments_results = characterizer.analisar_fraturas()
        
        # Define a estrutura de colunas para exibição dos resultados
        col1, col2, col3 = st.columns([5, 1, 5])
        
        # Verifica se há resultados de análise disponíveis
        if analysis_results is not None and not analysis_results.empty:
            # Dependendo da seleção do usuário, exibe diferentes tipos de resultados
            if result == 'Fraturas Detectadas':
                # Seção de Estatísticas Resumidas
                st.markdown("#### Estatísticas Resumidas")
                
                # Define as colunas a serem descritas, removendo 'Fratura_Original' se presente
                columns_to_describe = [
                    "Comprimento (mm)", 
                    "Abertura Mínima (mm)", 
                    "Abertura Média (mm)", 
                    "Abertura Máxima (mm)", 
                    "Abertura Std (mm)", 
                    "Orientação (graus)", 
                    "Permeabilidade (D)", 
                    "Porosidade"
                ]
                available_columns = [col for col in columns_to_describe if col in analysis_results.columns]
                stats = analysis_results.drop(columns=["Fratura_Original"], errors='ignore')[available_columns].describe()
                st.dataframe(stats)

                # Exibe as estatísticas fractais
                st.dataframe(fractal_stats)

                # Cria um objeto BytesIO para armazenar o arquivo Excel em memória
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Escreve cada DataFrame em uma planilha separada
                    analysis_results.to_excel(writer, index=False, sheet_name='Resultados_Detalhados_Fraturas')
                    stats.to_excel(writer, sheet_name='Estatisticas_Resumidas_Fraturas')
                    fractal_stats.to_excel(writer, index=False, sheet_name='Dimensao_Fractal_Fraturas')
                processed_data = output.getvalue()  # Obtém os dados após fechar o writer

                # Botão para baixar todas as tabelas em formato Excel
                st.sidebar.download_button(
                    label="Baixar todas as tabelas em Excel",
                    data=processed_data,
                    file_name='Caracterizacao_Fraturas.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                # Preparar o CSV para download, removendo 'branch_image' se presente
                analysis_results_csv = analysis_results.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Baixar tabela de resultados em CSV",
                    data=analysis_results_csv,
                    file_name='Resultados_Detalhados_Fraturas.csv',
                    mime='text/csv'
                )

                # Exibe a tabela completa de resultados das fraturas e segmentos, excluindo 'Fratura_Original'
                st.markdown("#### Tabela de Resultados das Fraturas e Segmentos")
                st.dataframe(analysis_results.drop(columns=["Fratura_Original"], errors='ignore'))
                
                # Exibição opcional: Imagem das fraturas detectadas com IDs sobrepostos
                if "imagem_com_numero" in processed_images:
                    st.image(
                        processed_images["imagem_com_numero"], 
                        caption='Fraturas e Segmentos Detectados com ID', 
                        use_container_width=True
                    )
                else:
                    # Se 'imagem_com_numero' não estiver disponível, exibe a imagem sem IDs
                    st.image(
                        processed_images["overlay_image"], 
                        caption='Fraturas Detectadas', 
                        use_container_width=True
                    )

            elif result == 'Fraturas e Segmentos':
                # Seção para seleção e visualização de fraturas e seus segmentos
                st.sidebar.markdown("### Visualização de Fraturas e Segmentos")

                # Filtrar apenas as fraturas originais (sem segmentos)
                original_fractures = sorted(set(analysis_results["ID_Fratura"]))
                selected_fracture = col1.selectbox("Selecione Fratura 1", original_fractures)
                selected_fracture1 = col3.selectbox("Selecione Fratura 2", original_fractures)

                if selected_fracture:
                    
                    # Exibe a tabela de resultados da fratura selecionada
                    col1.markdown(f"**Fratura Selecionada:** {selected_fracture}")
                    col3.markdown(f"**Fratura Selecionada:** {selected_fracture1}")
                    # Cria uma máscara para a fratura selecionada com base no label
                    try:
                        # Extrai o número da fratura a partir do ID
                        fracture_label = int(selected_fracture.split("_")[1])
                        fracture_label1 = int(selected_fracture1.split("_")[1])
                    except (IndexError, ValueError):
                        st.error("ID da fratura inválido.")
                        return
                    fracture_mask = (characterizer.labeled_image == fracture_label).astype(np.uint8)
                    fracture_mask_rgb = cv2.cvtColor(fracture_mask * 255, cv2.COLOR_GRAY2RGB)

                    fracture_mask1 = (characterizer.labeled_image == fracture_label1).astype(np.uint8)
                    fracture_mask_rgb1 = cv2.cvtColor(fracture_mask1 * 255, cv2.COLOR_GRAY2RGB)

                    # Exibe a imagem da fratura selecionada
                    combined_image = fracture_mask_rgb.copy()
                    col1.image(combined_image, caption=f'Fratura {selected_fracture}', use_container_width=True)

                    combined_image1 = fracture_mask_rgb1.copy()
                    col3.image(combined_image1, caption=f'Fratura {selected_fracture1}', use_container_width=True)


            elif result == 'Diagrama de Rose e Estereograma':
                # Seção para visualização de orientações das fraturas
                st.sidebar.markdown("### Diagrama de Rosa e Estereograma das Orientações")
                if analysis_results["Orientação (graus)"].notnull().any():
                    with col1:
                        # Plota o diagrama de orientações (Diagrama de Rosa)
                        plot_orientacoes(analysis_results)
                else:
                    col1.warning("Nenhuma orientação válida disponível para plotar.")

                if analysis_results["Orientação (graus)"].notnull().any():
                    with col3:
                        # Plota o estereograma das orientações das fraturas
                        plot_stereogram(
                            analysis_results, 
                            dip_angle=dip_angle
                        )
                else:
                    col3.warning("Nenhuma orientação válida disponível para plotar o estereograma.")

            elif result == 'Distribuição e Box Plot das Aberturas':
                # Seção para visualização da distribuição das aberturas das fraturas
                st.sidebar.markdown("### Distribuição do Tamanho das Fraturas e Box Plot das Aberturas")
                with col1:
                    # Plota a distribuição do tamanho das fraturas
                    plot_fracture_size_distribution(analysis_results)
                with col3:
                    # Plota o box plot das aberturas das fraturas
                    plot_boxplot_aberturas(analysis_results)

            elif result == 'Histograma de Comprimento e Aberturas':
                # Seção para visualização de histogramas de comprimento e aberturas
                st.sidebar.markdown("### Histogramas de Comprimentos e Aberturas") 
                with col1:
                    # Plota o histograma da distribuição dos comprimentos das fraturas
                    plot_histogram(
                        analysis_results["Comprimento (mm)"].astype(float),
                        title='Distribuição dos Comprimentos das Fraturas',
                        xaxis_title='Comprimento (mm)'
                    )

                    # Plota o histograma da distribuição das aberturas das fraturas
                    plot_histogram(
                        analysis_results["Abertura Média (mm)"].astype(float),
                        title='Distribuição das Aberturas das Fraturas',
                        xaxis_title='Abertura (mm)'
                    )

                    # Plota o scatter plot entre abertura e comprimento das fraturas
                    plot_scatter_abertura_comprimento(analysis_results)
                
                with col3:
                    # Plota múltiplos histogramas das aberturas das fraturas
                    plot_multiple_histograms_aberturas(analysis_results)

            elif result == 'Matriz de Correlação e HeatMap das Aberturas':
                # Seção para visualização da matriz de correlação e heatmap das aberturas
                st.sidebar.markdown("### Matriz de Correlação e HeatMap das Aberturas") 
                with col1:
                    # Plota a matriz de correlação entre as métricas das fraturas
                    plot_matriz_correlacao(analysis_results)
                
                with col3:
                    # Plota o heatmap das aberturas das fraturas sobre a imagem real
                    plot_heatmap_aberturas(image_real, analysis_results, characterizer.labeled_image)

        else:
            # Aviso caso nenhuma fratura tenha sido detectada com os parâmetros atuais
            st.warning(
                "Nenhuma fratura foi detectada com os parâmetros atuais. Por favor, ajuste os filtros de abertura mínima e comprimento mínimo para valores menores."
            )
        
        # Se a segmentação avançada estiver habilitada e houver segmentos, permitir download
        if segmentation_enabled and segments_results:
            st.markdown("#### Tabela de Segmentos das Fraturas")
            segments_df = pd.DataFrame(segments_results)
            st.dataframe(segments_df)


        # Seção de botões de navegação na sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("Voltar para Detecção"):
            st.session_state.page = 'detection'
            st.experimental_rerun()  # Reinicia a aplicação para carregar a página de detecção
        if st.sidebar.button("Voltar para Processamento"):
            st.session_state.page = 'preprocessing'
            st.experimental_rerun()  # Reinicia a aplicação para carregar a página de pré-processamento
        if st.sidebar.button("Sair"):
            st.session_state.clear()  # Limpa o estado da sessão
            st.session_state.page = 'home'
            st.experimental_rerun()  # Reinicia a aplicação para carregar a página inicial
