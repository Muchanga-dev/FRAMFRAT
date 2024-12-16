# utils/visualization.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import mplstereonet
import pandas as pd


def plot_orientacoes(df_fraturas):
    """
    Plota o diagrama de rosa das orientações das fraturas.

    Args:
        df_fraturas (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    orientacoes = df_fraturas["Orientação (graus)"].dropna().astype(float) % 360

    if orientacoes.empty:
        st.warning("Não há fraturas disponíveis para plotar as orientações.")
        return

    # Definir bins (intervalos) para orientações de 0° a 360°
    bins = np.arange(0, 361, 10)  # Intervalos de 10 graus
    hist, bin_edges = np.histogram(orientacoes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(go.Barpolar(
        r=hist,
        theta=bin_centers,
        width=10,
        marker=dict(
            color=hist,
            colorscale='Viridis',
            line=dict(color='black')
        ),
        opacity=0.8
    ))

    fig.update_layout(
        title='Diagrama de Rosa das Orientações das Fraturas',
        polar=dict(
            radialaxis=dict(
                tickfont_size=10,
                showticklabels=True,
                ticks='outside',
                visible=True
            ),
            angularaxis=dict(
                tickfont_size=10,
                rotation=90,
                direction='clockwise',  # Sentido horário
                tickmode='array',
                tickvals=np.arange(0, 360, 30),
                ticktext=[f'{i}°' for i in np.arange(0, 360, 30)]
            )
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_histogram(data, title='', xaxis_title='', nbins=20):
    """
    Plota um histograma dos dados fornecidos.

    Args:
        data (array-like): Dados numéricos para plotar.
        title (str): Título do gráfico.
        xaxis_title (str): Título do eixo x.
        nbins (int): Número de bins do histograma.
    """
    if len(data) == 0:
        st.warning("Nenhum dado disponível para plotar o histograma.")
        return

    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=nbins, marker_color='skyblue', opacity=0.75)])
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title='Frequência',
        bargap=0.2,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_fracture_size_distribution(data):
    """
    Analisa a distribuição do tamanho das fraturas e plota o gráfico de regressão linear.

    Args:
        data (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    aperturas = data["Abertura Média (mm)"].astype(float)
    aperturas = aperturas[aperturas > 0].sort_values(ascending=False).reset_index(drop=True)
    cumulative_count = np.arange(1, len(aperturas) + 1)

    if len(aperturas) == 0:
        st.warning("Nenhuma abertura média disponível para plotar a distribuição do tamanho das fraturas.")
        return

    log_aperturas = np.log10(aperturas)
    log_cumulative_count = np.log10(cumulative_count)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_aperturas, log_cumulative_count)

    # Preparar os dados para plotagem
    fit_line = slope * log_aperturas + intercept
    r_squared = r_value ** 2

    # Formatar a equação da regressão linear
    intercept_formatted = f"{intercept:.2f}"
    slope_formatted = f"{slope:.2f}"
    equation = f"y = {slope_formatted}x + {intercept_formatted}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(aperturas, cumulative_count, marker='o', linestyle='none', label="Dados")
    ax.loglog(
        aperturas,
        10**fit_line,
        'b-',
        label=f"Regressão Linear: {equation}\n$R^2$ = {r_squared:.4f}"
    )

    ax.set_xlabel("Abertura Média (mm)")
    ax.set_ylabel("Contagem Cumulativa de Fraturas")
    ax.set_title("Distribuição do Tamanho das Fraturas")
    ax.legend()

    st.pyplot(fig, use_container_width=True)


def plot_stereogram(data, dip_angle=90.0, latitude=None, longitude=None):
    """
    Plota o estereograma das orientações das fraturas.

    Args:
        data (pandas.DataFrame): DataFrame com dados das fraturas.
        dip_angle (float): Ângulo de mergulho.
        latitude (str): Latitude da localização (opcional).
        longitude (str): Longitude da localização (opcional).
    """
    orientations = data["Orientação (graus)"].dropna().astype(float) % 360
    if len(orientations) == 0:
        st.warning("Não há orientações de fraturas disponíveis para plotar o estereograma.")
        return

    strikes = orientations
    dips = np.full_like(strikes, dip_angle)  # Usando o ângulo de mergulho fornecido

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='stereonet')

    # Plotar os polos das fraturas
    ax.pole(strikes, dips, 'o', markersize=5, color='red', label="Polos das Fraturas")

    # Adicionar contornos de densidade para os polos
    ax.density_contourf(strikes, dips, measurement='poles', cmap='Reds', alpha=0.3)

    ax.grid()

    # Adicionar a legenda
    ax.legend(loc='best')

    # Adicionar coordenadas geográficas no título ou como texto adicional
    if latitude and longitude:
        plt.title(f"Estereograma das Fraturas\nLocalização: Lat {latitude}, Lon {longitude}")
    else:
        plt.title("Estereograma das Fraturas")

    st.pyplot(fig, use_container_width=True)


def plot_boxplot_aberturas(analysis_results):
    """
    Plota um box plot das aberturas das fraturas.

    Args:
        analysis_results (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    abertura_columns = ["Abertura Mínima (mm)", "Abertura Média (mm)", "Abertura Máxima (mm)"]
    abertura_data = analysis_results[abertura_columns].dropna()

    if abertura_data.empty:
        st.warning("Nenhuma abertura disponível para plotar o box plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    abertura_data.boxplot(ax=ax, grid=True, patch_artist=True,
                          boxprops=dict(facecolor='skyblue', color='black'),
                          medianprops=dict(color='red'),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'),
                          flierprops=dict(color='black', markeredgecolor='black'))
    ax.set_title("Box Plot das Aberturas das Fraturas")
    ax.set_ylabel("Abertura (mm)")
    st.pyplot(fig, use_container_width=True)


def plot_scatter_abertura_comprimento(analysis_results):
    """
    Plota um scatter plot das aberturas médias vs. comprimentos das fraturas.

    Args:
        analysis_results (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    if analysis_results.empty:
        st.warning("Nenhuma fratura disponível para plotar o scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(analysis_results["Comprimento (mm)"], analysis_results["Abertura Média (mm)"], alpha=0.7, edgecolors='w', color='blue')
    ax.set_title("Abertura Média vs. Comprimento das Fraturas")
    ax.set_xlabel("Comprimento (mm)")
    ax.set_ylabel("Abertura Média (mm)")
    ax.grid(True)
    st.pyplot(fig, use_container_width=True)


def plot_multiple_histograms_aberturas(analysis_results):
    """
    Plota histogramas separados para abertura mínima, média e máxima.

    Args:
        analysis_results (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    abertura_columns = ["Abertura Mínima (mm)", "Abertura Média (mm)", "Abertura Máxima (mm)"]
    for col in abertura_columns:
        abertura_data = analysis_results[col].dropna().astype(float)
        if abertura_data.empty:
            st.warning(f"Nenhuma abertura disponível para plotar o histograma de {col}.")
            continue
        fig = go.Figure(data=[go.Histogram(x=abertura_data, nbinsx=30, marker_color='teal', opacity=0.75)])
        fig.update_layout(
            title=f"Histograma de {col}",
            xaxis_title="Abertura (mm)",
            yaxis_title="Frequência",
            bargap=0.2,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_matriz_correlacao(analysis_results):
    """
    Plota uma matriz de correlação das métricas das fraturas.

    Args:
        analysis_results (pandas.DataFrame): DataFrame com dados das fraturas.
    """
    metrics = [
        "Comprimento (mm)", 
        "Abertura Mínima (mm)", 
        "Abertura Média (mm)", 
        "Abertura Máxima (mm)", 
        "Abertura Std (mm)", 
        "Abertura IC Lower (mm)", 
        "Abertura IC Upper (mm)", 
        "Permeabilidade (D)", 
        "Porosidade"
    ]
    available_metrics = [metric for metric in metrics if metric in analysis_results.columns]
    if not available_metrics:
        st.warning("Nenhuma métrica disponível para plotar a matriz de correlação.")
        return
    corr = analysis_results[available_metrics].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_yticks(range(len(available_metrics)))
    ax.set_xticklabels(available_metrics, rotation=90)
    ax.set_yticklabels(available_metrics)

    # Adicionar os valores na matriz
    for (i, j), val in np.ndenumerate(corr):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')

    ax.set_title("Matriz de Correlação das Métricas das Fraturas", pad=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def plot_heatmap_aberturas(image_real, analysis_results, labeled_image):
    """
    Plota um heatmap das aberturas sobre a imagem original.

    Args:
        image_real (PIL.Image.Image): Imagem original.
        analysis_results (pandas.DataFrame): DataFrame com dados das fraturas.
        labeled_image (numpy.ndarray): Imagem rotulada das fraturas.
    """
    if image_real is None or labeled_image is None or analysis_results.empty:
        st.warning("Dados insuficientes para plotar o heatmap das aberturas.")
        return

    # Criar uma imagem de aberturas
    abertura_map = np.zeros(labeled_image.shape, dtype=np.float32)

    for index, row in analysis_results.iterrows():
        # Extrair o número da fratura ou segmento
        fratura_id = row["ID_Fratura"]
        segmento_id = row["ID_Segmento"]
        label_original = segmento_id if segmento_id else fratura_id
        try:
            parts = label_original.split('_')
            if segmento_id:
                label_num = int(parts[2])
            else:
                label_num = int(parts[1])
        except (IndexError, ValueError):
            continue
        mask = (labeled_image == label_num)
        abertura_map[mask] = row["Abertura Média (mm)"] if "Abertura Média (mm)" in row and not pd.isna(row["Abertura Média (mm)"]) else 0

    if np.all(abertura_map == 0):
        st.warning("Nenhuma abertura média disponível para plotar o heatmap.")
        return

    # Normalizar as aberturas para o intervalo [0, 1]
    norm = plt.Normalize(vmin=np.min(abertura_map), vmax=np.max(abertura_map))
    cmap = plt.cm.jet

    # Aplicar o colormap
    heatmap = cmap(norm(abertura_map))[:, :, :3]  # Ignorar o canal alfa

    # Sobrepor o heatmap na imagem original
    image_real_np = np.array(image_real).astype(np.float32) / 255
    heatmap_np = heatmap.astype(np.float32)

    # Assegurar que ambas as imagens têm o mesmo tipo e escala
    if image_real_np.shape[:2] != heatmap_np.shape[:2]:
        heatmap_np = cv2.resize(heatmap_np, (image_real_np.shape[1], image_real_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    if image_real_np.dtype != heatmap_np.dtype:
        image_real_np = image_real_np.astype(heatmap_np.dtype)

    try:
        overlay = cv2.addWeighted(image_real_np, 0.7, heatmap_np, 0.3, 0)
    except cv2.error as e:
        st.error(f"Erro no addWeighted: {e}")
        overlay = image_real_np  # Fallback caso ocorra erro

    # Converter de volta para uint8 para exibição
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay_uint8)
    ax.set_title("Heatmap das Aberturas Médias das Fraturas")
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)
