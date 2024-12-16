# utils/core.py

import numpy as np
import cv2
import pandas as pd
from skimage.morphology import skeletonize, label
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from scipy.ndimage import convolve
import math
from skimage import measure, color
import networkx as nx  # Importado para manipulação de grafos


class ImageProcessorBackend:
    """
    Classe para processar imagens durante a etapa de detecção, incluindo pré-processamento,
    aplicação de limiarização adaptativa e análise de conectividade.
    """
    def __init__(self, image: Image.Image):
        """
        Inicializa o processador de imagem.

        Args:
            image (PIL.Image.Image): Imagem original carregada.
        """
        self.image = image
        self.gray_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
        self.processed_image = self.pre_processamento(self.gray_image)

    def pre_processamento(self, gray_image):
        """
        Aplica um filtro mediano para reduzir ruído na imagem.

        Args:
            gray_image (numpy.ndarray): Imagem em escala de cinza.

        Returns:
            numpy.ndarray: Imagem processada.
        """
        gray_image = cv2.medianBlur(gray_image, ksize=3)
        return gray_image

    def ruido_scanner(self, ruido_level=0):
        """
        Elimina ruído da imagem usando o filtro Non-local Means Denoising.

        Args:
            ruido_level (int, opcional): Nível de ruído a ser eliminado. Valor entre 0 e 100. Defaults to 0.

        Returns:
            numpy.ndarray: Imagem denoised.
        """
        # Garantir que o nível de ruído está dentro do intervalo
        ruido_level = max(0, min(ruido_level, 100))
        # Aplica o filtro Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(self.processed_image, None, h=ruido_level + 10, templateWindowSize=7, searchWindowSize=21)
        self.processed_image = denoised
        return denoised

    def aplicar_otsu_potencializado(self, adjustment=-10):
        """
        Aplica o método de Otsu para binarização com ajuste adicional.

        Args:
            adjustment (int, opcional): Ajuste adicional para o limiar. Pode ser positivo ou negativo.
                                         Defaults to -10.

        Returns:
            tuple:
                numpy.ndarray: Imagem binarizada após Otsu e processamento morfológico.
                float: Valor do limiar final aplicado.
        """
        # Aplica um desfoque Gaussian para reduzir ruídos
        blur = cv2.GaussianBlur(self.processed_image, (5, 5), 0)

        # Calcula o limiar de Otsu usando scikit-image
        otsu_thresh = threshold_otsu(blur)
        final_thresh = otsu_thresh + adjustment

        # Aplica o limiar final para binarização
        _, binary_otsu = cv2.threshold(blur, final_thresh, 255, cv2.THRESH_BINARY)

        # Inverte a imagem binarizada se necessário
        binary_otsu = cv2.bitwise_not(binary_otsu)

        # Aplica processamento morfológico na imagem binarizada
        processed_otsu = self.processamento_morfologico(binary_otsu)

        return processed_otsu, final_thresh

    def processamento_morfologico(self, binary_image):
        """
        Aplica operações morfológicas para melhorar a qualidade da detecção.

        Args:
            binary_image (numpy.ndarray): Imagem binarizada.

        Returns:
            numpy.ndarray: Imagem após operações morfológicas.
        """
        kernel = np.ones((3, 3), np.uint8)
        # Remove pequenos ruídos
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        # Preenche pequenos buracos nas fraturas
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closing

    def conect_fraturas(self, connectivity=2):
        """
        Analisa a conectividade das fraturas na imagem binarizada.

        Args:
            connectivity (int, opcional): Tipo de conectividade (1 para 4-conectividade, 2 para 8-conectividade).
                                         Defaults to 2.

        Returns:
            tuple:
                numpy.ndarray: Imagem colorida com rótulos de conectividade.
                int: Número de componentes conectadas detectadas.
        """
        # Utiliza a imagem binarizada após Otsu e morfologia
        binary_image = self.processed_image

        # Assegura que a imagem está binarizada
        if len(binary_image.shape) != 2:
            raise ValueError("A imagem para conect_fraturas deve ser binarizada (2D).")

        # Aplica o rótulo de conectividade
        labeled_image, count = measure.label(binary_image, connectivity=connectivity, return_num=True)
        colored_label_image = color.label2rgb(labeled_image, bg_label=0)

        return colored_label_image, count

    def exibir_esqueleto(self, processed_image):
        """
        Gera o esqueleto da imagem binarizada.

        Args:
            processed_image (numpy.ndarray): Imagem binarizada após processamento.

        Returns:
            numpy.ndarray: Imagem do esqueleto.
        """
        if processed_image is None:
            return None
        skeleton = skeletonize(processed_image > 0)
        skeleton_image = (skeleton.astype(np.uint8) * 255)
        return skeleton_image

    def overlay_images(self, image_background, image_foreground, color=(0, 0, 255), alpha=0.3, contour=True):
        """
        Sobrepõe a imagem de detecção na imagem original para visualização aprimorada.

        Args:
            image_background (numpy.ndarray): Imagem original.
            image_foreground (numpy.ndarray): Imagem binarizada de detecção.
            color (tuple): Cor para destacar as trafuras (R, G, B). Padrão é vermelho.
            alpha (float): Transparência da sobreposição. Valor entre 0 e 1. Padrão é 0.3.
            contour (bool): Se True, desenha contornos das trafuras.

        Returns:
            numpy.ndarray: Imagem sobreposta.
        """
        if image_foreground is None:
            return image_background

        # Aplicar filtro para reduzir ruídos
        kernel = np.ones((3, 3), np.uint8)
        image_foreground = cv2.morphologyEx(image_foreground, cv2.MORPH_OPEN, kernel, iterations=2)
        image_foreground = cv2.dilate(image_foreground, kernel, iterations=1)

        # Converter a imagem binária para RGB
        overlay = cv2.cvtColor(image_foreground, cv2.COLOR_GRAY2RGB)

        # Redimensionar se necessário
        if image_background.shape[:2] != overlay.shape[:2]:
            overlay = cv2.resize(overlay, (image_background.shape[1], image_background.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Aplicar a cor escolhida nas áreas detectadas
        mask = (image_foreground > 0)
        overlay[mask] = color

        # Desenhar contornos se solicitado
        if contour:
            contours, _ = cv2.findContours(image_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_background, contours, -1, color, 2)

        # Garantir que ambas as imagens têm o mesmo tipo de dados
        if image_background.dtype != overlay.dtype:
            image_background = image_background.astype(overlay.dtype)

        # Combinar as imagens com a transparência especificada
        try:
            output = cv2.addWeighted(image_background, 1 - alpha, overlay, alpha, 0)
        except cv2.error as e:
            print(f"Erro no addWeighted: {e}")
            output = image_background  # Fallback caso ocorra erro

        return output

    def overlay_connectivity(self, image_background, colored_labels, alpha=0.4):
        """
        Sobrepõe a imagem de conectividade na imagem original para visualização.

        Args:
            image_background (numpy.ndarray): Imagem original.
            colored_labels (numpy.ndarray): Imagem colorida com rótulos de conectividade.
            alpha (float): Transparência da sobreposição. Valor entre 0 e 1. Padrão é 0.4.

        Returns:
            numpy.ndarray: Imagem sobreposta com conectividade.
        """
        if colored_labels is None:
            return image_background

        # Assegurar que ambas as imagens estão no mesmo tamanho
        if image_background.shape[:2] != colored_labels.shape[:2]:
            colored_labels = cv2.resize(colored_labels, (image_background.shape[1], image_background.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Converter para o mesmo tipo de dados
        if image_background.dtype != colored_labels.dtype:
            image_background = image_background.astype(colored_labels.dtype)

        # Combinar as imagens com a transparência especificada
        try:
            output = cv2.addWeighted(image_background, 1 - alpha, colored_labels.astype(np.uint8), alpha, 0)
        except cv2.error as e:
            print(f"Erro no addWeighted para conectividade: {e}")
            output = image_background  # Fallback caso ocorra erro

        return output

    def process_and_visualize(self, adjustment=-10, show_contours=True, connectivity=True, connectivity_alpha=0.4, overlay_alpha=0.3, overlay_color=(0, 0, 255)):
        """
        Processo completo de detecção e visualização das fraturas, incluindo conectividade.

        Args:
            adjustment (int, opcional): Ajuste adicional para o limiar de Otsu. Defaults to -10.
            show_contours (bool, optional): Se True, desenha contornos das fraturas. Defaults to True.
            connectivity (bool, optional): Se True, realiza e sobrepõe a conectividade das fraturas. Defaults to True.
            connectivity_alpha (float, optional): Transparência da sobreposição de conectividade. Defaults to 0.4.
            overlay_alpha (float, optional): Transparência da sobreposição de fraturas. Defaults to 0.3.
            overlay_color (tuple, optional): Cor para destacar as fraturas (R, G, B). Defaults to (0, 0, 255).

        Returns:
            dict: Dicionário contendo várias imagens processadas para visualização.
        """
        # Aplicar Otsu com ajuste
        binary_otsu, thresh = self.aplicar_otsu_potencializado(adjustment=adjustment)

        # Sobrepor fraturas na imagem original
        original_np = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        overlay = self.overlay_images(
            image_background=original_np.copy(),
            image_foreground=binary_otsu,
            color=overlay_color,
            alpha=overlay_alpha,
            contour=show_contours
        )

        results = {
            'binary_otsu': binary_otsu,
            'threshold': thresh,
            'overlay_fraturas': overlay
        }

        # Se a conectividade for requerida
        if connectivity:
            colored_labels, count = self.conect_fraturas()
            connectivity_overlay = self.overlay_connectivity(
                image_background=overlay,
                colored_labels=colored_labels,
                alpha=connectivity_alpha
            )
            results['colored_labels'] = colored_labels
            results['connectivity_overlay'] = connectivity_overlay
            results['num_connected_components'] = count

        return results


class AdvancedSegmentation:
    """
    Classe para realizar a segmentação avançada de fraturas complexas em segmentos lineares.
    """
    def __init__(self, skeleton_image, labeled_image, fracture_label, pixel_size_mm, min_length_segment=5):
        """
        Inicializa o segmentador avançado.

        Args:
            skeleton_image (numpy.ndarray): Imagem binária do esqueleto da fratura.
            labeled_image (numpy.ndarray): Imagem rotulada das fraturas.
            fracture_label (int): Label da fratura a ser segmentada.
            pixel_size_mm (float): Tamanho do pixel em mm.
            min_length_segment (float): Comprimento mínimo permitido para segmentos em mm.
        """
        self.skeleton_image = skeleton_image
        self.labeled_image = labeled_image
        self.fracture_label = fracture_label
        self.pixel_size_mm = pixel_size_mm
        self.min_length_segment = min_length_segment
        self.graph = self.build_graph()
        self.segments = []
        self.segments_metrics = []

    def build_graph(self):
        """
        Constrói um grafo a partir do esqueleto da fratura.

        Returns:
            networkx.Graph: Grafo representando o esqueleto.
        """
        G = nx.Graph()
        skeleton_coords = np.column_stack(np.where(self.skeleton_image > 0))
        for coord in skeleton_coords:
            x, y = coord
            G.add_node((x, y))
            # Adiciona arestas para vizinhos 8-conectados
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if (0 <= neighbor[0] < self.skeleton_image.shape[0] and
                        0 <= neighbor[1] < self.skeleton_image.shape[1]):
                        if self.skeleton_image[neighbor] > 0:
                            G.add_edge((x, y), neighbor)
        return G

    def find_branches(self):
        """
        Identifica os nós com grau maior que 2 (bifurcações) no grafo.

        Returns:
            list: Lista de nós bifurcados.
        """
        branches = [node for node, degree in self.graph.degree() if degree > 2]
        return branches

    def find_endpoints(self):
        """
        Identifica os nós com grau igual a 1 (endpoints) no grafo.

        Returns:
            list: Lista de endpoints.
        """
        endpoints = [node for node, degree in self.graph.degree() if degree == 1]
        return endpoints

    def find_segments(self):
        """
        Segmenta o esqueleto em partes lineares com base nas bifurcações.

        Returns:
            list: Lista de segmentos, cada um sendo uma lista de coordenadas.
        """
        branches = self.find_branches()
        endpoints = self.find_endpoints()

        # Se não houver endpoints, consideramos todos os nós como possíveis pontos de início
        if not endpoints:
            endpoints = branches

        for endpoint in endpoints:
            for neighbor in self.graph.neighbors(endpoint):
                path = [endpoint, neighbor]
                current_node = neighbor
                previous_node = endpoint
                while True:
                    neighbors = list(self.graph.neighbors(current_node))
                    if previous_node in neighbors:
                        neighbors.remove(previous_node)
                    if len(neighbors) == 0:
                        break
                    elif len(neighbors) > 1:
                        # Encontra a bifurcação
                        break
                    next_node = neighbors[0]
                    path.append(next_node)
                    previous_node, current_node = current_node, next_node

                # Calcula o comprimento do segmento em mm
                length_pixels = sum(
                    np.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)
                    for i in range(1, len(path))
                )
                length_mm = length_pixels * self.pixel_size_mm
                if length_mm >= self.min_length_segment:
                    self.segments.append(path)

        return self.segments

    def calculate_segment_metrics(self):
        """
        Calcula o comprimento e o ângulo de cada segmento.

        Returns:
            list: Lista de dicionários contendo métricas de cada segmento.
        """
        segment_metrics = []
        for idx, segment in enumerate(self.segments, start=1):
            if len(segment) < 2:
                continue  # Ignorar segmentos muito curtos

            # Calcula o comprimento total do segmento
            length_pixels = 0
            angles = []
            for i in range(1, len(segment)):
                dx = segment[i][1] - segment[i-1][1]
                dy = segment[i][0] - segment[i-1][0]
                distance = np.sqrt(dx**2 + dy**2)
                length_pixels += distance
                angle = np.degrees(np.arctan2(dy, dx)) % 360
                angles.append(angle)
            length_mm = length_pixels * self.pixel_size_mm
            average_angle = np.mean(angles) if angles else 0.0
            
            segment_metrics.append({
                "ID_Segmento": f"S_{self.fracture_label}_{idx}",
                "ID_Fratura": f"F_{self.fracture_label}",
                "Comprimento (mm)": round(length_mm, 2),
                "Orientação (graus)": round(average_angle, 2)
            })
        self.segments_metrics = segment_metrics
        return segment_metrics

    def visualize_segments(self, original_image):
        """
        Visualiza os segmentos na imagem original com identificação.

        Args:
            original_image (numpy.ndarray): Imagem original em BGR.

        Returns:
            numpy.ndarray: Imagem com segmentos visualizados.
        """
        overlay = original_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for idx, segment in enumerate(self.segments, start=1):
            # Cor azul escuro
            color = (139, 0, 0)  # Azul escuro em BGR

            # Desenhar o segmento
            for i in range(1, len(segment)):
                pt1 = (segment[i-1][1], segment[i-1][0])
                pt2 = (segment[i][1], segment[i][0])
                cv2.line(overlay, pt1, pt2, color, 2)

            # Adicionar texto com ID do segmento
            if segment:
                start_pt = (segment[0][1], segment[0][0])
                cv2.putText(overlay, f"S_{self.fracture_label}_{idx}", start_pt, font, font_scale, color, thickness, cv2.LINE_AA)

        return overlay



class FractureCharacterizer:
    """
    Classe para caracterizar as fraturas detectadas, incluindo cálculo de métricas e estatísticas.
    """
    def __init__(self, detection_image, area_total_mm2, image_real, min_length, min_opening, image_angle=0.0, segmentation_enabled=False):
        """
        Inicializa o caracterizador de fraturas.

        Args:
            detection_image (numpy.ndarray): Imagem binarizada com fraturas detectadas.
            area_total_mm2 (float): Área total da imagem em mm².
            image_real (PIL.Image.Image): Imagem original em formato PIL.
            min_length (float): Comprimento mínimo das fraturas em mm.
            min_opening (float): Abertura mínima das fraturas em mm.
            image_angle (float, opcional): Ângulo da imagem em relação ao norte geográfico. Defaults to 0.0.
            segmentation_enabled (bool, opcional): Flag para segmentação avançada. Defaults to False.
        """
        self.detection_image = detection_image
        self.area_total_mm2 = area_total_mm2
        self.image_real = image_real
        self.min_length = min_length
        self.min_opening = min_opening
        self.image_angle = image_angle  # Ângulo da imagem em relação ao norte geográfico
        self.segmentation_enabled = segmentation_enabled  # Flag para segmentação avançada
        self.analysis_results = None
        self.fractal_dimension = None
        self.fractal_stats = None
        # Variáveis adicionais
        self.labeled_image = None
        self.fracture_labels = []
        self.branches_info = {}
        self.segments_info = []  # Lista para armazenar informações dos segmentos
        # Instanciação do ImageProcessorBackend
        self.image_processor = ImageProcessorBackend(self.image_real)

    def calcular_abertura_preciso(self, fracture_binary_image):
        """
        Calcula a abertura mínima, média, máxima e desvio padrão da fratura utilizando a transformada de distância.

        Args:
            fracture_binary_image (numpy.ndarray): Imagem binária da fratura.

        Returns:
            tuple: (abertura_minima_mm, abertura_media_mm, abertura_maxima_mm, abertura_std_mm)
        """
        distance = cv2.distanceTransform(fracture_binary_image.astype(np.uint8), cv2.DIST_L2, 5)
        abertura = distance * 2  # A abertura em cada ponto é duas vezes o valor da distância

        # Filtrar valores extremos se necessário
        abertura = abertura[fracture_binary_image > 0]
        abertura = abertura[abertura > 0]  # Remover zeros

        if len(abertura) == 0:
            return (np.nan, np.nan, np.nan, np.nan)

        # Cálculo robusto da abertura mínima usando o 50º percentil
        abertura_min_pixels = np.percentile(abertura, 50)
        abertura_max_pixels = np.max(abertura)
        abertura_media_pixels = (abertura_min_pixels + abertura_max_pixels) / 2
        abertura_std_pixels = np.std(abertura)

        abertura_min_mm = abertura_min_pixels * self.pixel_size_mm
        abertura_max_mm = abertura_max_pixels * self.pixel_size_mm
        abertura_media_mm = abertura_media_pixels * self.pixel_size_mm
        abertura_std_mm = abertura_std_pixels * self.pixel_size_mm

        return (abertura_min_mm, abertura_media_mm, abertura_max_mm, abertura_std_mm)

    def calcular_orientacao_fratura(self, coords):
        """
        Calcula a orientação da fratura utilizando Análise de Componentes Principais (PCA).

        Args:
            coords (numpy.ndarray): Coordenadas dos pixels da fratura.

        Returns:
            float: Orientação da fratura em graus.
        """
        pca = PCA(n_components=1)
        pca.fit(coords)
        eigenvector = pca.components_[0]
        angle = np.degrees(np.arctan2(eigenvector[1], eigenvector[0]))
        if angle < 0:
            angle += 360
        return angle

    def calcular_dimensao_fractal(self, image):
        """
        Calcula a dimensão fractal da imagem binarizada.

        Args:
            image (numpy.ndarray): Imagem binarizada.

        Returns:
            float: Dimensão fractal.
        """
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k * k))[0])

        Z = (image > 0)
        p = min(Z.shape)
        if p <= 1:
            return np.nan
        n = 2 ** np.floor(np.log(p) / np.log(2))
        if n < 2:
            return np.nan
        n = int(np.log(n) / np.log(2))
        sizes = 2 ** np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        if len(sizes) < 2:
            return np.nan
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def calcular_comprimento_skeleton(self, skeleton_image):
        """
        Calcula o comprimento total da fratura utilizando o esqueleto principal sem linearização.

        Args:
            skeleton_image (numpy.ndarray): Imagem binária do esqueleto da fratura.

        Returns:
            float: Comprimento da fratura em mm.
        """
        # Rotular os componentes conectados no esqueleto
        labeled_skeleton, num_features = label(skeleton_image, connectivity=2, return_num=True)

        if num_features == 0:
            return np.nan

        # Encontrar o componente com o maior número de pixels (esqueleto principal)
        props = regionprops(labeled_skeleton)
        main_skeleton = max(props, key=lambda x: x.area).image

        # Calcular o comprimento como o número de pixels no esqueleto principal
        comprimento_pixels = np.count_nonzero(main_skeleton)
        comprimento_mm = comprimento_pixels * self.pixel_size_mm

        return comprimento_mm

    def analisar_fraturas(self):
        """
        Analisa as fraturas detectadas, calculando métricas como comprimento, abertura,
        orientação, permeabilidade e porosidade.

        Returns:
            tuple: DataFrame com resultados, imagens processadas, estatísticas fractais, Dicionário de segmentos (se avançado).
        """
        # Rotula componentes conectados (fraturas originais)
        labeled_image, num_features = label(self.detection_image > 0, return_num=True, connectivity=2)
        self.labeled_image = labeled_image
        props = regionprops(labeled_image)
        self.fracture_labels = [prop.label for prop in props]

        dados_fraturas = []
        height, width = self.detection_image.shape
        if height == 0 or width == 0:
            # Não é possível usar Streamlit aqui, então retornamos None
            return None, None, None, None

        # Cálculo do tamanho do pixel em mm
        try:
            self.pixel_size_mm = np.sqrt(self.area_total_mm2 / (height * width))
        except ZeroDivisionError:
            return None, None, None, None

        # Cria imagem de fundo preto para desenhar os esqueletos
        imagem_esqueleto = np.zeros((height, width, 3), dtype=np.uint8)  # Fundo preto

        total_fracture_area_mm2 = 0

        for prop in props:
            # Extrai imagem binária da fratura individual
            fracture_binary_image = (labeled_image == prop.label).astype(np.uint8)

            # Calcula as aberturas mínima, média, máxima e desvio padrão para a fratura original
            abertura_min_mm, abertura_media_mm, abertura_max_mm, abertura_std_mm = self.calcular_abertura_preciso(fracture_binary_image)

            # Se a abertura média não for válida ou menor que o mínimo, ignorar a fratura
            if abertura_media_mm < self.min_opening or np.isnan(abertura_media_mm):
                continue

            # Esqueletiza a fratura
            skeleton = skeletonize(fracture_binary_image > 0).astype(np.uint8)

            # Seleciona apenas o esqueleto principal (remove ramos menores)
            comprimento_mm = self.calcular_comprimento_skeleton(skeleton)

            if comprimento_mm < self.min_length or np.isnan(comprimento_mm):
                continue

            # Cálculo da orientação
            coords = np.column_stack(np.where(skeleton > 0))
            if len(coords) < 2:
                orientacao = np.nan
            else:
                orientacao = self.calcular_orientacao_fratura(coords)

            # Ajustar a orientação com o ângulo da imagem
            if not np.isnan(orientacao):
                orientacao = (orientacao + self.image_angle) % 360

            # Área da fratura
            area_pixels = np.sum(fracture_binary_image)
            area_mm2 = area_pixels * (self.pixel_size_mm ** 2)
            total_fracture_area_mm2 += area_mm2

            # Porosidade
            porosidade = area_mm2 / self.area_total_mm2

            # Permeabilidade
            permeabilidade_mm2 = (abertura_media_mm ** 2) / 12
            permeabilidade_darcy = permeabilidade_mm2 * 1e8  # Aproximação para Darcy

            # Cálculo dos Intervalos de Confiança para Abertura Média
            # Assumindo distribuição normal
            # Número de observações é o número de pixels na fratura com abertura > 0
            n = np.count_nonzero(fracture_binary_image)
            if n > 1:
                se_media = abertura_std_mm / np.sqrt(n)
                ic_lower = abertura_media_mm - 1.96 * se_media
                ic_upper = abertura_media_mm + 1.96 * se_media
            else:
                ic_lower = np.nan
                ic_upper = np.nan

            # Cor baseada no comprimento
            color = cm.jet(comprimento_mm / max(1.0, comprimento_mm))[:3]  # RGB
            color = tuple(int(255 * c) for c in color)  # Convert to 0-255 integers

            # Desenha o esqueleto principal na imagem
            coords_skeleton = np.column_stack(np.where(skeleton > 0))
            for coord in coords_skeleton:
                cv2.circle(imagem_esqueleto, (coord[1], coord[0]), 1, color, -1)

            # Adiciona dados da fratura, incluindo a cor
            dados_fraturas.append({
                "ID_Fratura": f"F_{prop.label}",
                "ID_Segmento": "",  # Inicialmente vazio para fraturas originais
                "Comprimento (mm)": f"{comprimento_mm:.2f}",
                "Abertura Mínima (mm)": f"{abertura_min_mm:.4f}",
                "Abertura Média (mm)": f"{abertura_media_mm:.4f}",
                "Abertura Máxima (mm)": f"{abertura_max_mm:.4f}",
                "Abertura Std (mm)": f"{abertura_std_mm:.4f}",
                "Abertura IC Lower (mm)": f"{ic_lower:.4f}" if not np.isnan(ic_lower) else "N/A",
                "Abertura IC Upper (mm)": f"{ic_upper:.4f}" if not np.isnan(ic_upper) else "N/A",
                "Orientação (graus)": f"{orientacao:.2f}" if not np.isnan(orientacao) else "N/A",
                "Área da Fratura (mm²)": f"{area_mm2:.4f}",
                "Porosidade": f"{porosidade:.6f}",
                "Permeabilidade (D)": f"{permeabilidade_darcy:.4e}",
                "Color": f"{color[0]}, {color[1]}, {color[2]}"  # Convertendo tupla para string
            })

            # Segmentação Avançada (opcional)
            if self.segmentation_enabled:
                # Realiza a segmentação avançada usando a classe AdvancedSegmentation
                segmenter = AdvancedSegmentation(
                    skeleton_image=skeleton,
                    labeled_image=labeled_image,
                    fracture_label=prop.label,
                    pixel_size_mm=self.pixel_size_mm,
                    min_length_segment=self.min_length
                )
                segmenter.find_segments()
                segments_metrics = segmenter.calculate_segment_metrics()
                self.segments_info.extend(segments_metrics)

                # Atualiza a imagem com os segmentos identificados
                imagem_esqueleto = segmenter.visualize_segments(imagem_esqueleto) #, segments_metrics)

                # Adiciona os dados dos segmentos à tabela de resultados
                for segment in segments_metrics:
                    dados_fraturas.append({
                        "ID_Fratura": segment["ID_Fratura"],
                        "ID_Segmento": segment["ID_Segmento"],
                        "Comprimento (mm)": f"{segment['Comprimento (mm)']:.2f}",
                        "Abertura Mínima (mm)": "N/A",
                        "Abertura Média (mm)": "N/A",
                        "Abertura Máxima (mm)": "N/A",
                        "Abertura Std (mm)": "N/A",
                        "Abertura IC Lower (mm)": "N/A",
                        "Abertura IC Upper (mm)": "N/A",
                        "Orientação (graus)": f"{segment['Orientação (graus)']:.2f}",  # Usando "Orientação (graus)"
                        "Área da Fratura (mm²)": "N/A",
                        "Porosidade": "N/A",
                        "Permeabilidade (D)": "N/A",
                        "Color": "0, 0, 139"  # Azul Escuro para segmentos
                    })

        if not dados_fraturas:
            # Se nenhum dado de fratura válido foi coletado, retornar None
            return None, None, None, None

        # Converter os dados das fraturas para um DataFrame
        self.analysis_results = pd.DataFrame(dados_fraturas)

        # Definir os tipos de dados, permitindo 'object' para campos que podem conter "N/A"
        self.analysis_results = self.analysis_results.astype({
            "ID_Fratura": str,
            "ID_Segmento": str,
            "Comprimento (mm)": object,
            "Abertura Mínima (mm)": object,
            "Abertura Média (mm)": object,
            "Abertura Máxima (mm)": object,
            "Abertura Std (mm)": object,
            "Abertura IC Lower (mm)": object,
            "Abertura IC Upper (mm)": object,
            "Orientação (graus)": object,  # Incluído para segmentos
            "Área da Fratura (mm²)": object,
            "Porosidade": object,
            "Permeabilidade (D)": object,
            "Color": object
        })

        # Converte campos numéricos onde aplicável
        numeric_fields = [
            "Comprimento (mm)", 
            "Abertura Mínima (mm)", 
            "Abertura Média (mm)", 
            "Abertura Máxima (mm)", 
            "Abertura Std (mm)", 
            "Abertura IC Lower (mm)", 
            "Abertura IC Upper (mm)", 
            "Orientação (graus)",  # Agora padronizado
            "Permeabilidade (D)", 
            "Porosidade"
        ]
        for field in numeric_fields:
            self.analysis_results[field] = pd.to_numeric(
                self.analysis_results[field].replace("N/A", np.nan),
                errors='coerce'
            )

        # Cálculo da dimensão fractal
        self.fractal_dimension = self.calcular_dimensao_fractal(self.detection_image)

        # Estatísticas Fractais
        self.fractal_stats = pd.DataFrame({
            'Dimensão Fractal': [self.fractal_dimension],
            'Densidade de Fraturas (por mm²)': [len(self.fracture_labels) / self.area_total_mm2],
            'Intensidade de Fraturas (mm/mm²)': [self.analysis_results["Comprimento (mm)"].sum() / self.area_total_mm2],
            'Porosidade Total': [total_fracture_area_mm2 / self.area_total_mm2]
        })

        # Rotulação e geração das imagens processadas
        imagem_esqueleto_rgb = cv2.cvtColor(imagem_esqueleto, cv2.COLOR_BGR2RGB)

        # Adiciona imagem sobreposta usando ImageProcessorBackend
        overlay_image_bgr = self.image_processor.overlay_images(
            image_background=np.array(self.image_real),
            image_foreground=self.detection_image,
            color=(0, 0, 255),
            alpha=0.3,
            contour=True
        )
        overlay_image_rgb = cv2.cvtColor(overlay_image_bgr, cv2.COLOR_BGR2RGB)
        self.processed_images = {
            "imagem_colorida": imagem_esqueleto_rgb,
            "overlay_image": overlay_image_rgb
        }


        # **Criação da Imagem com IDs Sobrepostos**
        # Inicializa a imagem_com_numero como uma cópia da overlay_image para manter as fraturas coloridas
        imagem_com_numero = self.processed_images["overlay_image"].copy()

        # Converter para PIL para facilitar a anotação
        imagem_pil = Image.fromarray(imagem_com_numero)
        draw = ImageDraw.Draw(imagem_pil)

        # Tentar carregar uma fonte padrão
        try:
            fonte = ImageFont.truetype("arial.ttf", size=12)
        except IOError:
            fonte = ImageFont.load_default()

        for index, row in self.analysis_results.iterrows():
            # Obter as coordenadas do centro da fratura ou segmento
            fratura_id = row["ID_Fratura"]
            segmento_id = row["ID_Segmento"]
            label_original = segmento_id if segmento_id else row["ID_Fratura"]
            try:
                # Para segmentos, extrai o terceiro elemento; para fraturas, o segundo
                parts = label_original.split('_')
                if segmento_id:
                    label_num = int(parts[2])
                else:
                    label_num = int(parts[1])
            except (IndexError, ValueError):
                continue
            mask = (self.labeled_image == label_num) if segmento_id else (self.labeled_image == label_num)

            # Encontrar contornos para obter o centro
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Desenhar o ID na posição central com a cor correspondente
                    text = segmento_id if segmento_id else fratura_id
                    color_str = row["Color"] if row["Color"] != "N/A" else "255, 0, 0"  # Vermelho padrão para segmentos
                    try:
                        # Converter a string de cor para tupla RGB
                        color = tuple(map(int, color_str.split(',')))
                    except:
                        color = (255, 0, 0)  # Vermelho padrão caso falhe a conversão
                    draw.text((cX, cY), text, fill=color, font=fonte)

        # Converter de volta para NumPy array
        imagem_com_numero = np.array(imagem_pil)

        # Adicionar a imagem com IDs ao dicionário de imagens processadas
        self.processed_images["imagem_com_numero"] = imagem_com_numero

        return self.analysis_results, self.processed_images, self.fractal_stats, self.segments_info if self.segmentation_enabled else None
