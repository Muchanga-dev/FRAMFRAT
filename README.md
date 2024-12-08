# FRAMFRAT: Ferramenta de Detecção e Análise de Fraturas em Materiais Rochosos por Processamento de Imagem Digital

FRAMFRAT é uma ferramenta avançada para detecção e caracterização de fraturas em materiais geológicos utilizando processamento de imagens digitais. Desenvolvida em Python, a ferramenta permite que usurio carreguem imagens, apliquem ajustes de pré-processamento e realizem análises geométricas detalhadas de fraturas. A ferramenta fornece recursos avançados para visualização interativa e exportação de resultados, facilitando estudos geológicos e de engenharia de materiais.

## Destaques

- **Detecção Automática:** Identificação e esqueleto das fraturas em imagens digitais.
- **Análise Geométrica Avançada:** Cálculo de abertura, comprimento, porosidade, conectividade e permeabilidade.
- **Dimensão Fractal:** Ferramenta integrada para determinar a complexidade fractal das fraturas.
- **Visualização Interativa:** Diagramas de orientação, estereogramas e gráficos estatísticos.
- **Exportação de Dados:** Resultados prontos para análise em formatos como CSV e Excel.

## Pré-Requisitos

Certifique-se de que Python 3.10 ou superior esteja instalado. Para verificar, execute:

```bash
python3 --version
```

Caso não esteja instalado, faça o download em [python.org](https://www.python.org/).

## Instalação

### 1. Clone o Repositório

```bash
git clone https://github.com/Muchanga-dev/FRAMFRAT.git
cd FRAMFRAT
```

### 2. Crie e Ative um Ambiente Virtual (Recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows, use venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Execute a Aplicação

Inicie a aplicação diretamente com o comando:

```bash
python run.py
```

### 2. Interface do Usuário

- **Upload de Imagens:** Carregue imagens nos formatos suportados (PNG, JPG).
- **Pré-Processamento:** Ajuste brilho, contraste, recorte e rotação.
- **Detecção de Fraturas:** Utilize algoritmos integrados para identificação automática de fraturas.
- **Caracterização Avançada:** Realize análises geométricas e calcule propriedades específicas.
- **Visualização e Exportação:** Gere relatórios visuais e salve os resultados em Excel ou CSV.

## Funcionalidades Principais

### Detecção Avançada

- Segmentação automática utilizando thresholding adaptativo.
- Esqueletização para cálculo de conectividade.

### Análises Estatísticas

- Cálculo da dimensão fractal e intensidade de fraturas.
- Estatísticas detalhadas de abertura e comprimento.

### Visualizações Interativas

- Gráficos de distribuição e orientação.
- Estereogramas e diagramas de roseta.

### Exportação Personalizável

- Relatórios prontos para análise externa em formatos populares.

## Resolução de Problemas

### Erros Comuns

#### Carregamento de Arquivos

- Certifique-se de que os formatos de imagem são compatíveis.

#### Dependências

- Verifique se todas as bibliotecas foram instaladas corretamente:

```bash
pip install -r requirements.txt
```

## Contribuindo com FRAMFRAT

Contribuições são bem-vindas! Para participar:

1. Faça um fork do repositório.
2. Crie uma nova branch:

    ```bash
    git checkout -b feature-minha-melhoria
    ```

3. Envie suas mudanças e abra um pull request.

## Licença

FRAMFRAT é disponibilizado sob a Licença Apache 2.0. Consulte o arquivo [LICENSE](LICENSE) para mais informações.

## Contato

**Autor:** Armando Muchanga  
**Instituição:** Universidade Federal de Pernambuco (UFPE)  
**Email:** [armando.muchanga@ufpe.br](mailto:armando.muchanga@ufpe.br)
