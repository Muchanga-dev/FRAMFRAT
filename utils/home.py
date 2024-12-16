# utils/home.py
import streamlit as st
from PIL import Image

def render_home(image):
    co0, co1, co2 = st.columns(3)
    with co1:
        logo = Image.open(image)
        st.image(logo)
        st.write("____")
    st.markdown("<h3 style='text-align: center; color: Green;'>UNIVERSIDADE FEDERAL DE PERNAMBUCO (UFPE)</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Programa de Pós Graduação em Engenharia Civil</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Instituto de Pesquisa em Petróleo e Energia (LITPEG)</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Laboratório de Métodos Computacionais em Geomecânica (LMCG)</h5>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h3 style='text-align: center; color: Green;'>Autor</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Armando Muchanga</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>armando.muchanga@ufpe.br</h6>", unsafe_allow_html=True) 

    st.markdown("<h3 style='text-align: center; color: Green;'>Orientadores</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Prof. Dr. Igor Fernandes Gomes</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Prof. Dr. José Antônio Barbosa</h6>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: Green;'>Colaboradores</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Dr. Osvaldo Correia</h6>", unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center; color: Green;'>Contatos</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>LMCG/LITPEG - UFPE, 5º Andar</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Av. da Arquitetura - Cidade Universitária, Recife - PE</h6>", unsafe_allow_html=True)

    return  
