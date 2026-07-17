import base64
import io
import json
import re
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st

# CONFIGURAÇÃO DA API
st.set_page_config(page_title="Cortador de Caixas de PDF", layout="centered")
st.title("📦 Extrator e Cortador de Caixas (PDF) - joão victor lindo")
st.write("Insira o PDF com as páginas dos medicamentos. O sistema vai extrair, recortar e formatar em 500x500 com fundo branco.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def image_to_base64(pil_image):
    """Converte uma imagem do Pillow diretamente para Base64 sem salvar no disco."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def identificar_caixas_na_pagina(pil_image):
    """Envia a página do PDF para o GPT-4o e pede a lista de coordenadas das caixas."""
    base64_image = image_to_base64(pil_image)
    width, height = pil_image.size

    prompt = (
        f"Esta imagem tem exatamente {width} pixels de largura e {height} pixels de altura. "
        "Analise a página e identifique todas as caixas de remédio/produtos presentes. "
        "Para cada caixa, retorne uma bounding box ampla que envolva a caixa INTEIRA e "
        "também QUALQUER TEXTO ou título do medicamento que esteja escrito logo acima ou ao lado dela. "
        "Não economize espaço: é melhor deixar a caixinha maior do que cortar as bordas ou os textos do produto. "
        "Retorne estritamente um array contendo objetos com as chaves exatas: 'ymin', 'xmin', 'ymax', 'xmax'. "
        "Use a escala de 0 a 1000 para as coordenadas."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        # Força o modelo a responder estritamente com um objeto ou array JSON válido
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        temperature=0.0
    )
    
    resposta_texto = response.choices[0].message.content
    dados = json.loads(resposta_texto)
    
    # O gpt-4o usando json_object costuma encapsular o array dentro de uma chave principal.
    # Vamos caçar o array de coordenadas dentro da resposta de forma inteligente.
    if isinstance(dados, dict):
        for chave, valor in dados.items():
            if isinstance(valor, list):
                return valor
        if "coordenadas" in dados:
            return dados["coordenadas"]
        if "caixas" in dados:
            return dados["caixas"]
    
    return dados if isinstance(dados, list) else []

def formatar_para_500x500(imagem_cortada):
    """Coloca o recorte no centro de uma imagem de 500x500 com fundo branco."""
    fundo_branco = Image.new("RGB", (500, 500), (255, 255, 255))
    imagem_cortada.thumbnail((480, 480), Image.Resampling.LANCZOS)
    
    x_offset = (500 - imagem_cortada.width) // 2
    y_offset = (500 - imagem_cortada.height) // 2
    
    fundo_branco.paste(imagem_cortada, (x_offset, y_offset))
    return fundo_branco

# --- INTERFACE STREAMLIT ---
uploaded_file = st.file_uploader("Escolha o arquivo PDF contendo os medicamentos...", type=["pdf"])

if uploaded_file is not None:
    if st.button("Processar PDF e Recortar Caixas"):
        with st.spinner("Convertendo páginas do PDF e processando com Inteligência Artificial..."):
            try:
                pdf_bytes = uploaded_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                st.success(f"PDF carregado com sucesso! Encontrada(s) {len(doc)} página(s).")
                contador_caixas = 0
                
                for i in range(len(doc)):
                    st.markdown(f"### Analisando Página {i+1}...")
                    page = doc.load_page(i)
                    
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_data = pix.tobytes("png")
                    pagina_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    width, height = pagina_pil.size
                    lista_coordenadas = identificar_caixas_na_pagina(pagina_pil)
                    
                    with st.expander(f"🔍 Coordenadas retornadas pela IA (Página {i+1})"):
                        st.json(lista_coordenadas)

                    if not lista_coordenadas or not isinstance(lista_coordenadas, list):
                        st.warning(f"Nenhuma caixa detectada na página {i+1}.")
                        continue
                        
                    for coords in lista_coordenadas:
                        # TRAVA DE SEGURANÇA: Garante que as 4 chaves necessárias existem no dicionário
                        if not all(k in coords for k in ["ymin", "xmin", "ymax", "xmax"]):
                            continue
                            
                        contador_caixas += 1
                        
                        # Converte escala 0-1000 para pixels reais
                        ymin = int((coords["ymin"] / 1000) * height)
                        xmin = int((coords["xmin"] / 1000) * width)
                        ymax = int((coords["ymax"] / 1000) * height)
                        xmax = int((coords["xmax"] / 1000) * width)

                        # Margem de segurança de 8%
                        margem_x = int(width * 0.08)
                        margem_y = int(height * 0.08)
                        xmin = max(0, xmin - margem_x)
                        ymin = max(0, ymin - margem_y)
                        xmax = min(width, xmax + margem_x)
                        ymax = min(height, ymax + margem_y)

                        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                            continue
                            
                        recorte = pagina_pil.crop((xmin, ymin, xmax, ymax))
                        imagem_final = formatar_para_500x500(recorte)
                        
                        st.image(imagem_final, caption=f"Caixa {contador_caixas} (Página {i+1})", width=250)
                        
                        img_byte_arr = io.BytesIO()
                        imagem_final.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        st.download_button(
                            label=f"Baixar Caixa {contador_caixas}",
                            data=img_byte_arr,
                            file_name=f"caixa_remedio_{contador_caixas}.jpg",
                            mime="image/jpeg"
                        )
                        st.write("---")
                        
                st.success(f"Processamento concluído! Total de caixas extraídas: {contador_caixas}")
                
            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {e}")
