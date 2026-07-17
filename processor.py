import base64
import io
import json
import re
from openai import OpenAI
from PIL import Image
import fitz  # PyMuPDF (Não precisa de Poppler!)
import streamlit as st

# CONFIGURAÇÃO DA API
st.set_page_config(page_title="Cortador de Caixas de PDF", layout="centered")
st.title("📦 Extrator e Cortador de Caixas (PDF) - joão victor lindo")
st.write("Insira o PDF com as páginas dos medicamentos. O sistema vai extrair, recortar e formatar em 500x500 com fundo branco.")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def image_to_base64(pil_image):
    """Converte uma imagem do Pillow diretamente para Base64 sem salvar no disco."""
    buffered = io.BytesIO()
    # Qualidade alta preserva os detalhes que a IA usa para localizar as bordas.
    pil_image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def identificar_caixas_na_pagina(pil_image):
    """Envia a página do PDF para o GPT-4o e pede a lista de coordenadas das caixas."""
    base64_image = image_to_base64(pil_image)
    width, height = pil_image.size

    prompt = (
        = (
    f"Esta imagem tem exatamente {width} pixels de largura e {height} pixels de altura. "
    "Analise a página e identifique todas as caixas de remédio/produtos presentes. "
    "Para cada caixa, retorne uma bounding box ampla que envolva a caixa INTEIRA e "
    "também QUALQUER TEXTO ou título do medicamento que esteja escrito logo acima ou ao lado dela. "
    "Não economize espaço: é melhor deixar a caixinha maior do que cortar as bordas ou os textos do produto. "
    "Retorne estritamente um array JSON com as coordenadas na escala de 0 a 1000..."
)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        # "detail": "high" evita que a OpenAI reduza a imagem,
                        # melhorando muito a precisão das coordenadas.
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
    resposta_limpa = re.sub(r"```json|```", "", resposta_texto).strip()
    return json.loads(resposta_limpa)

def formatar_para_500x500(imagem_cortada):
    """Coloca o recorte no centro de uma imagem de 500x500 com fundo branco."""
    fundo_branco = Image.new("RGB", (500, 500), (255, 255, 255))
    
    # Redimensiona a imagem cortada mantendo a proporção para caber em 500x500
    imagem_cortada.thumbnail((480, 480), Image.Resampling.LANCZOS)
    
    # Calcula a posição para centralizar o remédio no fundo branco
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
                # Abre o PDF diretamente da memória usando o PyMuPDF
                pdf_bytes = uploaded_file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                st.success(f"PDF carregado com sucesso! Encontrada(s) {len(doc)} página(s).")
                contador_caixas = 0
                
                # Percorre cada página do arquivo
                for i in range(len(doc)):
                    st.markdown(f"### Analisando Página {i+1}...")
                    page = doc.load_page(i)
                    
                    # Renderiza a página em alta definição (Matrix 2.0 melhora a nitidez para a IA ler)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img_data = pix.tobytes("png")
                    pagina_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                    
                    width, height = pagina_pil.size
                    
                    # Busca as caixas daquela página via API
                    lista_coordenadas = identificar_caixas_na_pagina(pagina_pil)
                    
                    # Debug: mostra o que a IA retornou (útil para validar os recortes)
                    with st.expander(f"🔍 Coordenadas retornadas pela IA (Página {i+1})"):
                        st.json(lista_coordenadas)

                    if not lista_coordenadas:
                        st.warning(f"Nenhuma caixa detectada na página {i+1}.")
                        continue
                        
                    # Recorta cada caixa detectada na página atual
                    for coords in lista_coordenadas:
                        contador_caixas += 1
                        
                        # Converte escala 0-1000 para pixels reais
                        ymin = int((coords["ymin"] / 1000) * height)
                        xmin = int((coords["xmin"] / 1000) * width)
                        ymax = int((coords["ymax"] / 1000) * height)
                        xmax = int((coords["xmax"] / 1000) * width)

                        # Margem de segurança: se a IA errar um pouco a borda,
                        # a caixa não vem cortada. 3% da dimensão da página.
                        margem_x = int(width * 0.10)
                        margem_y = int(height * 0.10)
                        xmin = max(0, xmin - margem_x)
                        ymin = max(0, ymin - margem_y)
                        xmax = min(width, xmax + margem_x)
                        ymax = min(height, ymax + margem_y)

                        # Evita cortes inválidos
                        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                            continue
                            
                        # Faz o Crop do remédio
                        recorte = pagina_pil.crop((xmin, ymin, xmax, ymax))
                        
                        # Aplica o fundo branco e redimensiona para 500x500
                        imagem_final = formatar_para_500x500(recorte)
                        
                        # Mostra o resultado final na tela
                        st.image(imagem_final, caption=f"Caixa {contador_caixas} (Página {i+1})", width=250)
                        
                        # Botão de download individual
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
