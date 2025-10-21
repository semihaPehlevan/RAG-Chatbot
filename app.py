# app.py

# Gerekli Kütüphaneler
import os
import json
import gradio as gr
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
from google import genai
from google.genai.errors import APIError
from langchain.embeddings import HuggingFaceEmbeddings
import sys
import time
from dotenv import load_dotenv

# =============================================
# 1. Kurulum ve Ortam Ayarları
# =============================================

# Colab ortamında olup olmadığımızı kontrol et
IN_COLAB = 'google.colab' in sys.modules

# ---------------------------------------------
# API Anahtarını Çekme ve Ortama Yükleme
# ---------------------------------------------

if IN_COLAB:
    # Bu blok, yerel bir Python dosyasında genellikle çalışmaz, 
    # ancak Colab'de çalıştırılma ihtimaline karşı kodu temiz tutmak için bırakılabilir.
    # Yerel ortamda drive mount edilmez.
    # Colab'e özgü drive.mount() ve userdata.get() satırları BURADAN SİLİNMİŞTİR.
    
    # Varsayım: Eğer bu dosya Colab'de çalışıyorsa, API anahtarı zaten ortamda set edilmiştir.
    JSON_PATH = "/content/drive/MyDrive/Datasets/SSS.json" # Colab yolu
    
    # Uyarı: Bu .py dosyasını Colab'de çalıştırıyorsanız, drive.mount ve userdata'yı manuel olarak
    # Colab not defterinizde yapmanız veya kodu .ipynb'de bırakmanız önerilir.

else:
    # Yerel ortamda isek, .env dosyasını yükle
    load_dotenv()
    
    # GOOGLE_API_KEY'i doğrudan .env'den çek
    if os.getenv('GOOGLE_API_KEY'):
        print("API anahtarı .env dosyasından yüklendi.")
    else:
        # Hata mesajı. GOOGLE_API_KEY'in .env'de tanımlı olduğundan emin olun.
        print("Hata: GOOGLE_API_KEY .env dosyasında bulunamadı. Lütfen .env dosyanızı kontrol edin.")

    # Yerel dosya yolu
    JSON_PATH = "data/SSS.json" 
    # Lütfen SSS.json dosyasının bu yolda (data klasörü içinde) olduğundan emin olun.


# SSS Tam Eşleşme Sözlüğü için global tanımlanır
SSS_LOOKUP_DICT = {}
client = None # Global olarak tanımlanır, başlatma aşağıda yapılır.

# Gemini client başlatma
# GOOGLE_API_KEY ortam değişkeni set edildiyse Client otomatik olarak bulacaktır
try:
    if os.getenv('GOOGLE_API_KEY'):
        client = genai.Client()
        print("Gemini istemcisi başarıyla başlatıldı.")
    else:
        print("Gemini istemcisi başlatılamadı: API anahtarı mevcut değil.")
except APIError as e:
    print(f"Hata: Gemini Client başlatılamadı (API Hatası): {e}")
except Exception as e:
    print(f"Hata: Gemini Client başlatılamadı: {e}")

# =============================================
# 2. Veri İşleme ve Vektör Veritabanı (FAISS)
# =============================================

# ... (load_data_from_json ve create_vector_store fonksiyonları buraya aynen kopyalanacak) ...

def load_data_from_json(json_path):
    """
    Belirtilen JSON dosyasını okur, veriyi RAG için Document'lara ve
    tam eşleşme için global sözlüğe hazırlar. Hata yönetimi içerir.
    """
    global SSS_LOOKUP_DICT
    docs = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            sss_data = json.load(f)

        for entry in sss_data:
            q = entry.get("Question", "").strip()
            a = entry.get("Answer", "").strip()

            if not q or not a:
                continue

            content = f"Soru: {q}\nYanıt: {a}"
            docs.append(Document(
                page_content=content,
                metadata={"source_type": "Q&A"}
            ))

            SSS_LOOKUP_DICT[q] = a

    except Exception as e:
        print(f"Hata: JSON yüklenemedi: {e}. JSON yolu: {json_path}") # JSON_PATH'i kontrol etmek için eklendi

    print(f"Toplam {len(docs)} belge (Q&A) işlendi.")
    return docs


def create_vector_store(docs):
    """
    Girdi olarak alınan belgeleri (Documents) kullanarak metinleri vektörlere dönüştürür
    ve bu vektörleri FAISS (Vektör Veritabanı) içinde indeksler.
    """

    print("Embedding modeli yükleniyor ve vektör veritabanı oluşturuluyor...")

    # Türkçe RAG sistemleri için önerilen HuggingFace Embedding modeli.
    embedding_function = HuggingFaceEmbeddings(model_name="trmteb/turkish-embedding-model")

    if not docs:
        raise ValueError("Vektör veritabanı oluşturulamadı: Giriş belgesi (docs) listesi boş.")

    # FAISS veritabanını oluşturur.
    faiss_db = FAISS.from_documents(
        docs,
        embedding_function,
        distance_strategy=DistanceStrategy.COSINE
    )
    print("Vektör veritabanı başarıyla oluşturuldu.")
    return faiss_db


# =============================================
# 3. HyDE ve RAG Fonksiyonları
# =============================================

# ... (expand_question_hyde, retrieve_context ve generate_response fonksiyonları buraya aynen kopyalanacak) ...

# Kullanıcıların sıkça kullandığı kısaltmaları veya spesifik terimleri,
# vektör aramasını iyileştirmek için uzun ve açıklayıcı terimlerle eşler.
KEYWORD_MAP = {
    "çap": "çift anadal programı",
    "yandal": "yandal programı sertifika",
    "dgs": "dikey geçiş sınavı kontenjan"
}

#  HyDE: Hipotetik Yanıt Üretimi
def expand_question_hyde(question):
    """
    Bu teknikte, kullanıcı sorusu LLM'e gönderilerek, sanki bir SSS belgesinden gelmiş gibi
    olası bir yanıt taslağı (hipotetik belge) üretilir.
    """

    global client
    if client is None:
        return f"Hipotetik yanıt: {question}"

    expansion_prompt = f"""
Aşağıdaki soruya yönelik, elinizde bir yönerge veya SSS verisi varmış gibi kısa,
ancak içerik olarak detaylı ve olası bir yanıt taslağı (hipotetik belge) oluşturun.
Yanıt doğru olmak zorunda değil, arama için anahtar kelimeler yeterlidir.

Soru: {question}

Hipotetik Yanıt:
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=expansion_prompt,
            config={"max_output_tokens": 200}
        )
        return response.text.strip() if response and response.text else f"Hipotetik yanıt: {question}"
    except Exception as e:
        print(f"HyDE Hatası: {e}")
        return f"Hipotetik yanıt: {question}"

# Bağlam Retrieval

def retrieve_context(faiss_db, question, k=5, max_tokens=5000):
    """
    Geri Çağırma (Retrieval) aşamasını gerçekleştirir.
    """

    # 1. HyDE ile hipotetik yanıtı al.
    hypothetical_answer = expand_question_hyde(question)
    search_query = f"{question} {hypothetical_answer}"

    # 2. Manuel anahtar kelime enjeksiyonu:
    q_lower = question.lower()
    for key, val in KEYWORD_MAP.items():
        if key in q_lower:
            search_query += f" {val}"

    # 3. Vektör araması:
    retrieved_docs = faiss_db.similarity_search(search_query, k=k)

    final_docs = []

    # 4. Özel kısaltma önceliklendirme:
    if any(key in q_lower for key in KEYWORD_MAP.keys()):
        relevant_docs = [
            doc for doc in retrieved_docs
            if any(val.lower() in doc.page_content.lower() for val in KEYWORD_MAP.values())
        ]
        final_docs.extend(relevant_docs[:3])
        remaining = [d for d in retrieved_docs if d not in final_docs]
        final_docs.extend(remaining[:k - len(final_docs)])
    else:
        final_docs.extend(retrieved_docs[:k])

    # 5. Context birleştirme:
    context_chunks = []
    for idx, doc in enumerate(final_docs[:k], 1):
        context_chunks.append(f"[Doküman {idx}]\n{doc.page_content}")

    context = "\n---\n".join(context_chunks)

    # 6. Token bazlı kesme:
    if len(context) > max_tokens:
        context = context[:max_tokens]

    return context


#  LLM ile RAG Yanıt Üretimi

def generate_response(question, context, retry_count=3, wait_sec=2):
    """
    RAG'ın Üretim (Generation) aşamasıdır.
    """

    global client
    if client is None:
        return "Hata: Gemini API istemcisi başlatılamadı."
    if not context.strip():
        return "Bağlam boş veya uygun bilgi bulunamadı. Lütfen .env dosyasını ve JSON yolunu kontrol edin."

    prompt = f"""
Aşağıda birden fazla bağlam parçası verilmiştir.
Lütfen yalnızca verilen bağlamdan yararlanarak yanıt üretin. Ek bilgi eklemeyin.
Türkçe ve profesyonel bir tonda yanıtlayın.

Bağlam:
{context}

Soru:
{question}

Cevap:
"""

    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"max_output_tokens": 1024}
            )
            if response and response.text:
                return response.text.strip()
            else:
                return "Yanıt oluşturulamadı."
        except Exception as e:
            print(f"Gemini API Hatası (deneme {attempt+1}): {e}")
            if attempt < retry_count - 1:
                time.sleep(wait_sec)
            else:
                return f"Gemini API Hatası: {e}"


# =============================================
# 4. Gradio Arayüzü ve Ana İş Akışı
# =============================================

# Veri yükleme ve Vektör DB oluşturma (Uygulama başladığında bir kez yapılır)
try:
    all_documents = load_data_from_json(JSON_PATH)
    faiss_db = create_vector_store(all_documents)
except Exception as e:
    print(f"KRİTİK HATA: Veritabanı oluşturulamadı. {e}")
    faiss_db = None

def chat_fn(question):
    """
    Kullanıcı sorusunu işleyen ana fonksiyondur (Hibrit Arama: Tam Eşleşme + HyDE RAG).
    """

    global faiss_db, SSS_LOOKUP_DICT

    if faiss_db is None:
        return "GENEL HATA: Veritabanı yüklenemedi. Lütfen konsolu kontrol edin."

    # Tam Eşleşme Kontrolü
    normalized_question = question.strip()

    if normalized_question in SSS_LOOKUP_DICT:
        # Eğer tam eşleşme varsa, direkt sözlükteki cevabı döndür
        return SSS_LOOKUP_DICT[normalized_question] 
    
    # RAG Süreci
    context = retrieve_context(faiss_db, question, k=5)
    answer = generate_response(question, context)

    # Tam eşleşme olmadığı için sadece RAG cevabını döndür.
    return answer

# --- Gradio Arayüzü Tanımlama ---

# Kullanılacak bileşenler
with gr.Blocks(title="SAÜ ÇAP-YANDAL-MUAFİYET ASİSTANI") as demo:
    gr.Markdown(
        """
        # Sakarya Üniversitesi RAG Asistanı
        Sakarya Üniversitesi Çap-Yandal Yönergesi, Muafiyet-İntibak Yönergesi ve Yatay Geçiş Yönergesi kullanılmıştır.
        """
    )

    # Giriş metin kutusu
    question_input = gr.Textbox(
        label="Sorunuzu Buraya Girin",
        lines=2,
        placeholder="Bir soru yazın..."
    )

    # Çıkış metin kutusu
    response_output = gr.Textbox(
        label="Yanıt",
        lines=15,
        interactive=False
    )

    # Gönder butonu
    submit_btn = gr.Button("Soru Sor", variant="primary")

    # Olayları bağlama
    submit_btn.click(
        fn=chat_fn,
        inputs=question_input,
        outputs=response_output
    )

    # Örnekler
    gr.Examples(
        examples=[
            "ÇAP başvurusu nasıl yapılır?",
            "Yandal programını tamamlamak için gerekli GNO şartı nedir?",
            "Yandal programı sertifikası almak için ne yapmalıyım?",
            "Dikey Geçiş (DGS) yolu ile kayıt yaptıran öğrenci hangi yarıyıldan öğrenime başlar?"
        ],
        inputs=question_input
    )

# Arayüzü başlat
if __name__ == "__main__":
    print("\n Gradio Arayüzü Başlatılıyor...")
    # debug=True, local olarak çalışırken hataları görmek için iyidir.
    # share=True, arayüzü geçici bir genel link üzerinden paylaşır (isteğe bağlı).
    demo.launch(debug=True)