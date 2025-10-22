# 🎓 SAÜ RAG Chatbot: Gemini Destekli Yönergeler Asistanı  

**Deploy Linki:** [https://huggingface.co/spaces/SemihaPehlevan/rag-chatbot-demo](https://huggingface.co/spaces/SemihaPehlevan/rag-chatbot-demo)

---


## 📘 Proje Özeti  
Bu proje, **Sakarya Üniversitesi'nin ÇAP, Yandal, Muafiyet ve Yatay Geçiş** gibi yönergelerini temel alarak kullanıcı sorularına yanıt üreten **RAG (Retrieval-Augmented Generation)** tabanlı bir sohbet asistanıdır.  
Sistem, **Google Gemini 2.5 Flash** modelini kullanarak yönergelerdeki ilgili maddeleri bağlam olarak çekip anlamlı yanıtlar üretir.

| ![Uygulama Çalışma Akışı](https://github.com/semihaPehlevan/RAG-Chatbot/raw/main/arayuz.gif) |  
|:--:|  

### 💬 Örnek Kullanıcı Soruları  

Sistem, Sakarya Üniversitesi yönergelerine dair çeşitli türde sorulara yanıt verebilir.  
Aşağıda örnek sorgular yer almaktadır:  

```bash
Çift anadal programına başvuru şartları nelerdir?  
ÇAP öğrencisi bir dönemde anadal derslerine ilave olarak en fazla kaç AKTS ders alabilir?  
Ders muafiyet başvurusu hangi tarihlerde yapılır?  
Öğrencinin yandal programından kaydı hangi durumda silinir?  
Kredi ve not transferi için azami AKTS limitleri nedir?  
Yandal programı kaç AKTS ve kaç dersten oluşur?  
```

---


## 📊 Veri Seti Hakkında Bilgi  

**Kaynak:** Sakarya Üniversitesi’nin resmi yönerge metinleri (ÇAP, Yandal, Muafiyet, Yatay Geçiş).  

**Format:**  
Veriler, RAG için optimize edilmiş **Soru-Cevap (Q&A)** formatında hazırlanmış olup  
`data/SSS.json` dosyasında saklanmaktadır.  

**Hazırlık Metodolojisi:**  
Her yönerge maddesi, kullanıcıların doğal dilde sorabileceği sorularla eşleştirilerek  
manuel biçimde **atomik bilgi birimlerine** dönüştürülmüştür.  

---


## ⚙️ Çözüm Mimarisi & Kullanılan Teknolojiler  

| Bileşen | Teknoloji |
|----------|------------|
| **RAG Çatısı** | LangChain|
| **LLM (Dil Modeli)** | Google Gemini 2.5 Flash |
| **Embedding Modeli** | `trmteb/turkish-embedding-model` |
| **Vektör Vetitabanı** | FAISS |
| **Arayüz** | Gradio |  

---


## 🔍 RAG Akışı ve Sorgu Optimizasyonu  

Sistem, en alakalı bilgiyi elde etmek için **iki aşamalı bir optimizasyon** süreci uygular:  

### 1. **Sorgu Zenginleştirme (HyDE)**  
Kullanıcının sorusu, **Gemini** modeline gönderilerek olası bir yanıt taslağı oluşturulur.  
Bu taslak, orijinal sorguyla birleştirilir ve **vektörel arama hassasiyeti** önemli ölçüde artırılır.  

### 2. **Hibrit Arama (FAISS + Kısaltma Haritası)**  
Zenginleştirilmiş sorgu, **FAISS** üzerinde taranırken aynı anda sistemde tanımlı **kısaltma haritası** (örnek: `ÇAP → Çift Anadal Programı`) devreye girer.  
Bu yaklaşım, hem **anlamsal benzerlik** hem de **kesin anahtar kelime eşleşmesi** sağlayarak en ilgili yönerge maddelerinin çekilmesini garantiler.



---

## 🧩 Gereksinimler  
- Python 3.8 veya üzeri  
- Google Gemini API Anahtarı  

---



## 🚀 Kurulum  

### 1. Projeyi Klonlayın  
```bash
git clone https://github.com/semihaPehlevan/RAG-Chatbot.git
cd RAG-Chatbot

```
### 2.Bağımlılıkları Kurun 
```bash
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarlayın  

.env dosyasına aşağıdaki satırı ekleyin:
```bash
GOOGLE_API_KEY="GERÇEK_GEMINI_API_ANAHTARINIZI_YAZIN"
```

### 4. Uygulamayı Başlatın
```
python app.py

```
Terminalde görüntülenen bağlantıyı tarayıcınızda açın:

---


## 🔧 RAG Konfigürasyonu  

| Parametre | Değer |
|------------|--------|
| **Generative Model** | `gemini-2.5-flash` |
| **Embedding Model** | `trmteb/turkish-embedding-model` |
| **Retrieval k** | `5` |

---


## 📂 Proje Yapısı  

```bash
RAG-Chatbot/
├── data/
│   └── SSS.json             # ⚠️ Yönerge verileri (Soru-Cevap formatında JSON)
├── app.py                   # Gradio arayüzü + RAG akışı
├── requirements.txt         # Proje bağımlılıkları
├── .env                     # GOOGLE_API_KEY (yerel kullanım için)
└── README.md                # Proje dokümantasyonu
```

---

## ✨ Elde Edilen Sonuçlar  
 
  RAG mimarisi sayesinde yanıtlar yalnızca yönerge metinlerine dayanır; LLM kaynaklı hatalar en aza indirilir.  
  Türkçe için optimize edilmiş embedding modeli ve Hibrit Arama (HyDE + Kısaltma Haritası) kullanılarak en alakalı Soru-Cevap dokümanı hedeflenir. 
  Proje, yanıt tutarlılığını ve kapsamını artırmak amacıyla geliştirilmeye devam edilecektir.  
