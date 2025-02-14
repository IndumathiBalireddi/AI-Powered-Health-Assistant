# 🤖 AI Healthcare Assistant Chatbot 🏥

## 🌟 Overview
The **AI Healthcare Assistant** is a chatbot designed to provide medical information and assist users in understanding symptoms, medications, and general health advice. It utilizes state-of-the-art **Natural Language Processing (NLP)** models for intent classification and medical question answering.

## ✨ Features

- **🏷 Medical Intent Classification:** Uses DistilBERT to classify user queries into different medical intents.
- **💡 Healthcare Q&A:** Employs FLAN-T5 for generating detailed medical responses.
- **🩺 Symptom Checker with Confidence Score** *(Upcoming)*
- **🔍 Follow-up Questions for Better Diagnosis** *(Upcoming)*
- **👤 Personalized Health Advice Based on User Profile** *(Upcoming)*
- **📄 Medical Article Summarization** *(Upcoming)*
- **🎤 Speech-to-Text and Text-to-Speech Support** *(Upcoming)*
- **🌍 Multilingual Response Support** *(Upcoming)*

---

## 🛠 Tech Stack

**Programming Language:** Python 🐍

**Frontend:** Streamlit 🎨

**Machine Learning Models:**
- 🤖 `distilbert-base-uncased` (Intent Classification)
- 🏥 `google/flan-t5-small` (Medical Question Answering)
- 🔬 `microsoft/BioGPT` *(Planned for future enhancement)*

**Libraries:**
- 🏗 `torch`
- 📚 `transformers`
- 📖 `nltk`
- 🌐 `streamlit`
- 🔍 `re` (for text preprocessing)

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-healthcare-assistant.git
cd ai-healthcare-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
# Run the chatbot
streamlit run app.py
```

- Open the chatbot interface in your browser.
- Enter your medical question or symptoms. 📝
- The chatbot will classify the intent and generate a response. 🤖
- *(Planned)* Receive follow-up questions for more accurate suggestions. 💬

---

## ⚠️ Disclaimer
This **AI Healthcare Assistant** is for informational purposes only and should **not** be used as a substitute for professional medical advice. Always consult a **doctor** for medical concerns. 👨‍⚕️👩‍⚕️

---

## 🔮 Future Enhancements

- **Adding `microsoft/BioGPT` for enhanced medical Q&A.** 🧬
- **Implementing a symptom checker with confidence scores.** 🏥
- **Enabling voice-based interactions.** 🎙
- **Expanding support for multilingual responses.** 🌍

---

## 👨‍💻 Contributors
**Indumathi Balireddi** - Developer 🚀

---

## 📜 License
This project is licensed under the **MIT License** 📄.

---

## ⭐ Hit the Star!
If you like this project, don't forget to **hit the star** ⭐ on GitHub!

