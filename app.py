import streamlit as st
import torch
import nltk
import re
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Select device: Use MPS on Apple Silicon if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ----- Chatbot Models & Functions -----
# Load DistilBERT for Intent Classification
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL_NAME, num_labels=3
).to(device)

# Load T5 model for medical Q&A
T5_MODEL_NAME = "google/flan-t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(device)

# Function to preprocess user input (for chatbot)
def preprocess_text(user_input):
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)
    tokens = word_tokenize(user_input)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(filtered_words)

# Healthcare chatbot function
def healthcare_chatbot(user_input):
    processed_input = preprocess_text(user_input)
    input_tokens = distilbert_tokenizer(
        processed_input, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = distilbert_model(**input_tokens)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Mapping for intent classification
    intent_map = {
        0: "Describe symptoms of",
        1: "Provide detailed medical advice on",
        2: "Give medication guidance for"
    }
    
    t5_prompt = f"{intent_map.get(predicted_class, 'Give general health advice in English')}: {user_input}"
    encoded_prompt = t5_tokenizer(
        t5_prompt, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    output_tokens = t5_model.generate(
        **encoded_prompt,
        max_length=250,      # Increased response length for more details
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2
    )
    
    response_text = t5_tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    return response_text if response_text else "I'm sorry, but I couldn't generate a response."

# ----- Health Calculators Functions -----
# BMI Calculator function
def bmi_calculator(weight, height_cm):
    height_m = height_cm / 100  # Convert height to meters
    bmi = weight / (height_m ** 2)
    return bmi

# Blood Pressure Evaluator function
def bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic < 130 and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    else:
        return "Consult your doctor"

# Blood Sugar Evaluator function
def blood_sugar_evaluator(fasting):
    if fasting < 100:
        return "Normal"
    elif 100 <= fasting < 126:
        return "Prediabetes"
    elif fasting >= 126:
        return "Diabetes"
    else:
        return "Consult your doctor"

# ----- Main Streamlit App -----
def main():
    st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🏥")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Functionality",
        ["Chatbot", "BMI Calculator", "Blood Pressure Evaluator", "Blood Sugar Evaluator"]
    )
    
    # Sidebar: Info & Guidelines
    st.sidebar.markdown("---")
    st.sidebar.header("Health Guidelines")
    st.sidebar.markdown("**BMI Guidelines:**")
    st.sidebar.markdown("- Underweight: < 18.5")
    st.sidebar.markdown("- Normal: 18.5 – 24.9")
    st.sidebar.markdown("- Overweight: 25 – 29.9")
    st.sidebar.markdown("- Obese: ≥ 30")
    st.sidebar.markdown("**Blood Pressure (mmHg):**")
    st.sidebar.markdown("- Normal: < 120/80")
    st.sidebar.markdown("- Elevated: 120-129/<80")
    st.sidebar.markdown("- Hypertension Stage 1: 130-139/80-89")
    st.sidebar.markdown("- Hypertension Stage 2: ≥ 140/≥90")
    st.sidebar.markdown("**Fasting Blood Sugar (mg/dL):**")
    st.sidebar.markdown("- Normal: < 100")
    st.sidebar.markdown("- Prediabetes: 100-125")
    st.sidebar.markdown("- Diabetes: ≥ 126")
    
    # Main page disclaimer
    st.markdown("**Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice.")
    
    if app_mode == "Chatbot":
        st.title("🤖 AI Healthcare Assistant Chatbot")
        user_input = st.text_input("👤 How can I assist you today?")
        if st.button("Submit"):
            if user_input:
                with st.spinner("🤖 Generating response..."):
                    response = healthcare_chatbot(user_input)
                st.success(f"**Healthcare Assistant:** {response}")
            else:
                st.warning("⚠️ Please enter a message to get a response.")
    
    elif app_mode == "BMI Calculator":
        st.title("BMI Calculator")
        weight = st.number_input("Enter your weight (kg)", min_value=1.0, value=70.0)
        height = st.number_input("Enter your height (cm)", min_value=50.0, value=170.0)
        if st.button("Calculate BMI"):
            bmi = bmi_calculator(weight, height)
            st.write(f"Your BMI is: **{bmi:.2f}**")
            if bmi < 18.5:
                st.info("You are underweight.")
            elif bmi < 25:
                st.info("You have normal weight.")
            elif bmi < 30:
                st.info("You are overweight.")
            else:
                st.info("You are obese.")
    
    elif app_mode == "Blood Pressure Evaluator":
        st.title("Blood Pressure Evaluator")
        systolic = st.number_input("Enter your systolic pressure (mmHg)", min_value=50, value=120)
        diastolic = st.number_input("Enter your diastolic pressure (mmHg)", min_value=30, value=80)
        if st.button("Evaluate Blood Pressure"):
            category = bp_category(systolic, diastolic)
            st.write(f"Your blood pressure category is: **{category}**")
            st.info("Ideal blood pressure is less than 120/80 mmHg.")
    
    elif app_mode == "Blood Sugar Evaluator":
        st.title("Blood Sugar Evaluator")
        fasting = st.number_input("Enter your fasting blood sugar level (mg/dL)", min_value=50, value=90)
        if st.button("Evaluate Blood Sugar"):
            category = blood_sugar_evaluator(fasting)
            st.write(f"Your fasting blood sugar level is: **{fasting} mg/dL**")
            st.write(f"Category: **{category}**")
            st.info("Normal fasting blood sugar is less than 100 mg/dL.")

if __name__ == "__main__":
    main()
