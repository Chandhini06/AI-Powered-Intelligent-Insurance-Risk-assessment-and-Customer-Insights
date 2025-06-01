
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
from streamlit_lottie import st_lottie
import requests
import numpy as np
import joblib
import torch
import torch.nn as nn
import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM


# Set page config
st.set_page_config(page_title="Insurance risk assessment and customer insights", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



st.sidebar.title("Navigate")
choice = st.sidebar.radio(label="Select a section",options=["Home",
            "Risk Score Classification",
            "Claim Amount Prediction",
            "Customer Segmentation",
            "Fraud Detection",
            "Sentiment Analysis",
            "Translation",
            "Summarization",
            "Chatbot"])

if choice == "Home":
    
    st.markdown("## Welcome to the AI-Powered Insurance System!")
    st.markdown("This application provides intelligent insights and automation for the insurance industry using advanced ML, NLP, and deep learning techniques.")

    lottie_ai = load_lottieurl("https://gist.githubusercontent.com/Chandhini06/69f9ffc750dcd940d43b84875003d645/raw/14222354517786b1961900df9f40156da8049743/insurance.json")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st_lottie(lottie_ai, height=300, key="ai-brain")


elif choice == "Risk Score Classification":
    

    st.markdown("## Risk Score Classification")
    # Load pre-trained RandomForest model
    model_path = "C:/Users/Admin/OneDrive/Documents/Final Project 2/models/Randomforest_risk_classification.pkl"
    Scaler_path = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\risk_classification_scaler.pkl"


    with open(model_path, "rb") as f:
        loaded_rf_model = pickle.load(f)

    with open(Scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)


    # Define feature columns (Ensure same order as training)
    feature_names = loaded_rf_model.feature_names_in_
    

    # Manual Prediction Section
    st.write("### 🔍 Enter Policy Details")

    annual_income = st.number_input("Annual Income", min_value=50000.0, format="%.2f", step = 500.0)
    claim_amount = st.number_input("Claim Amount", min_value=0.0, format="%.2f", step = 500.0)
    premium_amount = st.number_input("Premium Amount", min_value=0.0, format="%.2f", step = 500.0)
    claim_history = st.number_input("Claim History (Number of claims)", min_value=0, step=1)

    policy_type = st.selectbox("Policy Type", ["Health", "Auto", "Life", "Property"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Predict Risk Score"):
        # Create input dictionary with one-hot encoding
        input_dict = {
            'Annual_Income': [annual_income],
            'Claim_Amount': [claim_amount],
            'Premium_Amount': [premium_amount],
            'Claim_History': [claim_history],
            'Policy_Type_Auto': [1 if policy_type == "Auto" else 0],
            'Policy_Type_Health': [1 if policy_type == "Health" else 0],
            'Policy_Type_Life': [1 if policy_type == "Life" else 0],
            'Policy_Type_Property': [1 if policy_type == "Property" else 0],
            'Gender_Female': [1 if gender == "Female" else 0],
            'Gender_Male': [1 if gender == "Male" else 0],
            'Gender_Other': [1 if gender == "Other" else 0]
        }
        input_df = pd.DataFrame(input_dict)

        # Ensure input DataFrame matches training features *in the correct order*
        input_df = input_df.reindex(columns=feature_names, fill_value=0)


        # Apply MinMaxScaler (same as used during training)
        scaled_features = loaded_scaler.transform(input_df[["Annual_Income", "Claim_Amount", "Premium_Amount"]])
        input_df[["Annual_Income", "Claim_Amount", "Premium_Amount"]] = scaled_features

        # Make prediction
        prediction = loaded_rf_model.predict(input_df)
        
        # Risk Score Mapping
        risk_mapping = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        predicted_risk = risk_mapping.get(int(prediction[0]), "Unknown")

        st.success(f"### 🛡️ Predicted Risk Category: **{predicted_risk}**")

elif choice == "Claim Amount Prediction" :

    st.markdown("## Claim Amount Prediction")

    model_path = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\random_forest_claim_prediction.pkl"
    scaler_path = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\claims_prediction_scaler.pkl"


    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    
    # Load the scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Scaler file not found at: {scaler_path}")
        st.stop()


    st.markdown("### 🧾 Fill in the customer details below:")

    # Input fields with icons, no min/max restrictions
    age = st.number_input("Customer Age", value=10, step=1)
    income = st.number_input("Annual Income", value=50000, step=500)
    premium = st.number_input("Premium Amount", value=500, step=500)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    policy = st.selectbox("Policy Type", ["Health", "Life", "Auto", "Property"])
    claim_history = st.number_input("Claim History (Number of claims)", min_value=0, step=1)
    risk_score = st.selectbox("Risk Score", ["Low", "Medium", "High"])


    if st.button("Predict Claim Amount"):
        if None in [age, income, premium] or gender == "Select Gender" or policy == "Select Policy Type":
            st.warning("Please fill out all fields before predicting.")
        else:
            # Manual encoding
            gender_map = {
                "Gender_Female": int(gender == "Female"),
                "Gender_Male": int(gender == "Male"),
                "Gender_Other": int(gender == "Other")
            }

            policy_map = {
                "Policy_Type_Auto": int(policy == "Auto"),
                "Policy_Type_Health": int(policy == "Health"),
                "Policy_Type_Property": int(policy == "Property"),
                "Policy_Type_Life": int(policy == "Life")
            }

            risk_map = {
                "Risk_Score_Low": int(risk_score == "Low"),
                "Risk_Score_Medium": int(risk_score == "Medium"),
                "Risk_Score_High": int(risk_score == "High")
            }



            # Construct input DataFrame
            user_df = pd.DataFrame([{
                "Customer_Age": age,
                "Annual_Income": income,
                "Claim_History": claim_history,
                "Premium_Amount": premium,
                **gender_map,
                **policy_map,
                **risk_map
            }])


            scale_cols = ["Annual_Income", "Premium_Amount", "Customer_Age"]
            user_df[scale_cols] = scaler.transform(user_df[scale_cols])


            # Reorder columns to match model training
            final_cols = [
                'Customer_Age', 'Annual_Income', 'Claim_History', 'Premium_Amount','Policy_Type_Auto', 'Policy_Type_Health', 'Policy_Type_Life','Policy_Type_Property',
                'Gender_Female', 'Gender_Male', 'Gender_Other',
                "Risk_Score_High", "Risk_Score_Low", "Risk_Score_Medium"
            ]
            user_df = user_df[final_cols]

            # Make prediction
            prediction = model.predict(user_df)[0]
            st.success(f"Predicted Claim Amount: ₹{prediction:,.2f}") 


elif choice == "Customer Segmentation" :

    st.markdown("## Customer Segmentation")

    model_path = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\kmeans_segment.pkl"


    with open(model_path, "rb") as f:
        loaded_kmeans_model = pickle.load(f)

     # Load the trained PCA model
    pca = joblib.load(r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\pca_segment.pkl")  # Load PCA model

    # Load the trained scaler
    scaler = joblib.load(r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\scaler_segment.pkl")  # Load StandardScaler



    # Define function to assign segment labels
    cluster_labels = {
        0: "High-Value, High-Claim Customers",
        1: "Young and Growing Customers",
        2: "Senior Customers with High Premiums",
        3: "Low Engagement, Low-Risk Customers"
    }

    def assign_segment(label):
        return cluster_labels.get(label, "Unknown Segment")


    st.markdown("### 🧾 Provide customer information for segmentation:")

    # User Inputs with icons and no input restrictions
    age = st.number_input("Age", value=18, step=1)
    annual_income = st.number_input("Annual Income", value=5000.00, step = 500.00, format="%.2f")
    policy_count = st.number_input("Number of Active Policies", value=0, step=1)
    total_premium_paid = st.number_input("Total Premium Paid ($)", value=100.00, step = 100.00,  format="%.2f")
    claim_frequency = st.number_input("Number of Claims Filed", value=0, step=1)
    policy_upgrades = st.number_input("Number of Policy Changes", value=0, step=1)


    # Predict Segment
    if st.button("Predict Segment"):
        user_data = np.array([[age, annual_income, policy_count, total_premium_paid, claim_frequency, policy_upgrades]])
        user_data_scaled = scaler.transform(user_data)
        user_data_pca = pca.transform(user_data_scaled)
        predicted_cluster = loaded_kmeans_model.predict(user_data_pca)[0]
        segment = assign_segment(predicted_cluster)
        st.success(f"Predicted Segment: {segment}") 


elif choice == "Fraud Detection":
 
    st.markdown("## Fraudulent Claims Detection")
     # Load models and scaler using absolute paths
    with open("C:/Users/Admin/OneDrive/Documents/Final Project 2/models/best_lgb_model_fraud_detection.pkl", "rb") as f:
        lgb_model = pickle.load(f)

    with open("C:/Users/Admin/OneDrive/Documents/Final Project 2/models/nn_model_fraud_detection.pkl", "rb") as f:
        nn_model = pickle.load(f)

    # Load Scaler
    with open("C:/Users/Admin/OneDrive/Documents/Final Project 2/models/scaler_fraud_detection.pkl", "rb") as f:
        scaler = pickle.load(f)


    st.markdown("### 🧾 Enter claim details to predict fraud probability:")

    # Input fields (initially empty, with icons)
    claim_amount = st.number_input("Claim Amount", value=100.00, step = 100.00)
    annual_income = st.number_input("Annual Income", value=20000.00, step = 500.00)

    # Compute claim-to-income ratio safely
    if claim_amount and annual_income:
        claim_to_income_ratio = claim_amount / annual_income
    else:
        claim_to_income_ratio = 0
    
    st.markdown(f"### Claim-to-Income Ratio: {claim_to_income_ratio:.4f}")  # Display ratio for debugging

    claim_within_short = st.number_input('Policy_Claim_Diff_Days(Number of days after the issuance)', value = 5, step = 10)

    suspicious_flag = st.selectbox("Suspicious Flag?", ["Select", "Yes", "No"])
    suspicious_flag_val = 1 if suspicious_flag == "Yes" else 0 
   
    # Claim type input field
    claim_type = st.selectbox("Claim Type", ["Select", 'Auto', 'Home', 'Life', 'Medical'])
    suspicious_flag = 1 if suspicious_flag == "Yes" else 0 
    claim_type_auto = 1 if claim_type == 'Auto' else 0
    claim_type_home = 1 if claim_type == 'Home' else 0
    claim_type_life = 1 if claim_type == 'Life' else 0
    claim_type_medical = 1 if claim_type == 'Medical' else 0

    # Simplified input for anomaly detection
    is_anomalous = st.selectbox("Is the claim detected as anomalous?", ["Select","Yes", "No"])
    is_anomalous_val = 1 if is_anomalous == "Yes" else 0 

    if st.button("Predict Fraud Probability"):
        # Prepare input array in correct feature order
        
        feature_names = lgb_model.booster_.feature_name()

        input_dict = {
            'Claim_Amount': claim_amount,
            'Annual_Income': annual_income,
            'Suspicious_Flags': suspicious_flag_val,
            'Claim_to_income_ratio': claim_to_income_ratio,
            'Policy_Claim_Diff_Days': claim_within_short,
            'Claim_Type_Auto': claim_type_auto,
            'Claim_Type_Home': claim_type_home,
            'Claim_Type_Life': claim_type_life,
            'Claim_Type_Medical': claim_type_medical,
            'Anomaly_Flag': is_anomalous_val
        }

        for col in feature_names:
            if col not in input_dict:
                input_dict[col] = 0
        
        input_df = pd.DataFrame([input_dict])
        
        input_df = input_df.apply(pd.to_numeric)

        # Scale only claim_amount and annual_income (columns 0 and 2)
        input_df[['Claim_Amount', 'Annual_Income']] = scaler.transform(
            input_df[['Claim_Amount', 'Annual_Income']]
        )

        # Predict probabilities
        lgb_prob = lgb_model.predict_proba(input_df)[0][1]
        nn_prob = nn_model.predict_proba(input_df)[0][1]
        fraud_score = (lgb_prob + nn_prob) / 2
    
        # Business rule checks
        manual_flags = []

        if claim_to_income_ratio > 2:
            manual_flags.append("High claim-to-income ratio")

        elif claim_within_short < 100:
            manual_flags.append("Claim made very soon after policy issuance")

        elif suspicious_flag == 1:
            manual_flags.append("Suspicious claim flag detected")

        elif is_anomalous == 1:
            manual_flags.append("Detected as anomalous claim")

        # Final fraud decision (model + rules)
        if lgb_prob > 0.1 or len(manual_flags) >= 1:
            st.error("🔍 Prediction: ⚠️ **Fraudulent Claim Detected**")
        else:
            st.success("🔍 Prediction: ✅ Genuine Claim")

        # Display the fraud score and reasons
        # st.markdown(f"🔢 Fraud Score: **{lgb_prob:.4f}**")

        # if manual_flags:
        #     st.warning("🚨 Manual Red Flags Triggered:\n" + "\n".join([f"- {flag}" for flag in manual_flags]))


elif choice == "Sentiment Analysis" :


    # from sklearn.preprocessing import LabelEncoder

    # st.markdown('## Sentiment Analysis')
    # class SentimentModel(nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super(SentimentModel, self).__init__()
    #         self.fc1 = nn.Linear(input_dim, 128)
    #         self.fc2 = nn.Linear(128, 64)
    #         self.fc3 = nn.Linear(64, output_dim)
    #         self.relu = nn.ReLU()

    #     def forward(self, x):
    #         x = self.relu(self.fc1(x))
    #         x = self.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x 
        
    # @st.cache_resource
    # def load_model():
    #     try:
    #         input_dim = 768  # BERT embedding size
    #         output_dim = 3   # Sentiment classes: Negative, Neutral, Positive
    #         model = SentimentModel(input_dim, output_dim)
    #         model.load_state_dict(torch.load("C:/Users/Admin/OneDrive/Documents/Final Project 2/models/sentiment_model.pth", map_location=torch.device("cpu")))
    #         model.eval()
    #         return model
    #     except Exception as e:
    #         st.error(f"Error loading model: {e}")
    #         return None

    # @st.cache_resource
    # def load_label_encoder():
    #     with open("C:/Users/Admin/OneDrive/Documents/Final Project 2/models/label_encoder.pkl", "rb") as f:
    #         return pickle.load(f)
    
    # label_encoder = load_label_encoder()

    # @st.cache_resource
    # def load_bert():
    #     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     bert = BertModel.from_pretrained("bert-base-uncased")
    #     return tokenizer, bert

    # tokenizer, bert_model = load_bert()

    # def get_bert_embedding(text):
    #     tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         output = bert_model(**tokens)
    #     return output.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # model = load_model()
    
    # input_text = st.text_area("✏️ Enter your text for sentiment prediction:")

   

    # if st.button("🔍 Predict Sentiment"):

    #     if not input_text.strip():
    #         st.warning("Please enter some text for prediction.")
    #     else:
    #         # Get embedding
    #         embedding = get_bert_embedding(input_text)
    #         input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

    #         # Predict
    #         with torch.no_grad():
    #             output = model(input_tensor)
    #             probs = torch.softmax(output, dim=1)
    #             predicted_class = torch.argmax(probs, dim=1).item()

    #         predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    #         threshold = 0.6  # Example threshold for a sentiment classification
    #         if probs[0][predicted_class] > threshold:
    #             predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    #         else:
    #             predicted_label = "Neutral"  

    #         st.success(f"### Predicted Sentiment: **{predicted_label}**")
    #         st.write("Softmax Probabilities:", dict(zip(label_encoder.classes_, probs.numpy().flatten())))
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sentiment_analyzer = SentimentIntensityAnalyzer()
    st.title("Sentiment Analysis of Customer Feedback")
    feedback_text = st.text_area("Enter Customer Feedback:")
    if st.button("Analyze Sentiment"):
        if feedback_text:
            sentiment_score = sentiment_analyzer.polarity_scores(feedback_text)
            compound_score = sentiment_score['compound']
            sentiment_label = "Positive" if compound_score >= 0.4 else "Negative" if compound_score <= -0.05 else "Neutral"
            # st.write(f"Sentiment Score: {compound_score}")
            st.success(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.error("Please enter feedback text to analyze.")


elif choice == "Translation":

    from transformers import MarianMTModel, MarianTokenizer
    
    st.markdown('## Translator')

    device = "cpu"

    MODEL_PATH_fr = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\translator_finetuned_mbart_fr"
    MODEL_PATH_es = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\translator_finetuned_mbart_es"

    # ✅ Cache French model loading
    @st.cache_resource
    def load_model_fr():
        try:
            tokenizer_fr = MarianTokenizer.from_pretrained(MODEL_PATH_fr)
            model_fr = MarianMTModel.from_pretrained(MODEL_PATH_fr)
            model_fr = model_fr.to(device)
            return tokenizer_fr, model_fr
        except Exception as e:
            st.error(f"❌ Failed to load French model: {str(e)}")
            return None, None

    # ✅ Cache Spanish model loading
    @st.cache_resource
    def load_model_es():
        try:
            tokenizer_es = AutoTokenizer.from_pretrained(MODEL_PATH_es)
            model_es = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_es)
            model_es = model_es.to(device)
            return tokenizer_es, model_es
        except Exception as e:
            st.error(f"❌ Failed to load Spanish model: {str(e)}")
            return None, None

    # ✅ Load both models
    tokenizer_fr, model_fr = load_model_fr()
    tokenizer_es, model_es = load_model_es()

    # ✅ Translation function
    def translate_fr(text, model_fr, tokenizer_fr):
        try:
            inputs = tokenizer_fr(text, return_tensors="pt", truncation=True, padding=True).to(device)
            generated = model_fr.generate(
                **inputs,
                max_length=128,
                num_beams=4
            )
            return tokenizer_fr.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            return f"⚠️ Error: {e}"

    def translate_es(text, model_es, tokenizer_es):
        try:
            inputs = tokenizer_es(text, return_tensors="pt", truncation=True, padding=True).to(device)
            generated = model_es.generate(
                **inputs,
                max_length=128,
                num_beams=4
            )
            return tokenizer_es.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            return f"⚠️ Error: {e}"


    # ✅ Streamlit UI
    input_text = st.text_area("✍️ Enter English text to translate:", height=150)

    if st.button("Translate"):
        if not input_text.strip():
            st.warning("Please enter valid English text.")
        else:
            with st.spinner("🔄 Translating..."):
                fr_text = translate_fr(input_text, model_fr, tokenizer_fr) if model_fr else "Unavailable"
                es_text = translate_es(input_text, model_es, tokenizer_es) if model_es else "Unavailable"

            st.subheader("✅ Translations:")
            st.markdown(f"**🇫🇷 French:** {fr_text}")
            st.markdown(f"**🇪🇸 Spanish:** {es_text}")


elif choice == "Summarization" :

    # st.markdown('## Summarization')
    # @st.cache_resource()
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer


    device = "cpu"

    # def load_model():

    MODEL_PATH = r"C:\Users\Admin\OneDrive\Documents\Final Project 2\models\summarized_fine_tuned_model"
    @st.cache_resource
    def load_model():
        tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)  # auto-reads safetensors
        return tokenizer, model.to("cpu")

    tokenizer, model = load_model()


    # Input Text Area
    # input_text = st.text_area("Enter text to summarize:", height=200)

    # model = model.to(device)

    # Summarization function
    def generate_summary(text):
        if not text.strip():
            return ""
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}  

        # with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=300,
            min_length=60,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Streamlit UI
    st.title("📄 Insurance Policy Summarizer")
    # st.markdown("Summarize long insurance policy documents using your fine-tuned mT5 or mBART model.")

    # Text input
    input_text = st.text_area("Enter Policy Text", height=200)

    if st.button("Generate Summary"):
        if not input_text.strip():
            st.warning("Please enter some policy text.")
        else:
            with st.spinner("Generating summary..."):
                summary = generate_summary(input_text)
            st.success("Summary:")
            st.write(summary)



elif choice  == "Chatbot":
    
    import streamlit as st
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    import torch

    # 🟠 Must be the first Streamlit command
    #st.set_page_config(page_title="Insurance Chatbot 🤖", layout="centered")

    # 🎯 Load model and tokenizer
    MODEL_PATH = "C:/Users/Admin/OneDrive/Documents/Final Project 2/models/insurance_chatbot_t5_model/content/insurance_t5_model"

    device = "cpu"

    @st.cache_resource
    def load_model():
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
        model = model.to(device)
        return tokenizer, model

    tokenizer, model = load_model()

    # 💬 App UI
    st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .stTextInput>div>div>input {
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #ccc;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align: center; font-family: "Impact", "Arial Black", sans-serif;
               color: #0D47A1; font-size: 52px; letter-spacing: 1px;'>
        🤖 <span style='color:#1976D2;'>Insurance Query</span> <span style='color:#0D47A1;'>Chatbot</span>
    </h1>
    """, unsafe_allow_html=True)

    def clean_bot_response(response):
        return response.replace('-LRB-', '').replace('-RRB-', '').replace('?','.')
    
    st.write("Ask me anything about your insurance policies, claims, or coverage!")

    # 💾 Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 🧠 Chat form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Question", placeholder="e.g. How do I file a car insurance claim?")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            input_text = "insurance query: " + user_input.strip()
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

            # ✅ Improved generation settings to reduce repetition
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
                early_stopping=True
            )

            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # ✅ Store messages
            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Bot", decoded_output))

    # 💬 Display conversation history
    # if st.session_state.messages:
        # st.markdown("### 💬 Chat History")
       # Show only the latest response
    if submitted and user_input.strip():
        # st.markdown("### 🤖 Bot Response:")
        # st.success(clean_bot_response(decoded_output))

        # st.markdown("---")
        # st.info("Ask another question or close the app anytime.")
        st.markdown("### 💬 Chat History")
        st.markdown(f"**🧑 You:** {user_input}")
        st.markdown(f"**🤖 Bot:** {clean_bot_response(decoded_output)}")

        st.markdown("---")
        st.info("Ask another question or close the app anytime.")