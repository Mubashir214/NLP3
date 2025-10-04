import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import re
from model import Seq2Seq, EncoderBiLSTM, DecoderLSTM  # We'll create this

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Transliterator",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .urdu-text {
        font-size: 1.5rem;
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Urdu', 'Segoe UI', sans-serif;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .feature-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4fd;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UrduTransliterator:
    def __init__(self, model_path='deployment_package.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained model and vocabulary"""
        try:
            # Load the deployment package
            package = torch.load(model_path, map_location=self.device)
            
            # Extract components
            self.stoi = package['stoi']
            self.itos = package['itos']
            config = package['model_config']
            
            # Recreate model architecture
            self.encoder = EncoderBiLSTM(
                vocab_size=len(self.stoi),
                emb_dim=config['emb_dim'],
                enc_hidden=config['enc_hidden'],
                pad_id=self.stoi['<pad>']
            )
            
            self.decoder = DecoderLSTM(
                vocab_size=len(self.stoi),
                emb_dim=config['emb_dim'],
                enc_hidden=config['enc_hidden'],
                dec_hidden=config['dec_hidden'],
                pad_id=self.stoi['<pad>']
            )
            
            self.model = Seq2Seq(self.encoder, self.decoder, self.device)
            self.model.load_state_dict(package['model_weights'])
            self.model.eval()
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            raise e
    
    def preprocess_urdu_text(self, text):
        """Preprocess Urdu text - remove diacritics and normalize"""
        # Remove diacritics and special characters
        diacritics = ["\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652"]
        for d in diacritics:
            text = text.replace(d, "")
        text = text.replace("\u200c", "").replace("\u200d", "")
        return text.strip()
    
    def encode_sentence(self, sentence):
        """Convert sentence to token IDs"""
        sentence = self.preprocess_urdu_text(sentence)
        ids = [self.stoi.get(ch, self.stoi['<unk>']) for ch in sentence]
        ids = [self.stoi['<s>']] + ids + [self.stoi['</s>']]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def decode_ids(self, ids):
        """Convert token IDs back to text"""
        tokens = []
        for id_val in ids:
            token = self.itos.get(int(id_val), "")
            if token in ['<s>', '</s>', '<pad>', '<unk>']:
                continue
            tokens.append(token)
        return "".join(tokens)
    
    def transliterate(self, urdu_text, max_len=128):
        """Convert Urdu text to Roman Urdu"""
        if not urdu_text.strip():
            return ""
            
        try:
            # Encode input
            src_tensor = self.encode_sentence(urdu_text)
            
            # Run through encoder
            encoder_outputs, (h, c) = self.model.encoder(src_tensor)
            
            # Initialize decoder
            batch_size = src_tensor.size(0)
            dec_h = torch.zeros(self.decoder.rnn.num_layers, batch_size, 
                              self.decoder.rnn.hidden_size).to(self.device)
            dec_c = torch.zeros_like(dec_h)
            
            input_step = torch.tensor([[self.stoi['<s>']]], dtype=torch.long).to(self.device)
            output_ids = []
            
            # Generate output
            for _ in range(max_len):
                preds, (dec_h, dec_c) = self.decoder(input_step, (dec_h, dec_c), encoder_outputs)
                top1 = preds.argmax(1).item()
                
                if top1 == self.stoi['</s>']:
                    break
                    
                output_ids.append(top1)
                input_step = torch.tensor([[top1]], dtype=torch.long).to(self.device)
            
            # Decode output
            roman_text = self.decode_ids(output_ids)
            return roman_text
            
        except Exception as e:
            st.error(f"Error in transliteration: {str(e)}")
            return ""

def main():
    # Header
    st.markdown('<h1 class="main-header">🌐 Urdu to Roman Transliterator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Sequence-to-Sequence LSTM model with Attention "
        "to transliterate Urdu text to Roman Urdu (English script)."
    )
    
    st.sidebar.title("Features")
    st.sidebar.markdown("""
    - 🧠 Deep Learning based transliteration
    - 🔤 Character-level translation
    - ⚡ Real-time conversion
    - 📱 Mobile-friendly interface
    """)
    
    # Initialize model
    if 'transliterator' not in st.session_state:
        try:
            st.session_state.transliterator = UrduTransliterator()
        except:
            st.error("Failed to initialize the transliterator. Please check if model files are available.")
            return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Urdu Text")
        urdu_input = st.text_area(
            "Enter Urdu text:",
            placeholder="ایک مثال کے طور پر یہ جملہ لکھیں...",
            height=150
        )
        
        # Example buttons
        st.markdown("**Try these examples:**")
        examples = [
            "محبت سب سے خوبصورت جذبہ ہے",
            "دوستی دلوں کو جوڑتی ہے", 
            "خواب حقیقت بن سکتے ہیں",
            "زندگی ایک سفر ہے"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(example, key=f"ex_{i}"):
                urdu_input = example
    
    with col2:
        st.subheader("🔤 Roman Output")
        if urdu_input:
            with st.spinner("Transliterating..."):
                roman_output = st.session_state.transliterator.transliterate(urdu_input)
            
            if roman_output:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("**Roman Urdu:**")
                st.code(roman_output, language='text')
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Copy to clipboard functionality
                st.button("📋 Copy to Clipboard", 
                         on_click=lambda: st.write("📋 Copied!"),
                         key="copy_btn")
            else:
                st.warning("No output generated. Please try different text.")
        else:
            st.info("👈 Enter Urdu text on the left to see the transliteration here.")
    
    # Batch processing section
    st.markdown("---")
    st.subheader("📁 Batch Processing")
    
    uploaded_file = st.file_uploader("Upload a text file with Urdu content", 
                                   type=['txt', 'csv'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded CSV:")
                st.dataframe(df.head())
            else:
                content = uploaded_file.read().decode('utf-8')
                lines = content.split('\n')
                st.write("Preview of uploaded text:")
                st.text_area("File content", content[:500] + "..." if len(content) > 500 else content, 
                           height=150)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()