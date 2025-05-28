import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify
import gradio as gr
from model import ToxicCommentDetector

app = Flask(__name__)
detector = ToxicCommentDetector()
detector.load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model_name', 'DistilBERT')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        results = detector.predict(text, model_name)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_gradio_interface(detector):
    def predict_toxicity(text, model_name):
        if not text.strip():
            return "Please enter some text to analyze."
        try:
            results = detector.predict(text, model_name)
            output = f"🔍 **Analysis Results using {model_name}:**\n\n"
            for label, score in results.items():
                emoji = "🚨" if score > 0.5 else "✅"
                output += f"{emoji} **{label.replace('_', ' ').title()}**: {score:.3f} ({score*100:.1f}%)\n"
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks(title="🛡️ Toxic Comment Detector", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🛡️ Toxic Comment Detector
        This app uses three different pre-trained models to detect toxicity in comments.
        Enter your text below and choose a model to get predictions, or compare all models at once!
        """)

        with gr.Tab("Single Model Prediction"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter comment to analyze", placeholder="Type your comment here...", lines=3)
                    model_dropdown = gr.Dropdown(choices=list(detector.models.keys()), label="Select Model", value=list(detector.models.keys())[0])
                    predict_btn = gr.Button("🔍 Analyze Toxicity", variant="primary")

                with gr.Column():
                    single_output = gr.Markdown(label="Results")

            predict_btn.click(predict_toxicity, inputs=[text_input, model_dropdown], outputs=single_output)

    return interface

if __name__ == "__main__":
    interface = create_gradio_interface(detector)
    interface.launch()