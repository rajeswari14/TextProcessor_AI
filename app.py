import gradio as gr
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords


# Use a more advanced summarization model
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# Use a larger T5 model for rewriting
re_writer = pipeline("text2text-generation", model="t5-large")

# Improved summarization with parameter tuning

def summarize_text(text):
    summary = summarizer(
        text,
        max_length=120,  # shorter, more focused summary
        min_length=30,
        do_sample=False,
        clean_up_tokenization_spaces=True
    )
    # Post-process: remove extra spaces, fix capitalization
    result = summary[0]['summary_text'].strip().capitalize()
    return result

# Improved rewriting with parameter tuning

def rewrite_professional(text):
    prompt = "Rewrite this text in a professional tone: " + text
    response = re_writer(
        prompt,
        max_length=150,
        min_length=50,
        do_sample=False,
        clean_up_tokenization_spaces=True
    )
    # Post-process: remove extra spaces, fix capitalization
    result = response[0]['generated_text'].strip().capitalize()
    return result

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Gradio interface
import gradio as gr

# Main function to handle user input
def process_text(text, task):
    if task == "Summarize":
        result = summarize_text(text)
    elif task == "Rewrite Professional":
        result = rewrite_professional(text)
    else:
        result = "Invalid Task"
    return result

# Define Gradio UI with only text output

demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            lines=10, 
            placeholder="Paste your text here...",
            label="Enter Text"
        ),
        gr.Radio(
            choices=["Summarize", "Rewrite Professional"],
            label="Choose Task",
            value="Summarize"  # default task
        )
    ],
    outputs=gr.Textbox(label="AI Output Text"),
    title="✨ AI Text Helper App ✨",
    description="""
        Enter your text — choose to summarize or rewrite professionally.
    """
)


# Launch app
if __name__ == "__main__":
    demo.launch()
