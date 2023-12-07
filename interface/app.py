from transformers import pipeline
import gradio as gr

pipe_fine = pipeline(model="zeihers-mart/whisper-small-swedish-basic", device_map="auto") 
pipe_raw = pipeline(model="openai/whisper-small", device_map="auto")

# force swedish
pipe_fine.model.config.forced_decoder_ids = (
    pipe_fine.tokenizer.get_decoder_prompt_ids(
        language="sv", task="transcribe"
    )
)

pipe_raw.model.config.forced_decoder_ids = (
    pipe_raw.tokenizer.get_decoder_prompt_ids(
        language="sv", task="transcribe"
    )
)

def transcribe(audio):
    text_sv = pipe_fine(audio)["text"]
    print(f"Audio transcribed: {text_sv}")
    text_raw= pipe_raw(audio)["text"]
    print(f"Text translated: {text_raw}")
    return text_sv, text_raw

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=["microphone"], type="filepath"), 
    outputs=[gr.Textbox(label="Fine-tuned transcription"),
             gr.Textbox(label="Whisper Transcription")],
    title="Finetuned Whisper Swedish Small",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()