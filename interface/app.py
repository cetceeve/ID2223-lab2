from transformers import pipeline
import gradio as gr
import time

pipe_fine = pipeline(model="zeihers-mart/whisper-small-swedish-basic", device_map="auto") 
pipe_raw = pipeline(model="openai/whisper-small", device_map="auto")
sa = pipeline('sentiment-analysis', model='marma/bert-base-swedish-cased-sentiment')

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
    start = time.time()
    text_sv = pipe_fine(audio)["text"]
    time_fine = time.time() - start
    print(f"Fine-tuned: audio transcribed in {time_fine} seconds: {text_sv}")

    start = time.time()
    text_raw= pipe_raw(audio)["text"]
    time_raw = time.time() - start
    print(f"Raw: audio transcribed in {time_raw} seconds: {text_raw}")
    
    sentiment= sa(text_sv)
    print(f"Sentiment result: {sentiment}")
    sentiment= sentiment[0]["label"]
    happy_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/1200px-SNice.svg.png"
    sad_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Face-sad.svg/480px-Face-sad.svg.png"
    path = happy_path if sentiment == "POSITIVE" else sad_path
    
    description = f"The fine-tuned model took {time_fine} seconds while the original Whisper model took {time_raw} seconds.\nThe sentiment was evaluated from the fine-tuned model transcription as {sentiment.lower()}."
    return text_sv, text_raw, path, description

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=["microphone"], type="filepath"), 
    outputs=[gr.Textbox(label="Fine-tuned transcription"),
             gr.Textbox(label="Whisper transcription"),
             gr.Image(label="Sentiment from Fine-tuned transcription", width=250, height=250),
             gr.Textbox(label="Description")],
    title="Finetuned Whisper Swedish Small",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()