import speech_recognition as sr
import os
from datetime import datetime
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

from text_response import respond_to_text

def main():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    model_path = "C:/Users/sammy/Desktop/Speech/"
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone calibrated. Start speaking...")

        while True:
            try:
                # Listen for speech
                print("Listening...")
                audio_data = recognizer.listen(source, timeout=None, phrase_time_limit=10)
                print("Processing speech...")

                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio_data)
                print(f"Transcription: {text}")

                response = respond_to_text(text, tokenizer, model)
                print(f"Model response: {response}")

            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
            except sr.RequestError as e:
                print(f"Error with the speech recognition service: {e}")
            except KeyboardInterrupt:
                print("Stopping transcription...")
                break

if __name__ == "__main__":
    main()
