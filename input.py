import speech_recognition as sr
from google import genai
from google.genai import types
import base64
from google.cloud import texttospeech


"""Listens for user input from microphone and returns the recognized text."""
r = sr.Recognizer()
with sr.Microphone() as source:
    
  print("Listening...")
  audio = r.listen(source)
  try:
    text = r.recognize_google(audio)
    print(text)
  except sr.UnknownValueError:
    print("Could not understand audio")
    
  except sr.RequestError as e:
    print(f"Could not request results; {e}")
    

#*****************************

#Giving input text to gemini model 



def generate():
  client = genai.Client(
      vertexai=True,
      project="testingboys",
      location="us-central1"
  )


  model = "gemini-2.0-flash-exp"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text)
      ]
    )
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk, end="")

generate()

#**********************************************

"""Synthesizes speech from the input string of text."""


client = texttospeech.TextToSpeechClient()

input_text = texttospeech.SynthesisInput(text="hello")

# Note: the voice can also be specified by name.
# Names of voices can be retrieved with client.list_voices().
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Studio-O",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1
)

response = client.synthesize_speech(
    request={"input": input_text, "voice": voice, "audio_config": audio_config}
)

# The response's audio_content is binary.
with open("output.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')