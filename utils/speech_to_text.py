import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for symptoms...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print("Error:", e)
        return ""
