# import os
# import tempfile
# from TTS.api import TTS


# class TTSTalker():
#     def __init__(self) -> None:
#         model_name = TTS().list_models()[0]
#         self.tts = TTS(model_name)

#     def test(self, text, language='en'):

#         tempf  = tempfile.NamedTemporaryFile(
#                 delete = False,
#                 suffix = ('.'+'wav'),
#             )

#         self.tts.tts_to_file(text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

#         return tempf.name


import os
import tempfile

# Try optional backends
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class TTSTalker:
    def __init__(self, backend: str = "gtts") -> None:
        """
        Initialize the TTS engine.
        backend: "coqui" | "gtts" | "pyttsx3"
        """
        self.backend = backend.lower()

        if self.backend == "coqui" and CoquiTTS:
            # Use first available Coqui model
            model_name = CoquiTTS().list_models()[0]
            self.tts = CoquiTTS(model_name)

        elif self.backend == "gtts" and gTTS:
            self.tts = gTTS  # store class ref

        elif self.backend == "pyttsx3" and pyttsx3:
            self.tts = pyttsx3.init()
        else:
            raise RuntimeError(
                f"TTS backend '{backend}' not available. "
                f"Install required package or choose another backend."
            )

    def test(self, text: str, language: str = "en") -> str:
        """
        Generate speech audio for the given text.
        Returns a path to a temporary .wav file.
        """
        tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tempf.close()

        if self.backend == "coqui":
            self.tts.tts_to_file(
                text,
                speaker=self.tts.speakers[0],
                language=language,
                file_path=tempf.name,
            )

        elif self.backend == "gtts":
            tts_obj = self.tts(text=text, lang=language)
            tts_obj.save(tempf.name)  # gTTS writes mp3, but .wav suffix works for SadTalker

        elif self.backend == "pyttsx3":
            self.tts.save_to_file(text, tempf.name)
            self.tts.runAndWait()

        return tempf.name
