from __future__ import division
from __future__ import print_function

import re
import sys

from google.cloud import speech_v1p1beta1 as speech
import pyaudio
from six.moves import queue
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty,\
    ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.properties import StringProperty
import threading
import math

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 100)  # 10ms
NUMBER_OF_PEOPLE=2

lastState = [0,0]
timesArray = [0,0]
previousTimesArray = [0,0]
wordsArray = ["", ""]


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)
		
class Relojes(Widget):
	tiempo = NumericProperty(0)
	cronometro = StringProperty("00:00")
	
	def sumar(self):
		self.tiempo += 1
		hzero = ""
		mzero = ""
		tiempoAux = self.tiempo / 60
		minuto = tiempoAux % 60
		hora = tiempoAux / 60
		if minuto < 10:
			mzero = "0"
		if hora < 10:
			hzero = "0"
			
		self.cronometro = hzero + str(int(hora)) + ":" + mzero + str(int(minuto))
		
	def sumarSegundos(self, tiempoSeg):
		self.tiempo += tiempoSeg * 60
		hzero = ""
		mzero = ""
		tiempoAux = self.tiempo / 60
		minuto = tiempoAux % 60
		hora = tiempoAux / 60
		if minuto < 10:
			mzero = "0"
		if hora < 10:
			hzero = "0"
			
		self.cronometro = hzero + str(hora) + ":" + mzero + str(minuto)
	
	def setReloj(self, segundosTotales):
		szero = ""
		mzero = ""
		minuto = math.floor(segundosTotales/60)
		segundo = math.floor(segundosTotales - (minuto*60))
		if minuto < 10:
			mzero = "0"
		if segundo < 10:
			szero = "0"
		self.cronometro = mzero + str(int(minuto)) + ":" + szero + str(int(segundo))
		

class Zeus(Widget):

	stop = threading.Event()
	
	reloj1 = ObjectProperty(None)
	reloj2 = ObjectProperty(None)
	reloj3 = ObjectProperty(None)
	reloj4 = ObjectProperty(None)
	
	relojTema1 = ObjectProperty(None)
	relojTema2 = ObjectProperty(None)
	relojTema3 = ObjectProperty(None)
	
	lastIntTime = NumericProperty(0)
	lastIntIndex = NumericProperty(-1)
	
	interlocutor = 1 #Persona 1,2,3,4
	temaParlat = NumericProperty(1) #Tema 1,2,3
	
	tiempoDebugging = 0

	def inicializar(self):
		print ("Inicio")
		threading.Thread(target=self.main).start()

	def update(self, dt):

	####################################################################
	#Consultar Google Cloud
	
		interlocutor = 1 # = obtener su numero de Google Cloud
		fraseGoogle = "" # = obtener la frase de Google Cloud
			
		for i in range(0,NUMBER_OF_PEOPLE):
			for j in range(0, len(wordsArray[i])):
				if (wordsArray[i][j] == "Zeus" or wordsArray[i][j] == "zeus" or wordsArray[i][j] == "Neus" or wordsArray[i][j] == "test" or wordsArray[i][j] == "sistema"):
					if (len(wordsArray[i]) > j+1 and (wordsArray[i][j+1] == "tema" or wordsArray[i][j+1] == "Tema" or wordsArray[i][j] == "sistema")):
						ex = False
						if (wordsArray[i][j] == "sistema"):
							ex = True
						if (len(wordsArray[i]) > j+2 or ex):
							try:
								if (ex):
									n = int(wordsArray[i][j+1])
								else:
									n = int(wordsArray[i][j+2])
								if (n > self.temaParlat):
									self.temaParlat = n
							except ValueError:
								pass

	####################################################################
	
		self.reloj1.setReloj(timesArray[0])
		self.reloj2.setReloj(timesArray[1])
		#self.reloj3.setReloj(timesArray[2])
		#self.reloj4.setReloj(timesArray[3])		
		
		if self.temaParlat == 1:
			self.relojTema1.sumar()
		if self.temaParlat == 2:
			self.relojTema2.sumar()
		if self.temaParlat == 3:
			self.relojTema3.sumar()
			
	def main(self):
		# See http://g.co/cloud/speech/docs/languages
		# for a list of supported languages.
		language_code = 'es-ES'  # a BCP-47 language tag
		global previousTimesArray
		global CHUNK
		global RATE
		global NUMBER_OF_PEOPLE
		
		client = speech.SpeechClient()

		metadata = speech.types.RecognitionMetadata()
		metadata.interaction_type = (
			speech.enums.RecognitionMetadata.InteractionType.DISCUSSION)
		metadata.recording_device_type = (
			speech.enums.RecognitionMetadata.RecordingDeviceType.PC)

		config = speech.types.RecognitionConfig(
			encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
			sample_rate_hertz=RATE,
			language_code=language_code,
			enable_speaker_diarization=True,
			diarization_speaker_count=NUMBER_OF_PEOPLE,
			metadata=metadata)
		streaming_config = speech.types.StreamingRecognitionConfig(
			config=config)

		with MicrophoneStream(RATE, CHUNK) as stream:
			audio_generator = stream.generator()
			requests = (speech.types.StreamingRecognizeRequest(audio_content=content)
						for content in audio_generator)
			
			
			# Now, put the transcription responses to use.
			responses = client.streaming_recognize(streaming_config, requests)
			self.listen_print_loop(responses)
			
	
	def listen_print_loop(self, responses):
		"""Iterates through server responses and prints them.

		The responses passed is a generator that will block until a response
		is provided by the server.

		Each response may contain multiple results, and each result may contain
		multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
		print only the transcription for the top alternative of the top result.

		In this case, responses are provided for interim results as well. If the
		response is an interim one, print a line feed at the end of it, to allow
		the next result to overwrite it, until the response is a final one. For the
		final one, print a newline to preserve the finalized transcription.
		"""
		for response in responses:
			if self.stop.is_set():
				# Stop running this thread so the main Python process can exit.
				return
			if not response.results:
				continue

			result = response.results[0]
			if not result.alternatives:
				continue
				
			global previousTimesArray
			global timesArray
			global wordsArray
			global NUMBER_OF_PEOPLE
			
			# Display the transcription of the top alternative.
			transcript = result.alternatives[0].transcript
			for i in range(0,NUMBER_OF_PEOPLE):
				timesArray[i] = previousTimesArray[i]
				wordsArray[i] = []
			for word in result.alternatives[0].words:
				startTime = float(word.start_time.seconds)+(float(word.start_time.nanos)/float(1000000000))
				endTime = float(word.end_time.seconds)+(float(word.end_time.nanos)/float(1000000000))
				timesArray[word.speaker_tag - 1] += endTime - startTime
				wordsArray[word.speaker_tag - 1].append(word.word)
			for i in range(0,NUMBER_OF_PEOPLE):
				print("Text persona " + str(i+1) + ": ")
				for word in wordsArray[i]:
					print(word, end=' ')
				print('')
				print("Temps total persona " + str(i+1) + ": " + str(timesArray[i]))
				print('')
			print('------------')
			
			global lastState
			tmpIndex = -1
			tmpDiff = 0
			
			for i in range(0,NUMBER_OF_PEOPLE):
				if lastState[i] != timesArray[i]:
					if self.lastIntIndex == i:
						self.lastIntTime = self.lastIntTime + (timesArray[i] - lastState[i])
					else:
						if ((timesArray[i] - lastState[i]) > tmpDiff):
							self.lastIntIndex = i
							self.lastIntTime = (timesArray[i] - lastState[i])
							tmpDiff = self.lastIntTime
			
			print("Temps ultima intervencio persona " + str(self.lastIntIndex+1) + ": " + str(self.lastIntTime))
			
			lastState = list(timesArray)

			
class PongApp(App):

	def on_stop(self):
        # The Kivy event loop is about to stop, set a stop signal;
        # otherwise the app window will close, but the Python process will
        # keep running until all secondary threads exit.
		self.root.stop.set()

	def build(self):
		programa = Zeus()
		programa.inicializar()
		Clock.schedule_interval(programa.update, 1.0 / 60.0)
		return programa


if __name__ == '__main__':
	ui = PongApp()
	ui.run()