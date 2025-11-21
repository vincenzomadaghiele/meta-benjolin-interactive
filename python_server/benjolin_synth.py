from signalflow import *
import numpy as np
import time
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    try:
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            os.close(fd)

class BenjolinSynth:
	'''A synthesizer that loads a synthesis patch'''
	def __init__(self, 
				startup_synth_parameters, 
				sample_rate=44100
				):

		self.config = AudioGraphConfig()
		self.config.sample_rate = sample_rate
		self.startup_synth_parameters = startup_synth_parameters
		self.is_playing = False
		self.is_buffer_synth_playing = False  # Track buffer synth separately
		self.connectGraph()

	def connectGraph(self, parameters=None):
		# Suppress AudioGraph status logs
		with suppress_stdout_stderr():
			self.graph = AudioGraph(self.config)
		# Use provided parameters or fall back to startup parameters
		params = parameters if parameters is not None else self.startup_synth_parameters
		self.synth = BenjolinPatch(params, self.graph)
		self.synthForBuffer = BenjolinPatch(params, self.graph)
		# Don't auto-play on initialization

	def resetGraph(self, parameters=None):
		if self.graph:
			self.graph.destroy()
		self.connectGraph(parameters)

	def fadeParameters(self, new_parameters: list):
		for i, new_p in enumerate(new_parameters):
			current_p = self.synth.inputs[f"parameter-{i}"].value
			p_fade = Line(current_p, new_p, self.synth.fade_time)
			self.synth.set_input(f"parameter-{i}", p_fade) 

	def resetParameters(self, new_parameters: list):
		for i, new_p in enumerate(new_parameters):
			self.synth.set_input(f"parameter-{i}", new_p)

	def resetBufferParameters(self, new_parameters: list):
		"""Update synthForBuffer parameters independently from self.synth"""
		for i, new_p in enumerate(new_parameters):
			self.synthForBuffer.set_input(f"parameter-{i}", new_p) 

	def getInputs(self):
		return self.synth.inputs

	def play(self, synth_parameters: list):
		'''Update parameters and start playing'''
		if not self.is_playing:
			print(f"Graph not playing, starting playback")
			try:
				with suppress_stdout_stderr():
					self.graph.play(self.synth)
				self.is_playing = True
			except Exception as e:
				# If node is already playing, reset and try again
				print(f"Error starting graph, resetting: {e}")
				self.resetGraph(synth_parameters)
				with suppress_stdout_stderr():
					self.graph.play(self.synth)
				self.is_playing = True
		else:
			print(f"Graph already playing, updating parameters with: {synth_parameters}")
		
		self.fadeParameters(synth_parameters)

	def stop(self):
		'''Stop audio playback'''
		if self.is_playing and self.graph:
			self.graph.stop()

	def render_audio_as_buffer(self, synth_parameters: list, duration_seconds=1.0):
		'''
		Capture audio with given parameters for analysis WITHOUT producing audible sound.
		Uses synthForBuffer with zero gain to silently render audio.
		Does not interrupt playback from self.synth.
		
		Args:
			synth_parameters: List of 9 parameters (8 benjolin + gain)
			duration_seconds: Duration of audio to capture (default: 1.0 second)
			
		Returns:
			numpy array: Captured audio buffer as 1D float32 array (mono)
		'''
		print(f"Capturing audio with parameters: {synth_parameters[:8]}...")  # Don't print gain
		
		# Calculate number of samples
		num_samples = int(duration_seconds * self.config.sample_rate)
		
		# Create a buffer to store the captured audio
		buffer = Buffer(1, num_samples)
		
		# Create parameters with zero gain for silent rendering
		silent_params = synth_parameters[:8] + [0.0]  # Use first 8 params + zero gain
		
		# Update synthForBuffer with silent parameters (doesn't affect self.synth)
		self.resetBufferParameters(silent_params)
		
		# Create a BufferRecorder to capture the output without interrupting playback
		recorder = BufferRecorder(buffer, self.synthForBuffer.output)
		
		# Start the buffer synth if not already playing
		if not self.is_buffer_synth_playing:
			with suppress_stdout_stderr():
				self.graph.play(self.synthForBuffer)
			self.is_buffer_synth_playing = True
		
		# Start recording (runs in parallel with playback)
		with suppress_stdout_stderr():
			self.graph.play(recorder)
		
		# Wait for recording to complete
		time.sleep(duration_seconds + 0.1)  # Add small buffer
		
		# Stop only the recorder, not the synth
		with suppress_stdout_stderr():
			recorder.stop()
		
		# Get the audio data from the buffer
		audio_buffer = np.array(buffer.data[0][:num_samples], dtype=np.float32)
		
		print(f"Captured {len(audio_buffer)} samples")
		# Return as 1D numpy array (mono) for get_features
		return audio_buffer


class BenjolinPatch(Patch):
	def __init__(self, parameter_values: list, graph):
		super().__init__()

		FRQ01 = self.add_input("parameter-0", parameter_values[0])
		FRQ02 = self.add_input("parameter-1", parameter_values[1])
		RUN01 = self.add_input("parameter-2", parameter_values[2])
		RUN02 = self.add_input("parameter-3", parameter_values[3])
		FIL_FRQ = self.add_input("parameter-4", parameter_values[4])
		FIL_RES = self.add_input("parameter-5", parameter_values[5])
		FIL_RUN = self.add_input("parameter-6", parameter_values[6])
		FIL_SWP = self.add_input("parameter-7", parameter_values[7])
		gain = self.add_input("parameter-8", parameter_values[8])

		self.fade_time = 0.005

		## PARAMETERS
		FRQ01 = ScaleLinLin(input=FRQ01, a=0, b=1, c=0, d=127)
		FRQ02 = ScaleLinLin(input=FRQ02, a=0, b=1, c=0, d=127)
		RUN01 = ScaleLinLin(input=RUN01, a=0, b=1, c=0, d=127)
		RUN02 = ScaleLinLin(input=RUN02, a=0, b=1, c=0, d=127)
		FIL_FRQ = ScaleLinLin(input=FIL_FRQ, a=0, b=1, c=0, d=127)
		FIL_RES = ScaleLinLin(input=FIL_RES, a=0, b=1, c=0, d=127)
		FIL_RUN = ScaleLinLin(input=FIL_RUN, a=0, b=1, c=0, d=127)
		FIL_SWP = ScaleLinLin(input=FIL_SWP, a=0, b=1, c=0, d=127)

		buf = Buffer(1, 256)
		buf_pulse01 = Buffer(1, 256)
		buf_pulse02 = Buffer(1, 256)
		sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8 = 0, 0, 0, 0, 0, 0, 0, 0

		## RUNGLER
		XOR = Pow(a=GreaterThan(a=Clip(input=FeedbackBufferReader(buf_pulse01), min=0, max=1.0), b=0.5), 
			b=FeedbackBufferReader(buf))
		clock = GreaterThan(a=FeedbackBufferReader(buf_pulse02), b=0)
		sh1 = SampleAndHold(input=XOR, clock=clock)
		sh2 = SampleAndHold(input=sh1, clock=clock)
		sh3 = SampleAndHold(input=sh2, clock=clock)
		sh4 = SampleAndHold(input=sh3, clock=clock)
		sh5 = SampleAndHold(input=sh4, clock=clock)
		sh6 = SampleAndHold(input=sh5, clock=clock)
		sh7 = SampleAndHold(input=sh6, clock=clock)
		sh8 = SampleAndHold(input=sh7, clock=clock)
		graph.add_node(FeedbackBufferWriter(buf, sh8, graph.output_buffer_size / graph.sample_rate))
		RUNGLER = (sh6/8) + (sh7/4) + (sh8/2)

		## OSCILLATOR 1
		O1 = ScaleLinLin(input=FRQ01, a=0, b=127, c=-61, d=80) + RUNGLER * (RUN01 / 2)
		O1 = MidiNoteToFrequency(a=Clip(input=O1, min=-60, max=127))
		O1_tri = TriangleOscillator(frequency=O1)
		O1_pls = SquareOscillator(frequency=O1)
		graph.add_node(FeedbackBufferWriter(buf_pulse01, O1_pls, graph.output_buffer_size / graph.sample_rate))

		## OSCILLATOR 2
		O2 = ScaleLinLin(input=FRQ02, a=0, b=127, c=-59, d=80) + RUNGLER * (RUN02 / 2)
		O2 = MidiNoteToFrequency(a=Clip(input=O2, min=-60, max=127))
		O2_tri = TriangleOscillator(frequency=O2)
		O2_pls = SquareOscillator(frequency=O2)
		graph.add_node(FeedbackBufferWriter(buf_pulse02, O2_pls, graph.output_buffer_size / graph.sample_rate))

		PWM = GreaterThan(a=O1_tri, b=O2_tri)
		output = (RUNGLER + PWM) * 0.5

		## FILTER
		filter_f0 = FIL_FRQ + (FIL_RUN * RUNGLER * 127) + (FIL_SWP * O2_tri)
		filter_f0 = MidiNoteToFrequency(a = Clip(input=filter_f0, min=0, max=127))
		filter_res = ScaleLinLin(input=FIL_RES, a=0, b=127, c=0, d=1)
		filter_out = SVFilter(input=output, filter_type='band_pass', cutoff=filter_f0, resonance=filter_res)
		output = filter_out * ScaleLinLin(input=FIL_RES, a=0, b=127, c=2, d=12)
		output = SVFilter(input=output, filter_type='high_pass', cutoff=10, resonance=0.0)
		output = Compressor(input=output, threshold=0.2, ratio=4, attack_time=0.01, release_time=0.1)
		# output = output * DecibelsToAmplitude(gain)
		output = output * gain

		self.output = output
		self.set_auto_free(False)



if __name__ == "__main__":

	N_params = 9 # 8 benjolin paramters + gain
	synth_parameters = np.random.rand(N_params).tolist() # parameters are scaled between 0 and 1
	print(f'synth parameters: {synth_parameters}')

	# CREATE BENJOLIN INSTANCE
	synth = BenjolinSynth(synth_parameters)

	while True:
		synth_parameters = np.random.rand(N_params).tolist()
		print(f'synth parameters: {synth_parameters}')
		# change parameters fading from old values to new values
		synth.fadeParameters(synth_parameters)
		time.sleep(1)


