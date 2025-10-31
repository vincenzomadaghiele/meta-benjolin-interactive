from signalflow import *
import numpy as np
import time
import soundfile as sf
import itertools

class BenjolinSynthOffline:
	'''A synthesizer that loads a synthesis patch'''
	def __init__(self, 
				startup_synth_parameters, 
				sample_rate=44100,
				window_size = 2048,
				hop_size = 512
				):

		self.config = AudioGraphConfig()
		self.config.sample_rate = sample_rate
		self.startup_synth_parameters = startup_synth_parameters
		self.window_size = window_size
		self.hop_size = hop_size
		self.connectGraph()

	def connectGraph(self):
		self.graph = AudioGraph(self.config, output_device=AudioOut_Dummy(2), start=False)
		self.synth = BenjolinPatch(self.startup_synth_parameters, self.graph) # generalize this class
		self.graph.play(self.synth)
		self.buffer = Buffer(1, self.hop_size)
		self.window = np.zeros([self.window_size], dtype=float, order='C')
		# initialize FFT window
		for _ in range(int(self.window_size / self.hop_size)):
			self.graph.render_to_buffer(self.buffer)
			rendered_array = self.buffer.data[0,:]
			self.window[:-self.hop_size] = self.window[self.hop_size:]
			self.window[-self.hop_size:] = rendered_array # update FFT window

	def resetGraph(self):
		if self.graph:
			self.graph.destroy()
		self.connectGraph()

	def close(self):
		self.graph.destroy()

	def forward(self):
		# render audio in size [window size] with a hop [hop size]
		self.graph.render_to_buffer(self.buffer)
		rendered_array = self.buffer.data[0,:]
		self.window[:-self.hop_size] = self.window[self.hop_size:]
		self.window[-self.hop_size:] = rendered_array
		return self.window

	def fadeParameters(self, new_parameters: list):
		for i, new_p in enumerate(new_parameters):
			current_p = self.synth.inputs[f"parameter-{i}"].value
			p_fade = Line(current_p, new_p, self.synth.fade_time)
			self.synth.set_input(f"parameter-{i}", p_fade) 

	def resetParameters(self, new_parameters: list):
		for i, new_p in enumerate(new_parameters):
			self.synth.set_input(f"parameter-{i}", new_p) 

	def getInputs(self):
		return self.synth.inputs


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

	sr, window_size, hop_size = 44100, 2048, 1024
	N_synth_parameters = 9
	synth_parameters = np.random.rand(N_synth_parameters).tolist()
	synth = BenjolinSynthOffline(synth_parameters, sr, window_size, hop_size)

	duration_s = 30
	duration_samples = sr * duration_s 
	N_reps = int(duration_samples / hop_size)
	param_step_for_corpus_gen = 10
	elements = np.array(range(0, param_step_for_corpus_gen)) / param_step_for_corpus_gen
	permutations = [p for p in itertools.product(elements, repeat=N_synth_parameters-1)]
	synth_samples_features = []
	for param_list in permutations:
		param_list = np.array(param_list).tolist()
		param_list.append(1)
		print('-'.join(map(str, param_list)))
		synth.resetParameters(param_list)
		soundfile_array = np.zeros((hop_size*N_reps))
		for i in range(N_reps):
			soundfile_array[i*hop_size:(i+1)*hop_size] = synth.forward()[-hop_size:]
		sf.write(f'benjolin_corpus/{'-'.join(map(str, param_list))}.wav', soundfile_array, sr, subtype='PCM_24')


