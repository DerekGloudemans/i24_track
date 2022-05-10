from .TrackState import TrackState


class Tracker():
	def __init__(self):
		raise NotImplementedError

	def preprocess(self):
        """
        Receives a TrackState object as input, as well as the times for 
        """
		raise NotImplementedError

	def parse_detections(self):
		raise NotImplementedError

	def postprocess(self):
		raise NotImplementedError



