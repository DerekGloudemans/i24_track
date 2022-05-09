def tracker_dummy():
	print("Tracker Dummy Pass")



class Tracker():
	def __init__(self):
		raise NotImplementedError

	def preprocess(self):
		raise NotImplementedError

	def parse_detections(self):
		raise NotImplementedError

	def postprocess(self):
		raise NotImplementedError


class TrackState():

	def __init__(self):
		pass
	
	def as_dict(self):
		pass

	def as_tensors(self):
		pass

	def plot(self):
		pass

	def write_db(self):
		pass
