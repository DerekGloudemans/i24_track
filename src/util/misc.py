import configparser


def apply_config(obj,cfg,case = "DEFAULT"):
	# read config file	
	config = configparser.ConfigParser()
	config.read(cfg)
	params = config[case]

	# set object attributes according to config case
	# getany() automatically parses type (int,float,bool,string) from config
	[setattr(obj,key,config.getany(case,key)) for key in params.keys()]
