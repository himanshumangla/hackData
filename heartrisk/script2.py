from pyAudioAnalysis import audioAnalysis as aA

def func():
	ans = aA.classifyFolderWrapper("heartrisk/static/set", "svm", "heartrisk/static/svmMusicGenre3") 
	return ans
