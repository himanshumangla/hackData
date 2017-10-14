# hackData

RiskAnalysis

This is a Django web app that runs a model to predict a heart risk factor( i.e. the probability of a person having a heart disease) by analysing the vitals of an individual.
We have used The Cleveland Heart Disease Dataset: http://archive.ics.uci.edu/ml/datasets/heart+Disease
The use case is in Tier 2 and Tier 3 cities where there is a lack of specialised cardiologists. 
The model can be used to predict the risk factor of a person by performing simpler tests like based on Blood Pressure, Cholestrol level etc. which are easily available at rural medical setups as well.
It uses another model that analyzes heartbeat audio and classifies it as: normal, murmur or extra heart sound.
The two modules can be combined to give better suggestions.
