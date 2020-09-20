import math

import pandas
import statsmodels.formula.api as sm

# import the csv as a DF
import numpy

data = pandas.read_csv("C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\MLB_Prospect_Predictor\\Pitch_Data"
                       "\\pitch_project_data"
                       ".csv")

# add the new normalized statistics
data['wrate'] = data.wins / data.gm
data['rrate'] = data.runs / data.ip
data['srate'] = data.so / data.ip
data['lhits9'] = numpy.log(data.hits9)
data['lbb9'] = numpy.log(data.bb9)
data['lso9'] = numpy.log(data.so9)

# Lets now take only the vars we care about.
selected_vars = ['org_top10_post', 'levelID', 'wrate', 'rrate', 'lhits9', 'lbb9', 'lso9',
                 'whip', 'milb_exp', 'year']
dfa = data[selected_vars]

dfa_model = dfa.replace([pandas.np.inf, -pandas.np.inf], pandas.np.nan).dropna(axis=0)


results = sm.wls(data=dfa_model,
                 formula='org_top10_post ~ levelID*levelID + wrate + rrate + lhits9 + lbb9 + lso9 + '
                        '+ whip + milb_exp', missing='drop').fit()

print dfa[['levelID', 'wrate', 'rrate', 'lhits9', 'lbb9', 'lso9', 'whip']].cov().to_string()

print results.params
print results.summary()

# return a list of 2018 prospects ordered by likedlihood of top 10 org
#Remove draft_round bc tiny value
#IP introduces a low effect and a ton of noise through covariance
#Hr/9, era, so_bb
#Logs

#Now we find top 10 in the org
#30 teams with 10 prospects, so top 300 in likelihood from the model developed

data['predicted'] = results.predict(data)

data_2018 = data.drop(data[data.year != 2018].index)

data_2018 = data_2018[data_2018.predicted != pandas.np.inf]


top_prospects = data_2018.nlargest(300, 'predicted', keep='first')  # type: object

top_prospects.to_csv('top_prospects')