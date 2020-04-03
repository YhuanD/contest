import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt

def convert_dur(x,y):
	if (math.isnan(x) == 1) or (math.isnan(y) == 1):
		return np.nan 
	else:
		return datetime.fromtimestamp(y)-datetime.fromtimestamp(x)
	
weather = pd.read_csv("2015年5月到2017年5月城市天气.csv",header=0)
#action = pd.read_csv("2015年5月到2017年5月航班动态数据.txt",sep=',',header=0)
#action = pd.read_csv("action.csv",header=0)
action = pd.read_csv("action_2.csv",header=0)
cities = pd.read_csv("机场城市对应表.csv",header=0)
special = pd.read_csv("2015年5月到2017年5月特情.csv",header=0)
submission_sample = pd.read_csv("submission_sample.csv",header=0)

weather.columns = ['city','weather','lowtemp','hightemp','date']
#action.columns = ['dep_port','arr_port','airline','planned_deptime','planned_arrtime','real_deptime','real_arrtime','plane_id','canceled_or_not']
#mapping = {'正常': 1, '取消': 0}
#action = action.replace({'canceled_or_not':mapping})
cities.columns = ['airport','city']
special.columns = ['airport','collect_time','start_time','end_time','content']

# action---------------------
#action = action.drop_duplicates()
#
# the month need to predict is June, and the weather is highly dependent on month in a year, so extract 15th,May ~ 15th,July of 2015 and 2016 historical data only
'''
start_ts1 = int(time.mktime([2015,5,15,0,0,0,0,0,0]))
end_ts1 = int(time.mktime([2015,7,15,0,0,0,0,0,0]))
start_ts2 = int(time.mktime([2016,5,15,0,0,0,0,0,0]))
end_ts2 = int(time.mktime([2016,7,15,0,0,0,0,0,0]))
start_ts3 = int(time.mktime([2017,5,15,0,0,0,0,0,0]))
end_ts3 = int(time.mktime([2017,7,15,0,0,0,0,0,0]))
laction = ((action['planned_deptime'] > start_ts1) & (action['planned_deptime'] < end_ts1)) | ((action['planned_deptime'] > start_ts2) & (action['planned_deptime'] < end_ts2)) | ((action['planned_deptime'] > start_ts3) & (action['planned_deptime'] < end_ts3))
action = action[laction]
action = action.drop_duplicates()
action = action.reset_index(drop=True)
action.to_csv("action_2.csv",index=False)
'''

# number of nan in action
'''action.isnull().sum()
real_deptime       71486
real_arrtime       71280
plane_id           32024
others 		   0
len(action) = 1325127
71486/1325127.0 = 5.4%
Q: thoes are canceled?
len(action[action.canceled_or_not==0])  -- 71302
len(action[action.canceled_or_not==0 & action.real_deptime.isnull()])  --- 71302
A: roughly, lines where real_deptime is null are all canceled
len(action[action.canceled_or_not==1 & action.real_deptime.isnull()]) ----564
A: some that have no time information are not canceled 
'''
# norm_dur : norm duration = planned_arrtime - planned_deptime
# real_dur = real_arrtime - planned_deptime <--!!!!!!
# type of duration:  datetime.timedelta   .days, .seconds
action['norm_dur'] = map(lambda x,y: datetime.fromtimestamp(y)-datetime.fromtimestamp(x),action['planned_deptime'],action['planned_arrtime'])
action['delay'] = map(convert_dur,action['planned_arrtime'],action['real_arrtime'])
action['real_dur'] = map(convert_dur,action['real_deptime'],action['real_arrtime'])
# abnormal records: len(action[action.planned_deptime == action.planned_arrtime])/float(len(action)) 1%
# planned_deptime == planned_arrtime
action = action[action.planned_deptime != action.planned_arrtime]
# depart before planned dep time (20min before)---see as abnormal ---0.04%
action['plan-real_deptime'] = map(convert_dur,action['real_deptime'],action['planned_deptime'])
action = action[~(action['plan-real_deptime'] > pd.Timedelta('0 days 00:20:00'))]
# in action history, the percentage of delaying longer than 3 hours:
# len(action[action.delay > pd.Timedelta('0 days 03:00:00')])/float(len(action)) ----- 3.6%; 3h10m: 3.27%; 3h20m: 2.94%
# > 4 hours: 1.98%; > 5 hours: 1.1%; > 6 hours: 0.6%;------ > 2 hours: 7%; > 1 hour: 14.9%; > 30 min: 23.6%
# len(action[action.canceled_or_not == 0])/float(len(action)) ----- 5.2% flights are canceled
# len(set(submission_sample.Flightno)- set(action.airline)) -- 790 out of 144396 new airlines that does not appear in action history
# compare with pd.Timedelta('0 days 03:00:00'), consider prop, maybe set different weights 

# split first two chars of airline

#---------------------------------
# extract the special data
start1 = '2015-05-15 00:00:00Z'
end1 = '2015-07-15 00:00:00Z'
start2 = '2016-05-15 00:00:00Z'
end2 = '2016-07-15 00:00:00Z'
start3 = '2017-05-15 00:00:00Z'
end3 = '2017-07-15 00:00:00Z'
lspecial = ((special['collect_time'] > start1) & (special['collect_time'] < end1)) | ((special['collect_time'] > start2) & (special['collect_time'] < end2)) | ((special['collect_time'] > start3) & (special['collect_time'] < end3))
special = special[lspecial]
special = special.reset_index()
# three hours = 60*3*60 = 10800 seconds
#(datetime.fromtimestamp(1463357100) - datetime.fromtimestamp(1463352000)).seconds
#(datetime.fromtimestamp(1463357100) - datetime.fromtimestamp(1463352000)).days

#---------submission sample
submission_sample['norm_dur'] = map(lambda x,y: datetime.fromtimestamp(y)-datetime.fromtimestamp(x),submission_sample['PlannedDeptime'],submission_sample['PlannedArrtime'])
