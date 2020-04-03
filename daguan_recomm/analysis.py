import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math
from itertools import islice
from sklearn.cluster import KMeans

def split_by_lengths(input_list, num_list):
    it = iter(input_list)
    out =  [x for x in (list(islice(it, n)) for n in num_list) if x]
    remain = list(it)
    return out if not remain else out + [remain]	

def createC1(dataset):
	c1 = []
	for transaction in dataset:
		for item in transaction:
			if [item] not in c1:
				c1.append([item])
	c1.sort()
	return map(frozenset, c1)

def scanD(dataset, candidates, min_support):
	"Returns all candidates that meets a minimum support level"
	sscnt = {}
	for tid in dataset:
	    for can in candidates:
	        if can.issubset(tid):
	            sscnt.setdefault(can, 0)
	            sscnt[can] += 1

	num_items = float(len(dataset))
	retlist = []
	support_data = {}
	for key in sscnt:
		support = sscnt[key] / num_items
		if support >= min_support:
		    retlist.insert(0, key)
		support_data[key] = support
	return retlist, support_data

def aprioriGen(freq_sets, k):
	#"Generate the joint transactions from candidate sets"
	retList = []
	lenLk = len(freq_sets)
	for i in range(lenLk):
		for j in range(i + 1, lenLk):
			L1 = list(freq_sets[i])[:k - 2]
			L2 = list(freq_sets[j])[:k - 2]
			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(freq_sets[i] | freq_sets[j])
	return retList

def apriori(dataset, minsupport=0.5):
	#"Generate a list of candidate item sets"
	C1 = createC1(dataset)
	D = map(set, dataset)
	L1, support_data = scanD(D, C1, minsupport)
	L = [L1]
	k = 2
	while (len(L[k - 2]) > 0):
		Ck = aprioriGen(L[k - 2], k)
		Lk, supK = scanD(D, Ck, minsupport)
		support_data.update(supK)
		L.append(Lk)
		k += 1
	return L, support_data
def generateRules(L, support_data, min_confidence=0.7):
	"""Create the association rules
	L: list of frequent item sets
    support_data: support data for those itemsets
    min_confidence: minimum confidence threshold
    """
	rules = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			print("freqSet", freqSet, 'H1', H1)
			if (i > 1):
				rules_from_conseq(freqSet, H1, support_data, rules, min_confidence)
			else:
				calc_confidence(freqSet, H1, support_data, rules, min_confidence)
	return rules

def calc_confidence(freqSet, H, support_data, rules, min_confidence=0.7):
	#Evaluate the rule generated
	pruned_H = []
	for conseq in H:
		conf = support_data[freqSet] / support_data[freqSet - conseq]
		if conf >= min_confidence:
			print(freqSet - conseq, '--->', conseq, 'conf:', conf)
			rules.append((freqSet - conseq, conseq, conf))
			pruned_H.append(conseq)
	return pruned_H

def rules_from_conseq(freqSet, H, support_data, rules, min_confidence=0.7):
	#"Generate a set of candidate rules"
	m = len(H[0])
	if (len(freqSet) > (m + 1)):
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calc_confidence(freqSet, Hmp1,  support_data, rules, min_confidence)
		if len(Hmp1) > 1:
			rules_from_conseq(freqSet, Hmp1, support_data, rules, min_confidence)

def norm_mat_row(input_mat):
	nrow = input_mat.shape[0]
	ndim = input_mat.shape[1]
	out_mat = np.zeros((nrow,ndim))	
	norm_arr = np.sqrt((input_mat * input_mat).sum(axis=1))
	for i in range(nrow):
		out_mat[i] = (input_mat[i]/float(norm_arr[i]))*10 #norm to 100 instead of 1
		out_mat[i] = np.around(out_mat[i],decimals=6) 
	return out_mat	

def coll_users(in_mat):
	nusers = in_mat.shape[0]
	out_list = []
	count = 0
	for iarr in in_mat:
		count = count + 1
		tmp_list = []
		for jarr in in_mat:
			tmp_value = round(np.linalg.norm(iarr - jarr),6)
			tmp_list.append(tmp_value)
		tmp_sort = sorted(tmp_list)
		tmp_out = []
		for kmin in tmp_sort[:7]:#top7 nearest neighbours
			tmp_out.append(tmp_list.index(kmin))
		out_list.append(tmp_out[1:])
		if count%100 == 0:
			print(count)
	return out_list

if __name__ == "__main__":

all_news_info = pd.read_csv("all_news_info.csv",header=0)
news_info = pd.read_csv("news_info.csv",header=0)
train = pd.read_csv("train.csv",header=0)

train.action_type = train.action_type.map(convert_type)
train['day'] = train.action_time.map(get_day)

news_info['date_ym'] = news_info.timestamp.map(get_ym)
news_info.date_ym.value_counts()
all_news_info['date_ym'] = all_news_info.timestamp.map(get_ym)

'''
subtrain1 = train[['user_id','cate_id','action_type']]
subtrain2 = train[['user_id','item_id','action_type']]

subtrain1 = subtrain1.groupby(['user_id','cate_id']).sum()
subtrain1 = subtrain1.unstack()
subtrain1[np.isnan(subtrain1)]=0
normalize(subtrain1,norm='l1',axis=1)

subtrain2 = subtrain2.groupby(['user_id','item_id']).sum()
subtrain2 = subtrain2.unstack()
subtrain2[np.isnan(subtrain2)]=0
normalize(subtrain2,norm='l1',axis=1)
'''
# we have 9999 number of items in news_info but not in train
add_news = set(news_info.item_id) - set(train.item_id)
# 33961 item_id in train but not in news_info
del_news = set(train.item_id) - set(news_info.item_id)

# check the cate distribution properties in train and news_info
# low freq cate appearing in both train and news_info: 
# delete cate 6_1, 4_1, 1_24 items
train.cate_id.value_counts()
train = train[~train.cate_id.isin(['6_1','4_1','1_24'])]
# delete cate 4_5, 6_1, 4_1 items in news info
news_info.cate_id.value_counts() 
news_info = news_info[~news_info.isin(['4_5','6_1','4_1'])]

train = pd.merge(train,all_news_info[['item_id','date_ym']],on='item_id')
train.date_ym.value_counts()

# remove the items with very low frequency in train
# low freq item list
item_lowfreq = train['item_id'].value_counts()
plt.hist(item_lowfreq.values, bins=range(0,50,1))
plt.show()
item_lowfreq = item_lowfreq[item_lowfreq.values<=4]

# inactive users 
users_inactive = train['user_id'].value_counts()
plt.hist(users_inactive.values, bins=range(0,700,10))
plt.show()
users_inactive = users_inactive[users_inactive.values <= 10]

# delete low freq items in train
train = train[~train['item_id'].isin(item_lowfreq.index)]

# delete inactive users
train = train[~train['user_id'].isin(users_inactive.index)]

# delete lines of very low freq cate '4_1'
train = train[train['cate_id']!='4_1']
# popularity of cate_id
train_cate = train[['user_id','cate_id','action_type']]
train_cate = train_cate.groupby(['user_id','cate_id'],as_index=False).sum()
# delete the lines where action_type <= 4, possibly click by mistake
# check the distribution 
tmp_hist = train_cate.action_type.value_counts()
plt.hist(tmp_hist,bins=range(1,30,1))
plt.show()
# delete 
train_cate = train_cate[train_cate.action_type>4]
# check the number of cates distribution
tmp_list = train_cate.user_id.value_counts()
plt.hist(tmp_list.values,bins=range(1,20,1))
plt.show()
train_cate = train_cate.reset_index(drop=True)

# check the distribution of number of items users read
train_list = train.user_id.value_counts()
plt.hist(train_list.values,bins=range(1,860,5))
plt.show()
#--------------------
# do apriori, association rules analysis to train_cate
tmp_user_list = train_cate.user_id.value_counts()
tmp_user_list = tmp_user_list.sort_index()
tmp_list1 = list(tmp_user_list.values)
tmp_list2 = list(train_cate['cate_id'])
apriori_list = split_by_lengths(tmp_list2,tmp_list1)
C1 = createC1(apriori_list)
D = map(set,apriori_list)
L1,support_data = scanD(D,C1,0.9)
aprioriGen(L1,2)
L = apriori(apriori_list)
# generate association rules
L, support_data = apriori(apriori_list,minsupport=0.9)
rules = generateRules(L,support_data,min_confidence=0.9)
#---------------------
# cate based clustering of users
cate_cluster = train_cate.pivot('user_id','cate_id','action_type')
cate_cluster = cate_cluster.fillna(0)
cate_cluster = cate_cluster.reset_index()

#check the distribution of number of cates in users
# 
len_list = []
for ilist in apriori_list:
	ilen = len(ilist)
	len_list.append(ilen)

plt.hist(len_list,bins=range(1,20,1))
plt.show()

# check action times of same user to same item, see times > 15 as abnormal action
train_item = train[['user_id','item_id','action_type']]
train_item.action_type = train_item.action_type.map(lambda x: 1)
train_item = train_item.groupby(['user_id','item_id'],as_index=False).sum()
train_item.action_type.value_counts()
ab_user_item = train_item[train_item.action_type>15]
train['user_item'] = train['user_id'].map(str) + train['item_id'].map(str)
user_item_set = ab_user_item['user_id'].map(str) + ab_user_item['item_id'].map(str)
user_item_set = set(user_item_set)
train = train[~train['user_item'].isin(user_item_set)]
# further cut the items, delete items apearing < 20, cut half of the items
item_cut = train.item_id.value_counts()
item_cut = item_cut[item_cut.values<=20]
item_cut = set(item_cut.index)
train = train[~train['item_id'].isin(item_cut)]
del train_item
# create user-item matrix for Collaborative Filtering
train_item = train[['user_id','item_id','action_type']]
train_item = train_item.groupby(['user_id','item_id'],as_index=False).sum()
tmp_df = train_item.pivot(index='user_id',columns='item_id', values='action_type')
tmp_df = tmp_df.reset_index()
tmp_df = tmp_df.fillna(0)
users = tmp_df['user_id']
items = tmp_df.columns
del tmp_df['user_id']
item_mat = tmp_df.as_matrix(columns=None)
item_mat = norm_mat_row(item_mat)
# most similar user neighbour lists
# neigh_list = coll_users(item_mat) # too slow to run
# group similar users by cate
# delete action_type <=6 after grouping
train_cate = train_cate[train_cate.action_type>6]
tmp_df2 = train_cate.pivot('user_id','cate_id','action_type')
tmp_df2 = tmp_df2.fillna(0)
tmp_df2 = tmp_df2.reset_index()
user_list2 = tmp_df2['user_id']
del tmp_df2['user_id']
cate_list2 = tmp_df2.columns
cate_mat = tmp_df2.as_matrix()
cate_mat = norm_mat_row(cate_mat)
#users_pair_cate = coll_users(cate_mat)#still too large to calcuate
user_lowcate = train_cate.user_id.value_counts()
user_lowcate = user_lowcate[user_lowcate.values<=4]
user_lowcate = user_lowcate.index
# separate train_cate into <=4 and >4 cates
#train_cate_sub1 = train_cate[train_cate.user_id.isin(user_lowcate)]
train_cate.cate_id.value_counts()
# delete 3_23,3_7,3_15,3_13,3_20
train_cate = train_cate[~train_cate.cate_id.isin(['3_23','3_7','3_15','3_13','3_20'])]
train_cate.user_id.value_counts()
# correlate cates to sep large matrix
# ['1_1','1_2','1_3','1_5','1_6','1_11','1_17']
train_cate_sub1 = train_cate[train_cate.cate_id.isin(['1_1','1_2','1_3','1_5','1_6','1_11','1_17'])] # cant sep failed!!!
# cut item, times < 50
item_list = train_item.item_id.value_counts()
item_list = item_list[item_list.values<=50]
train_item2 = train_item[~train_item.item_id.isin(item_list.index)]
#user_item 20, 20-70, >70
user_l = train_item2.user_id.value_counts()
user_l1 = user_l[user_l.values <= 20].index
user_l2 = user_l[user_l.values >= 60].index
user_l3 = user_l[(user_l.values < 60) & (user_l.values > 20)].index
train_item_sub1 = train_item[train_item.user_id.isin(user_l1)]
train_item_sub2 = train_item[train_item.user_id.isin(user_l2)]
train_item_sub3 = train_item[train_item.user_id.isin(user_l3)]
train_item_sub1 = train_item_sub1.groupby(['user_id','item_id'],as_index=False).sum()
tmp_df = train_item_sub1.pivot(index='user_id',columns='item_id', values='action_type')
tmp_df = tmp_df.reset_index()
tmp_df = tmp_df.fillna(0)
users_sub1 = tmp_df['user_id']
items_sub1 = tmp_df.columns
del tmp_df['user_id']
item_mat_sub1 = tmp_df.as_matrix(columns=None)
item_mat_sub1 = norm_mat_row(item_mat_sub1)
coll_users_sub1 = coll_users(item_mat_sub1)
# still too large
# cate sub similar way, <3, 3-6, >=7
user_l = train_cate.user_id.value_counts()
user_l1 = user_l[user_l.values < 3].index
user_l2 = user_l[user_l.values >= 7].index
user_l3 = user_l[(user_l.values < 7) & (user_l.values >= 3)].index
train_cate_sub1 = train_cate[train_cate.user_id.isin(user_l1)]
train_cate_sub2 = train_cate[train_cate.user_id.isin(user_l2)]
train_cate_sub3 = train_cate[train_cate.user_id.isin(user_l3)]
train_cate_sub1 = train_cate_sub1.groupby(['user_id','cate_id'],as_index=False).sum()
tmp_df = train_cate_sub1.pivot(index='user_id',columns='cate_id', values='action_type')
tmp_df = tmp_df.reset_index()
tmp_df = tmp_df.fillna(0)
users_sub1 = tmp_df['user_id']
items_sub1 = tmp_df.columns
del tmp_df['user_id']
cate_mat_sub1 = tmp_df.as_matrix(columns=None)
cate_mat_sub1 = norm_mat_row(cate_mat_sub1)
coll_users_sub1 = coll_users(cate_mat_sub1)
# check the popularity distribution of items
item_std = train_item.std()
plt.hist(item_std)
plt.show()
# col= items, row = user

