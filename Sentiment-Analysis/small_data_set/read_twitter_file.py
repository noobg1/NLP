temp_list=[]
temp=""
with open('twitter_text.txt',"rb") as f:
	temp_list=f.read().splitlines()
f_pos=open("pos_tweet","w")
f_neg=open("neg_tweet","w")
f_neutral=open("neutral_tweet","w")
pos_tweet=[]
neutral_tweet=[]
neg_tweet=[]
for t in temp_list:
	temp=t.replace('\t',' ')
	temp=temp.split()	
	# print temp[1]	
	if float(temp[1])>=0:
		temp_str=""
		# print "a"
		for t1 in temp[2:]:
			temp_str=temp_str+" "+t1
		f_pos.write(temp_str+'\n')
	elif float(temp[1])<0:
		temp_str=""
		for t1 in temp[2:]:
			temp_str=temp_str+" "+t1
		f_neg.write(temp_str+'\n')
	# else:
	# 	temp_str=""
	# 	for t1 in temp[2:]:
	# 		temp_str=temp_str+" "+t1
	# 	f_neutral.write(temp_str+'\n')
	# temp1=temp.split()
	# print (temp1[1])	
