import os
lists = []
with open("out.txt",'rw') as f:
	for l in f:
		index = l.find("User")
		l = l.replace("PART","")
		l = l.replace("JOIN","")
		l = l.replace("ACTION","")
		if index != -1:
			temp = ""
			temp = l[0:index-9] + l[index+5:]
			lists.append(temp)

		elif len(l) >= 2: lists.append(l)
	

with open("out_preprocessed_1.txt",'w') as f:
	for l in lists:
		f.write(l)		

for l in lists[0:30]:
	print l		