from nltk.corpus import nps_chat
import sentiment_mod as s
posts = nps_chat.xml_posts()
print(len(posts))
output = open("out.txt","a")

<<<<<<< HEAD
"""for sen in posts:
=======
for sen in posts:
>>>>>>> 047065b02b8824ce88e5d2849cf4ba56c77b48e5
	print(sen.text )
	print(s.sentiment(sen.text) )
	print(" ")
	print ((s.sentiment(sen.text)))
	#sentiment_value = s.sentiment(sen.text)[1]
	output.write(str(s.sentiment(sen.text)[0]))
	output.write('\n')

<<<<<<< HEAD
output.close()"""

for sen in posts:
	output.write(sen.text)
	output.write('\n')	
=======
output.close()	
>>>>>>> 047065b02b8824ce88e5d2849cf4ba56c77b48e5
