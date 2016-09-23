'''from nltk.corpus import nps_chat
import sentiment_mod as s
posts = nps_chat.xml_posts()
#sent = input()
#sent = "This is my house. I live here. Lets plAY BALL .World War 1 I am extremely happy with this neighbourhood. I am unhappy with the ground floor."
sent = "Habitat depicts the post- apolyptic world after World War III.The main protogonist is a boy from Varda who aims to find his parents. His journey is not without challenges. He has to fight prirates, dictators and last but the least the demons. It is a fun thrilling and exciting movie. Everyine should definitely watch it."
print(sent )
print(s.sentiment(sent) )
print(" ")

#sentiment_value = s.sentiment(sen.text)[1]'''

from nltk.corpus import nps_chat
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
#import sentiment_mod as s
posts = nps_chat.xml_posts()
sentences = []
for post in posts:
	sentences.append(post.text)


#sentences = ["Habitat depicts the post- apolyptic world after World War III." ,"The main protogonist is a boy from Varda who aims to find his parents.","His journey is not without challenges.","He has to fight prirates, dictators and last but the least the demons.","It is a fun thrilling and exciting movie.","Everyine should definitely watch it."]
for sentence in sentences:
    print(sentence)
    ss = vaderSentiment(sentence) 
    print (ss)
	