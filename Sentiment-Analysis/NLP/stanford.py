import urllib
import json
url = "http://text-processing.com/api/sentiment/"


text = raw_input("Enter Text:")


googleResponse = urllib.urlopen(url,urllib.urlencode({'text':text}))
#googleResponse.add_header("Content-type",'application/x-www-form-urlencoded')
jsonResponse = json.loads(googleResponse.read())

a = dict(jsonResponse)
#print "The text is : " ,a['label']
b = dict(a['probability'])
#print "Negativeness : ",b['neg']
#print "Postiveness : " ,b['pos']
#print "Neutral-ness : ",b['neutral']
 
res = []
res.append((('neg'),b['neg']))
res.append((('pos'),b['pos']))
res.append((('neutral'),b['neutral']))

res.sort(reverse= True)
aux =  res[0]
print aux