import urllib.request, json
B="http://127.0.0.1:8000"
def post(p,d): 
    r=urllib.request.Request(B+p,json.dumps(d).encode(),{"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r).read())
# 1) cria uma análise para teres um analysis_id
a=post("/analyze",{"patient":{"patient_id":"T001","age":70,"sex":"F","main_problem":"pain","active_medications":["clopidogrel"]},"prescription":[{"medication":"ibuprofen"}]})
print("analysis_id:",a["analysis_id"])
# 2) regista o desfecho
print(post("/outcome",{"analysis_id":a["analysis_id"],"medication":"paracetamol","outcome":"resolved","comment":"melhorou em 1 semana"}))
# 3) lista pendentes (min_days=0 mostra todas sem desfecho)
print(urllib.request.urlopen(B+"/outcomes/pending?min_days=0").read().decode()[:300])