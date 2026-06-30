import time, json
import httpx
from datetime import datetime
BASE='http://127.0.0.1:8000/v1'
INTENT='2bef6d57-e3a2-4714-87b1-84ef43758396'
client=httpx.Client(timeout=120.0)
print('Starting detailed simulation...')
resp=client.post(BASE+'/simulate/start', json={'user_intent_id':INTENT, 'disable_scraping': True, 'depth':2, 'branching_factor':3, 'mode':'detailed'})
resp.raise_for_status()
job_id=resp.json()['job_id']
print('job id', job_id)
last=None
for _ in range(360):
    sj=client.get(f'{BASE}/simulate/jobs/{job_id}')
    sj.raise_for_status()
    data=sj.json()
    if data['status']!=last:
        print('status', data['status'], 'progress', data.get('progress'), 'step', data.get('current_step'))
        last=data['status']
    if data['status'] in ('completed','failed'):
        break
    time.sleep(3)
if data['status']!='completed':
    print('Simulation failed or timed out', data)
else:
    g=client.get(f'{BASE}/graph/{INTENT}')
    g.raise_for_status()
    graph=g.json()
    out=f"run_output_detailed_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out,'w',encoding='utf-8') as f:
        json.dump(graph,f,ensure_ascii=False,indent=2)
    print('Saved', out)
    print('Nodes', len(graph.get('nodes',[])))
