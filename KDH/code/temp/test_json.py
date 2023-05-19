import json
import os

data = {}
data['playlist'] = []
data['playlist'].append({
    "singer": "TAEYEON",
    "song": "Weekend",
    "date": 20210706
})
data['playlist'].append({
    "singer": "almost monday",
    "song": "til the end of time",
    "date": 20210709
})

with open('/opt/ml/dataset/example.json', 'w') as f:
    json.dump(data, f, indent=4)