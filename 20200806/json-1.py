import json
text = '{"name": "john", "age": 30, "city": "New York"}'

data = json.loads(text)
print(type(data))
print(data['name'])

with open('C:/Users/default.DESKTOP-6FG4SCS/Downloads/sample1.json') as f:
    data = json.load(f)
    print(data['lastName'])