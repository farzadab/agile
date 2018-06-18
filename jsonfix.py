import json

def dump(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ': '))