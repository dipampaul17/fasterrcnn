import os
import json

if __name__ == '__main__':
    ids_list = os.listdir('tomogram')
    match_list = json.load(open('density_map_to_tomogram__out_stat.json'))
    records = json.load(open('model_generation_imp__out.json'))
    for ids in ids_list:
        name = ids[:-4]
        model_id = []
        for item in match_list:
            if item['tomogram_uuid'] == name:
                model_id = item['model_id']
                break
        json.dump(records[str(model_id)], open('json/'+name+'.json','w'), indent=4)
