import profile
from understatapi import UnderstatClient
import json



def getUnderstatData():
    understat = UnderstatClient()
    understatdata = understat.league(league="EPL").get_match_data(2025)
    return understatdata



def sortCompleted(data):
  results = []
  fixtures = []

  
  for match in data:
    if match['isResult']:
      results.append(match)
    else:
      fixtures.append(match)
  return {'results': results, 'fixtures': fixtures}

def createDataJson(understatdata):
  file_path = 'data.json'

  try : 

    with open(file_path, 'r', encoding='utf-8') as json_file:
      try :
        data = json.load(json_file)
        print('Found Data')
      except :
        print('Empty File')
        pass
  except :
    open(file_path, 'w')


    with open(file_path, 'r', encoding='utf-8') as json_file:
      try :
        data = json.load(json_file)
        print('Found Data')
      except :
        print('Empty File')
        pass

  data = understatdata

  completed = sortCompleted(data)

  finaldata = {}
  finaldata['results'] = completed['results']
  finaldata['fixtures'] = completed['fixtures']

  with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(finaldata, json_file, indent=4)