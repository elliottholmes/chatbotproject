import profile
from understatapi import UnderstatClient
import json
import predict as pred



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

def headtohead(teams):
  if len(teams) < 2:
      return "I need two teams."

  home = teams[0]
  away = teams[1]

  hid = pred.title_to_id.get(home)
  aid = pred.title_to_id.get(away)

  data = []

  with open('data.json', 'r', encoding='utf-8') as json_file:
    try :
      data = json.load(json_file)
    except :
      pass

  if not data:
      return "No H2H data found."

  hw = 0
  aw = 0
  d = 0

  hg = 0
  ag = 0
  
  for i in range(len(data['results'])):
    result = data['results'][i]
    
    if result['h']['id'] == hid and result['a']['id'] == aid:
      hg += int(result['goals']['h'])
      ag += int(result['goals']['a'])
      
    if result['a']['id'] == hid and result['h']['id'] == aid:
      ag += int(result['goals']['h'])
      hg += int(result['goals']['a'])

    
    if hid == result['h']['id'] and result['a']['id'] == aid:
      if int(result['goals']['h']) > int(result['goals']['a']):
        hw += 1
      elif int(result['goals']['h']) == int(result['goals']['a']):
        d += 1
      elif int(result['goals']['h']) < int(result['goals']['a']):
        aw += 1
    elif hid == result['a']['id'] and result['h']['id'] == aid:
      print('home team away')
      if int(result['goals']['a']) > int(result['goals']['h']):
        hw += 1
      elif int(result['goals']['a']) == int(result['goals']['h']):
        d += 1
      elif int(result['goals']['a']) < int(result['goals']['h']):
        aw += 1

  return (
      f"{home.title()} vs {away.title()}:\n"
      f"{home.title()} wins: {hw}\n"
      f"Draws: {d}\n"
      f"{away.title()} wins: {aw} \n"
      f"Aggregate : {home.title()} {hg} - {ag} {away.title()}"
  )
  
  
  '''
  for g in games:

  hg = int(g["goals"]["h"])
  ag = int(g["goals"]["a"])

  if hg == ag:
      d += 1

  elif (
      (g["h"]["id"] == hid and hg > ag)
      or
      (g["a"]["id"] == hid and ag > hg)
  ):
      hw += 1

  else:
      aw += 1
  '''

  return data
  
  return (
      f"{home.title()} vs {away.title()}:\n"
      f"{home.title()} wins: {hw}\n"
      f"Draws: {d}\n"
      f"{away.title()} wins: {aw}"
  )
