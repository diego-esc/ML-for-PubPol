
from sodapy import Socrata
import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import censusgeocode as cg
import numpy as np
from scipy import stats

# Problem 1.1: Data
# Download data using a data token/key:
client = Socrata("data.cityofchicago.org", "WZEdPks1IM6q8xFmGAEpMjUrf")
data = client.get("6zsd-86xi", where="year = 2017 or year = 2018", 
	limit=1000000)
df_crime = pd.DataFrame.from_dict(data)

# Crimes by type:
df_crime['primary_type'].value_counts()
# Crimes type by year:
cross_year = pd.crosstab(df_crime['primary_type'], df_crime['year'])
# Crimes type by area:
cross_area = pd.crosstab(df_crime['primary_type'], df_crime['community_area'], 
	normalize='columns')
plt.figure(figsize=(10,5))
sns.heatmap(cross_area)
plt.show()

cross_area = pd.crosstab(df_crime['primary_type'], df_crime['community_area'])
plt.figure(figsize=(10,5))
sns.heatmap(cross_area)
plt.show()


# Merge with geographical data:
df_locations = df_crime[(df_crime.longitude.isnull()==False) & 
	(df_crime.latitude.isnull()==False)]
df_locations = df_crime[(df_crime.longitude.isnull()==False) & 
(df_crime.latitude.isnull()==False)]

df_locations['address'] = df_locations['block'].str.replace('XX','01')
df_locations['city'] = 'Chicago'
df_locations['state'] = 'IL'
df_locations = df_locations[['address', 'city', 'state']]
df_locations = df_locations.sample(frac=0.1, random_state=1)
df_locations = df_locations.reset_index()
df_locations.index.names = ['id'] # This is the format required by the geocoder
df_locations.drop('index', axis=1, inplace=True)

df_locations.to_csv('csv_locations.csv')

results = censusbatchgeocoder.geocode("csv_locations.csv", zipcode=None)

# Part 2:
url_acs = "https://api.census.gov/data/2017/acs/acs5?"
what_to_get = ",NAME&for=block%20group:*&in=state:17&in=county:031&in=tract:*"

request_url = url_acs + "get=B03003_002E"+ what_to_get
response = requests.get(request_url)
df_acs = pd.DataFrame(response.json()[1:], columns=response.json()[0])
df_acs['B03003_002E'] = df_acs['B03003_002E'].astype('int32')

variables = ["B03003_001E", "B03003_003E", "B02001_001E", "B02001_003E", 
	"B01001_001E", "B01001_002E", "B19001_001E"]
for variable in variables:
	request_url = url_acs + "get=" + variable + what_to_get
	response = requests.get(request_url)
	df_temp = pd.DataFrame(response.json()[1:], columns=response.json()[0])
	df_temp.drop(['NAME', 'state', 'county'], axis=1, inplace=True)
	df_temp[variable] = df_temp[variable].astype('int32')
	df_acs = df_acs.merge(df_temp, on=['tract', 'block group'])

# Variables Meaning:
# B03003_001E total; B03003_003E hispanic; 
# B02001_001E all; B02001_003E black;
# B19001_001E total income 2017 dollars inflation adjusted
# B01001_001E gender-all; B01001_002E males
df_acs['prop_hisp'] = df_acs.B03003_003E/df_acs.B03003_001E
df_acs['prop_black'] = df_acs.B02001_003E/df_acs.B02001_001E
df_acs['prop_male'] = df_acs.B01001_002E/df_acs.B01001_001E
df_acs['income'] = df_acs.B19001_001E

# Problem 3:
df_crime['date'] = pd.to_datetime(df_crime['date'])
df_crime['period'] = np.where((df_crime['date'] >= '2017-6-28 00:00:00') & 
	(df_crime['date'] < '2017-7-28 00:00:00'), "pre", None)
df_crime['period'] = np.where((df_crime['date'] >= '2018-6-28 00:00:00') & 
	(df_crime['date'] < '2018-7-28 00:00:00'), "post", df_crime['period'])
cross_alderman = pd.crosstab(df_crime['primary_type'], df_crime['period'])

# Problem 4:
def find_block(address):
	result = cg.address(address, city='Chicago', state='IL', zipcode='60616')
	tract = result[0]['geographies']['2010 Census Blocks'][0]['TRACT']
	block = result[0]['geographies']['2010 Census Blocks'][0]['BLOCK']
	block_group = result[0]['geographies']['2010 Census Blocks'][0]['BLKGRP']
	return tract, block, block_group

# a:
address = '2111 S Michigan Ave'
tract, block, block_group = find_block(address)
df_michigan = df_crime[df_crime['community_area']=='33']
df_michigan['primary_type'].value_counts()

# b:
address_uptown = '4800 N Winthrop Ave'
address_garfield = '5 N Kedzie Ave'
tract1, block1, block_group1 = find_block(address_uptown)
tract2, block2, block_group2 = find_block(address_garfield)
df_theft = df_crime[(df_crime['community_area']=='27') | 
	(df_crime['community_area']=='6')]
cross_area = pd.crosstab(df_theft['primary_type'], df_theft['community_area'])

# c:
# Using bayes rule we get:
# P(G|B)=P(B|G)*P(G)/(P(B|G)*P(G)+P(B|U)*P(U))
p_g_b = (100/600)*(600/1000)/((100/600)*(600/1000)+(160/400)*(400/1000))


