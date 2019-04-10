'''
Homework 1 Diego Escobar
April 9th 2019
'''
from sodapy import Socrata
import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import censusgeocode as cg
from shapely.geometry import shape
from shapely.geometry import Point
import rtree

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
def crosstab_plt(df, var1, var2, _norm=False, fig_size=(10,5)):
	'''
	Creates a graph of the cross-tabulation of the data.
	'''
	crossing = pd.crosstab(df[var1], df[var2], normalize=_norm)
	figure = plt.figure(figsize=fig_size)
	sns.heatmap(crossing)
	return figure

fig1 = crosstab_plt(df_crime, 'primary_type', 'community_area')
fig2 = crosstab_plt(df_crime, 'primary_type', 'community_area', _norm='columns')

# Merge with geographical data - I'll keep non-missing locations only:
df_crime = df_crime[(df_crime.longitude.isnull()==False) & 
	(df_crime.latitude.isnull()==False)]
df_crime['longitude'] = df_crime['longitude'].astype('float')
df_crime['latitude'] = df_crime['latitude'].astype('float')
df_crime['coordinates'] = list(zip(df_crime.longitude, df_crime.latitude))
df_crime['coordinates'] = df_crime['coordinates'].apply(Point)
gdf_crime = geopandas.GeoDataFrame(df_crime, geometry='coordinates')

data = client.get("bt9m-d2mf", limit=1000000)
df_boundaries = pd.DataFrame.from_dict(data)
df_boundaries['the_geom'] = df_boundaries['the_geom'].apply(shape)
gdf_boundaries = geopandas.GeoDataFrame(df_boundaries).set_geometry('the_geom')

df_with_codes = geopandas.sjoin(gdf_crime, gdf_boundaries, how="inner", 
	op='intersects')

# Part 2:
url_acs = "https://api.census.gov/data/2017/acs/acs5?"
what_to_get = ",NAME&for=tract:*&in=state:17&in=county:031"

request_url = url_acs + "get=B03003_002E"+ what_to_get
response = requests.get(request_url)
df_acs = pd.DataFrame(response.json()[1:], columns=response.json()[0])
df_acs['B03003_002E'] = df_acs['B03003_002E'].astype('int32')

def get_var_data(url, variable, what_to_get):
	'''
	Requests the data for a certain variable and converts it to a dataframe.
	Its arguments are the query arguments.
	'''
	request_url = url + "get=" + variable + what_to_get
	response = requests.get(request_url)
	df_temp = pd.DataFrame(response.json()[1:], columns=response.json()[0])
	df_temp.drop(['NAME', 'state', 'county'], axis=1, inplace=True)
	df_temp[variable] = df_temp[variable].astype('int32')
	return df_temp

variables = ["B03003_001E", "B03003_003E", "B02001_001E", "B02001_003E", 
	"B01001_001E", "B01001_002E", "B19001_001E"]
for variable in variables:
	df_temp = get_var_data(url, variable, what_to_get)
	df_acs = df_acs.merge(df_temp, on=['tract'])

df_acs['prop_hisp'] = df_acs.B03003_003E/df_acs.B03003_001E
df_acs['prop_black'] = df_acs.B02001_003E/df_acs.B02001_001E
df_acs['prop_male'] = df_acs.B01001_002E/df_acs.B01001_001E
df_acs['income'] = df_acs.B19001_001E

df_full = df_with_codes.merge(df_acs, left_on='tractce10', right_on='tract')

df_full['battery'] = np.where(df_full['primary_type']=='BATTERY', 1, 0)
df_full['homicide'] = np.where(df_full['primary_type']=='HOMICIDE', 1, 0)
df_full['sex_offense'] = np.where(df_full['primary_type']=='SEX OFFENSE', 1, 0)
df_full['deceptive_practice'] = np.where(df_full['primary_type']==
	'DECEPTIVE PRACTICE', 1, 0)

sub_df = df_full[['battery', 'homicide', 'sex_offense', 'deceptive_practice',
	'prop_hisp', 'prop_black', 'prop_male', 'income']]
sub_df.corr()

# Problem 3:
df_crime['date'] = pd.to_datetime(df_crime['date'])
df_crime['period'] = np.where((df_crime['date'] >= '2017-6-28 00:00:00') & 
	(df_crime['date'] < '2017-7-28 00:00:00'), "pre", None)
df_crime['period'] = np.where((df_crime['date'] >= '2018-6-28 00:00:00') & 
	(df_crime['date'] < '2018-7-28 00:00:00'), "post", df_crime['period'])
cross_alderman = pd.crosstab(df_crime['primary_type'], df_crime['period'])

# Problem 4:
def find_block(address):
	'''
	Finds the data tract id for a particular address using censusgeocode
	'''
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
