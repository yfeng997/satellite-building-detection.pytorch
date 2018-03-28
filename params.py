import torch
import numpy as np


# Project Parameter
# dataset = '/home/feng/data/fMoW-rgb'
dataset = '/Users/Feng/Desktop/CV/data/fmow'
num_threads = 1
dtype = torch.FloatTensor


# Training Parameters
batch_size = 40
num_epochs = 50
learning_rate = 1e-4

print_every = 10
# Dataset 
rbc_class_names = [
    'false_detection',
    'residential',
    'non_residential',
]

fmow_class_names_mini = [
# 'false_detection',
# 'airport', # TOO BIG
'airport_hangar',
# 'airport_terminal',
# 'amusement_park',
# 'aquaculture',
# 'archaeological_site',
'barn',
# 'border_checkpoint',
# 'burial_site',
# 'car_dealership',
# 'construction_site',
# 'crop_field',
# 'dam',
# 'debris_or_rubble',
'educational_institution',
# 'electric_substation',
'factory_or_powerplant',
# 'fire_station',
# 'flooded_road',
# 'fountain',
# 'gas_station',
# 'golf_course',
# 'ground_transportation_station',
# 'helipad',
'hospital',
# 'interchange',
# 'lake_or_pond',
'lighthouse',
# 'military_facility',
'multi-unit_residential',
# 'nuclear_powerplant',
'office_building',
# 'oil_or_gas_facility',
# 'park',
'parking_lot_or_garage',
'place_of_worship',
'police_station',
# 'port', # TOO BIG
'prison',
# 'race_track',
# 'railway_bridge',
'recreational_facility',
# 'impoverished_settlement',
# 'road_bridge',
# 'runway',
# 'shipyard',
# 'shopping_mall',
'single-unit_residential',
# 'smokestack',
# 'solar_farm',
'space_facility',
# 'stadium',
# 'storage_tank',
# 'surface_mine',
# 'swimming_pool',
# 'toll_booth',
# 'tower',
# 'tunnel_opening',
# 'waste_disposal',
# 'water_treatment_facility',
# 'wind_farm',
# 'zoo',
]

fmow_class_names = [
'false_detection',
'airport', # TOO BIG
'airport_hangar',
'airport_terminal',
'amusement_park',
'aquaculture',
'archaeological_site',
'barn',
'border_checkpoint',
'burial_site',
'car_dealership',
'construction_site',
'crop_field',
'dam',
'debris_or_rubble',
'educational_institution',
'electric_substation',
'factory_or_powerplant',
'fire_station',
'flooded_road',
'fountain',
'gas_station',
'golf_course',
'ground_transportation_station',
'helipad',
'hospital',
'interchange',
'lake_or_pond',
'lighthouse',
'military_facility',
'multi-unit_residential',
'nuclear_powerplant',
'office_building',
'oil_or_gas_facility',
'park',
'parking_lot_or_garage',
'place_of_worship',
'police_station',
'port', # TOO BIG
'prison',
'race_track',
'railway_bridge',
'recreational_facility',
'impoverished_settlement',
'road_bridge',
'runway',
'shipyard',
'shopping_mall',
'single-unit_residential',
'smokestack',
'solar_farm',
'space_facility',
'stadium',
'storage_tank',
'surface_mine',
'swimming_pool',
'toll_booth',
'tower',
'tunnel_opening',
'waste_disposal',
'water_treatment_facility',
'wind_farm',
'zoo',
]




