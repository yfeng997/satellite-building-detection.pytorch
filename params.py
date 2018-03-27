import numpy as np

# Project Parameter
dataset = '/home/feng/data/fMoW-rgb'
dtype = np.float32


# Training Parameters
batch_size = 100


# Dataset 
class_names = [
    'false_detection',
    'residential',
    'non_residential',
]

fmow_class_names = [
 'single-unit_residential',
 'multi-unit_residential',
 'lake_or_pond',
 'educational_institution',
 'parking_lot_or_garage',
 'military_facility',
 'runway',
 'port',
 'tower',
 'zoo',
 'aquaculture',
 'barn',
 'border_checkpoint',
 'dam',
 'tunnel_opening',
 'recreational_facility',
 'hospital',
 'police_station',
 'electric_substation',
 'railway_bridge',
 'fire_station',
 'swimming_pool',
 'lighthouse',
 'waste_disposal',
 'airport_hangar',
 'road_bridge',
 'toll_booth',
 'car_dealership',
 'office_building',
 'impoverished_settlement',
 'surface_mine',
 'crop_field',
 'fountain',
 'solar_farm',
 'prison',
 'ground_transportation_station',
 'factory_or_powerplant',
 'wind_farm',
 'storage_tank',
 'golf_course',
 'construction_site',
 'space_facility',
 'airport',
 'place_of_worship',
 'race_track',
 'smokestack',
]








