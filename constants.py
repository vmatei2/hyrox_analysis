ALL_RACES = "All races"

WORK_LABELS = []
RUN_LABELS = []
ROXZONE_LABELS = []
STATIONS = ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
            'Sandbag Lunges', 'Wall Balls']
for i in range(1, 9):
    WORK_LABELS.append('work_' + str(i))
    RUN_LABELS.append('run_' + str(i))
    ROXZONE_LABELS.append('roxzone_' + str(i))

# options used in user-display and then filtering the dataframe
REQUEST_ALL_VALUES = 'all'
