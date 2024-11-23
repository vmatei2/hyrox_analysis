ALL_RACES = "All races"

WORK_LABELS = []
RUN_LABELS = []
RACE_ORDER_LABELS = []
ROXZONE_LABELS = []
ROXZONE_TIME = "roxzone_time"
RUN_TIME = "run_time"
WORK_TIME = "work_time"
TOTAL_TIME = "total_time"

STATIONS = ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
            'Sandbag Lunges', 'Wall Balls']
for i in range(1, 9):
    WORK_LABELS.append('work_' + str(i))
    RUN_LABELS.append('run_' + str(i))
    ROXZONE_LABELS.append('roxzone_' + str(i))
    RACE_ORDER_LABELS.append('run_' + str(i))
    RACE_ORDER_LABELS.append('work_' + str(i))
# options used in user-display and then filtering the dataframe
REQUEST_ALL_VALUES = 'all'

USER_RUN_1 = "user_run_1"
USER_RUN_2 = "user_run_2"
USER_RUN_3 = "user_run_3"
USER_RUN_4 = "user_run_4"
USER_RUN_5 = "user_run_5"
USER_RUN_6 = "user_run_6"
USER_RUN_7 = "user_run_7"
USER_RUN_8 = "user_run_8"

USER_SKI_ERG = "user_ski_erg"
USER_SLED_PUSH = "user_sled_push"
USER_SLED_PULL = "user_sled_pull"
USER_BURPEE_BROAD_JUMP = "user_burpee_broad_jump"
USER_ROW_ERG = "user_row_erg"
USER_FARMERS_CARRY = "user_farmers_carry"
USER_SANDBAG_LUNGES = "user_sandbag_lunges"
USER_WALL_BALLS = "user_wall_balls"


ALL_USER_INPUTS = [USER_RUN_1, USER_RUN_2, USER_RUN_3, USER_RUN_4, USER_RUN_5, USER_RUN_6, USER_RUN_7, USER_RUN_8,
                   USER_SKI_ERG, USER_SLED_PUSH, USER_SLED_PULL, USER_BURPEE_BROAD_JUMP, USER_ROW_ERG, USER_FARMERS_CARRY, USER_SANDBAG_LUNGES, USER_WALL_BALLS]

