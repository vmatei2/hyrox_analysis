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

WORK_2_RUN = "work_to_run_ratio"
RUN_2_TOTAL = "run_to_total"
ROXZONE_2_TOTAL = "roxzone_to_total_ratio"
SLEDPULL_2_BURPEE = "sledpull_to_burpee_ratio"
RUN_1_TO_8 = "run1_to_run8_ratio"
RUN_2_TO_8 = "run2_to_run8_ratio"
SKI_ERG_TO_ROW_ERG = "ski_erg_to_row_ratio"
SLED_PUSH_2_PULL = "sled_push_to_sled_pull_ratio"
FIRST_HALF_TO_SECOND_HALF_RATIO = "first_half_to_second_half_ratio"
AVG_RUN_PACING_CHANGE = "avg_run_pacing_change"
STRENGTH_SCORE = "strength_score"
ENDURANCE_SCORE = "endurance_score"
SKI_ERG_TO_WALL_BALL = "ski_erg_to_wall_ball_ratio"
STRENGTH_TO_ENDURANCE_BALANCE = "strength_to_endurance_balance"

NETWORK_ANALYSIS_METRICS = [WORK_2_RUN, RUN_1_TO_8, SLEDPULL_2_BURPEE, FIRST_HALF_TO_SECOND_HALF_RATIO,
                            SKI_ERG_TO_WALL_BALL]
