from requests import request
from bs4 import BeautifulSoup
from enum import Enum
import re
import itertools
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from datetime import timedelta
import numpy as np
import pandas as pd
import os
from concurrent.futures  import ThreadPoolExecutor


## helper classes
class Division(Enum):
    open = "H"
    # pro = "HPRO"
    # doubles = "HD"
    # relay = "HMR"
    # goruck = "HG"
    # goruck_doubles = "HDG"


class Gender(Enum):
    male = "M"
    # female = "W"
    # mixed = "X"


## Helper Functions
def get_html(url: str):
    cookie_retrieval = request("GET", url)
    cookie = cookie_retrieval.request.headers.get("Cookie")
    response = request("GET", url, headers={"Cookie": cookie})
    return response.text


def removeprefix(x: str, prefix: str):
    if x.startswith(prefix):
        return x[len(prefix):]
    return x


class HyroxParticipant:
    def __init__(self, id: str, name: str, division: Division, gender: Gender, age_group: str, time: str, link: str):
        self.id = id
        self.name = name
        self.division = division
        self.gender = gender
        self.age_group = age_group
        self.time = time
        self.link = link
        self._raw_splits = []
        self.splits = {}
        self.ignore = False

    def get_timings(self):
        html = get_html(self.link)
        soup = BeautifulSoup(html, 'html.parser')

        try:
            splits = soup.find(class_="detail-box box-splits")
            split_rows = splits.find("tbody").find_all("tr")
        except AttributeError:
            self.ignore = True
            return

        def get_diff(i: int):
            if i >= len(split_rows):
                return timedelta()

            raw = split_rows[i].select("td.diff")[0].text
            if "–" in raw:
                return timedelta()
            else:
                return timedelta(minutes=int(raw.split(":")[0]), seconds=int(raw.split(":")[1]))

        def get_details():
            skierg_rox_in = self._raw_splits[0]
            skierg_in = self._raw_splits[1]
            skierg_out = self._raw_splits[2]
            skierg_rox_out = self._raw_splits[3]

            push_rox_in = self._raw_splits[4]
            push_in = self._raw_splits[5]
            push_out = self._raw_splits[6]
            push_rox_out = self._raw_splits[7]

            pull_rox_in = self._raw_splits[8]
            pull_in = self._raw_splits[9]
            pull_out = self._raw_splits[10]
            pull_rox_out = self._raw_splits[11]

            burpee_rox_in = self._raw_splits[12]
            burpee_in = self._raw_splits[13]
            burpee_out = self._raw_splits[14]
            burpee_rox_out = self._raw_splits[15]

            row_rox_in = self._raw_splits[16]
            row_in = self._raw_splits[17]
            row_out = self._raw_splits[18]
            row_rox_out = self._raw_splits[19]

            carry_rox_in = self._raw_splits[20]
            carry_in = self._raw_splits[21]
            carry_out = self._raw_splits[22]
            carry_rox_out = self._raw_splits[23]

            lunges_rox_in = self._raw_splits[24]
            lunges_in = self._raw_splits[25]
            lunges_out = self._raw_splits[26]
            lunges_rox_out = self._raw_splits[27]

            wall_in = self._raw_splits[28]
            finish = self._raw_splits[29]

            self.splits["total"] = np.sum(self._raw_splits)
            self.splits["stations"] = [skierg_out, push_out, pull_out, burpee_out, row_out, carry_out, lunges_out,
                                       finish]
            self.splits["work"] = np.sum(self.splits["stations"])
            self.splits["runs"] = [skierg_rox_in, push_rox_in, pull_rox_in, burpee_rox_in, row_rox_in, carry_rox_in,
                                   lunges_rox_in, wall_in]
            self.splits["run"] = np.sum(self.splits["runs"])
            self.splits["rests"] = [skierg_in + skierg_rox_out, push_in + push_rox_out, pull_in + pull_rox_out,
                                    burpee_in + burpee_rox_out, row_in + row_rox_out, carry_in + carry_rox_out,
                                    lunges_in + lunges_rox_out, timedelta()]
            self.splits["roxzone"] = np.sum(self.splits["rests"])

        self._raw_splits = [get_diff(x) for x in range(30)]
        get_details()

    def to_array(self):
        return [
            self.id,
            self.name,
            self.gender.name,
            self.age_group,
            self.division.name,
            str(self.splits["total"]),
            str(self.splits["work"]),
            str(self.splits["roxzone"]),
            str(self.splits["run"]),
            str(self.splits["runs"][0]),
            str(self.splits["stations"][0]),
            str(self.splits["rests"][0]),
            str(self.splits["runs"][1]),
            str(self.splits["stations"][1]),
            str(self.splits["rests"][1]),
            str(self.splits["runs"][2]),
            str(self.splits["stations"][2]),
            str(self.splits["rests"][2]),
            str(self.splits["runs"][3]),
            str(self.splits["stations"][3]),
            str(self.splits["rests"][3]),
            str(self.splits["runs"][4]),
            str(self.splits["stations"][4]),
            str(self.splits["rests"][4]),
            str(self.splits["runs"][5]),
            str(self.splits["stations"][5]),
            str(self.splits["rests"][5]),
            str(self.splits["runs"][6]),
            str(self.splits["stations"][6]),
            str(self.splits["rests"][6]),
            str(self.splits["runs"][7]),
            str(self.splits["stations"][7]),
            str(self.splits["rests"][7]),
        ]


class HyroxEvent:
    def __init__(self, event_id: str, season: int, print_name: str):
        self.event_id = event_id
        self.season = season
        self.event_name = ""
        self.event_participants = dict((x, dict((y, []) for y in Gender)) for x in Division)
        self.num_event_participants = dict((x, dict((y, 0) for y in Gender)) for x in Division)
        self.print_name = print_name  # naming this variable print name more due to being lazy as it's 11PM and have a 6:30 wake-up tomorrow - don't want to interfere with the previous logic of setting event_name empty and filling it in later

    @property
    def participants(self):
        arr = []
        for division, arr2 in self.event_participants.items():
            for gender, arr3 in arr2.items():
                arr.extend(arr3)
        return arr

    def generate_url(self, page: int, division: Division, gender: Gender):
        return f"https://hyrox.r.mikatiming.com/season-{self.season}/?page={page}&event={division.value}_{self.event_id}&num_results=100&pid=list&pidp=ranking_nav&ranking=time_finish_netto&search%5Bsex%5D={gender.value}&search%5Bage_class%5D=%25&search%5Bnation%5D=%25"

    def get_info(self):
        combinations = list(itertools.product(Division, Gender))
        print(f'Retrieving participants for {self.print_name}')
        # for combination in combinations:
        #     self.retrieve_combination(combination)
        with tqdm(total=len(combinations), desc='Retrieving Participants') as pbar:
            def wrapper(combination):
                self.retrieve_combination(combination)
                pbar.update(1)

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(wrapper,combinations)

        thread_map(lambda participant: participant.get_timings(), self.participants, max_workers=10,
                   desc="Retrieving Splits")

        def participant_filter(p: HyroxParticipant):
            if p.ignore:
                return False

            if any(list(map(lambda x: x < timedelta(), p.splits["stations"]))):
                return False

            if any(list(map(lambda x: x < timedelta(), p.splits["runs"]))):
                return False

            if any(list(map(lambda x: x < timedelta(), p.splits["rests"]))):
                return False

            return True

        for division, gender in combinations:
            self.event_participants[division][gender] = filter(participant_filter,
                                                               self.event_participants[division][gender])

    def retrieve_combination(self, combination):
        division, gender = combination
        page = 1
        while True:
            url = self.generate_url(page, division=division, gender=gender)
            html = get_html(url)
            soup = BeautifulSoup(html, 'html.parser')

            h2_title: str = soup.h2.text.strip()
            h2_title = removeprefix(h2_title, "Results: ").split(" / HYROX")[0]
            if h2_title == "General Ranking / All":
                break

            list_headers = soup.select(".list-group-header .list-field")
            if len(list_headers) > 0 and list_headers[0].text.strip() == "Race":
                break

            if self.event_name == "":
                self.event_name = f"S{self.season} {h2_title.strip()}"
                if self.event_name == "S4 WorldChampionship - Leipzig":
                    self.event_name = "S4 2021 Leipzig - World Championship"

            if self.num_event_participants[division][gender] == 0:
                list_info: str = soup.find(class_="list-info").li.text
                num_participants = int(re.findall("(\d+) Results", list_info)[0])
                self.num_event_participants[division][gender] = num_participants

            list_rows = list(soup.find_all("li", class_="list-group-item row"))
            list_rows.extend(list(soup.find_all("li", class_="list-active list-group-item row")))

            for row in list_rows:
                fields = row.select(".list-field")

                id = ""
                name = ""
                age_group = ""
                time = ""
                link = ""

                for field_i in range(len(list_headers)):
                    header = "".join(list_headers[field_i].find_all(string=True, recursive=False))
                    field = fields[field_i]
                    field_content = field.text.strip()
                    if header == "Number":
                        id = removeprefix(field_content, "Number")
                    elif header == "Age Group":
                        age_group = removeprefix(field_content, "Age Group")
                        if age_group == "–":
                            age_group = ""
                    elif header == "Name" or header == "Member":
                        name = field_content
                        link = f"https://hyrox.r.mikatiming.com/season-{self.season}/{field.a.get('href')}"
                    elif header == "Total":
                        time = removeprefix(field_content, "Total")

                participant = HyroxParticipant(
                    id=id,
                    name=name,
                    division=division,
                    gender=gender,
                    age_group=age_group,
                    time=time,
                    link=link
                )
                self.event_participants[division][gender].append(participant)
            if len(self.event_participants[division][gender]) < self.num_event_participants[division][gender]:
                page += 1
                print(f"Finished Fetching page: {page-1} for {division}/{gender}")
                print(f"Len of event participants for above combination is {len(self.event_participants[division][gender])}")
            else:
                break

    def save(self):
        def participant_map(p: HyroxParticipant):
            arr = [self.event_id, self.event_name]
            arr.extend(p.to_array())
            return np.array(arr)

        df = pd.DataFrame(
            np.array(list(map(participant_map, self.participants))),
            columns=["event_id", "event_name", "id", "name", "gender", "age_group", "division", "total_time",
                     "work_time", "roxzone_time", "run_time", "run_1", "work_1", "roxzone_1", "run_2", "work_2",
                     "roxzone_2", "run_3", "work_3", "roxzone_3", "run_4", "work_4", "roxzone_4", "run_5", "work_5",
                     "roxzone_5", "run_6", "work_6", "roxzone_6", "run_7", "work_7", "roxzone_7", "run_8", "work_8",
                     "roxzone_8"]
        )

        df.insert(5, "nationality", df["name"].str.extract(r'\(([A-Z]{3})\)', expand=False))
        df.drop(["id"], axis=1, inplace=True)
        directory = os.path.dirname(__file__)
        hyroxDirectory = directory + "/assets/hyroxData"
        df.to_csv(os.path.join(hyroxDirectory, f"{self.event_name}.csv"), index=False)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_event = HyroxEvent(
            event_id=self.event_id,
            season=self.season
        )
        new_event.event_name = self.event_name
        new_event.event_participants = self.event_participants
        new_event.num_event_participants = self.num_event_participants
        return new_event


def save_events(events):
    for event in events:
        # try catch in case issue with any event, make sure we are still saving all data for events without problems
        try:
            event.get_info()
            event.save()
            print('have saved event ', event.print_name)
        except Exception as e:
            print(f"have caught exception {e} when storing down event {event.print_name} ")


s5_manchester2023 = HyroxEvent(event_id="JGDMS4JI425", season=5, print_name="Manchester2023")
s5_euroChamps2023 = HyroxEvent(event_id="JGDMS4JI411", season=5, print_name="EuroChamps2023")


## 2024 DATA
s6_london_excel = HyroxEvent(event_id="JGDMS4JI62E", season=6, print_name="london_excel")
london2024 = HyroxEvent(event_id="JGDMS4JI7AA", season=6, print_name="london2024")
gdansk2024 = HyroxEvent(event_id="JGDMS4JI7FB", season=6, print_name="gdansk2024")
rimini2024 = HyroxEvent(event_id="JGDMS4JI80E", season=6, print_name="rimini2024")
newYork2024 = HyroxEvent(event_id="JGDMS4JI7E7", season=6, print_name="newYork2024")
birminghamS7 = HyroxEvent(event_id="UKBOveralll", season=7, print_name="birminghamOct2024")

dublinS7 = HyroxEvent(event_id="IEDOverall", season=7, print_name="dublinNov2024")
rotterdamS6 = HyroxEvent(event_id="JGDMS4JI747", season=6, print_name="rotterdamApr2024")

manchester_2024 = HyroxEvent(event_id="JGDMS4JI8B0", season=7, print_name="Manchester 2024")

if __name__ == '__main__':
    subset = [manchester_2024]
    save_events(subset)
