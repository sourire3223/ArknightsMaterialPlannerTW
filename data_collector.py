import datetime
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from os.path import isfile
from threading import Thread

import numpy as np
import pandas as pd
import requests
from loguru import logger

# useful urls from penguin-statistics (penguin-stats) and arknights-toolbox (arktools)
URL_ITEM = "https://penguin-stats.io/PenguinStats/api/v2/items"
URL_STAGE = "https://penguin-stats.io/PenguinStats/api/v2/stages"
URL_MATRIX = "https://penguin-stats.io/PenguinStats/api/v2/result/matrix?show_closed_zone=true"
URL_MATRIX_PRIVATE = "https://penguin-stats.io/PenguinStats/api/v2/_private/result/matrix/CN/global"
URL_FORMULA = "https://penguin-stats.io/PenguinStats/api/v2/formula"
URL_EVENT = "https://penguin-stats.io/PenguinStats/api/v2/period"


URL_OPERATOR = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/character.json"
# URL_OPERATOR_ORDER = "https://prts.wiki/w/%E6%A8%A1%E5%9D%97:GetCharCnOnlineTime"
URL_OPERATOR_ORDER = "https://prts.wiki/w/%E5%B9%B2%E5%91%98%E4%B8%8A%E7%BA%BF%E6%97%B6%E9%97%B4%E4%B8%80%E8%A7%88"
URL_SKILL = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/skill.json"
URL_UNIEQUIP = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/uniequip.json"
URL_CULTIVATE = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/data/cultivate.json"
URL_ZONE = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/zone.json"
URL_PRICE = "https://raw.githubusercontent.com/penguin-statistics/ArkPlanner/master/price.txt"

__UPGRADE_ITEM_VALUE = {"基础作战记录": 200 * 0.0035,
                        "初级作战记录": 400 * 0.0035,
                        "中级作战记录": 1000 * 0.0035,
                        "高级作战记录": 2000 * 0.0035,
                        "赤金": 400 * 0.0035,
                        "龙门币": 0.0035}


# TODO (paulpork): 合成用什麼賺

def load_when_call(func):
    """deocorator: load when call first
    args:
        func: function to decorate
    """
    def wrapper(cls):
        name = func.__name__
        # if not hasattr(cls, "_" + name):
        if getattr(cls, f"_{name}") == None:
            if not isfile(f"./data/{name}.json"):
                wrapper_logger = logger.patch(
                    lambda r: r.update(function=name))
                wrapper_logger.warning(f"{name}.json is not found.")
                # cls._name = cls._save_name()
                _ = getattr(cls, "_save_" + name)()
            else:
                with open(f"./data/{name}.json", 'r', encoding="utf-8-sig") as fr:
                    # cls._name = json.loads(fr.read())
                    setattr(cls, "_" + name, json.loads(fr.read()))

        # return cls._name
        return getattr(cls, f"_{name}")

    return wrapper


@dataclass
class DataCollector:
    _item_map: None | dict = None
    _stage_map: None | dict = None
    _operator_map: None | dict = None
    _skill_map: None | dict = None
    _uniequip_map: None | dict = None
    _cultivate_info: None | dict = None
    _event_list: None | dict = None
    _zone_map: None | dict = None
    _zone_matrix: None | dict = None
    _price: None | dict = None
    _formula18: None | dict = None
    _formula20: None | dict = None
    _formulaBlemishine: None | dict = None

    _operator_order: None | dict = None
    _LAST_OPERATOR_TIME: None | int = None

    _target_items: None | dict = None
    _level: None | dict = None

    # def __init__(self, update: bool = False):
    #     self.item_map = self.load_item_map(update)
    #     self.stage_map = self.load_stage_map(update)
    #     self.operator_map = self.load_operator_map(update)
    #     self.skill_map = self.load_skill_map(update)
    #     self.uniequip_map = self.load_uniequip_map(update)
    #     self.cultivate_info = self.load_cultivate_info(update)
    #     self.event_list = self.load_event_list(update)
    #     self.zone_map = self.load_zone_map(update)
    #     self.zone_matrix = self.load_zone_matrix(update)
    #     self.price = self.load_price(update)

    #     self.operator_order = self.load_operator_order(update)
    #     self.LAST_OPERATOR_TIME = self.operator_order.pop('LAST_OPERATOR_TIME')
    @classmethod
    def update_all(cls) -> None:
        thread_list = list()
        thread_list.append(Thread(target=cls._save_item_map))
        thread_list.append(Thread(target=cls._save_stage_map))
        thread_list.append(Thread(target=cls._save_operator_map))
        thread_list.append(Thread(target=cls._save_skill_map))
        thread_list.append(Thread(target=cls._save_uniequip_map))
        thread_list.append(Thread(target=cls._save_event_list))
        thread_list.append(Thread(target=cls._save_zone_map))
        thread_list.append(Thread(target=cls._save_price))
        thread_list.append(Thread(target=cls._save_operator_order))
        thread_list.append(Thread(target=cls._save_formula))
        # cls._LAST_OPERATOR_TIME = cls._operator_order.pop('LAST_OPERATOR_TIME')
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()  # wait for all threads to finish

        # dependency update
        # dependency: item_map, operator_map, skill_map, uniequip_map
        t1 = Thread(target=cls._save_cultivate_info)
        # dependency: item_map, stage_map, zone_map
        t2 = Thread(target=cls._save_zone_matrix)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    @classmethod
    @property
    @load_when_call
    def item_map(cls) -> dict[str, str]:
        pass

    @classmethod
    @property
    @load_when_call
    def stage_map(cls) -> dict[str, list[str]]:
        pass

    @classmethod
    @property
    @load_when_call
    def operator_map(cls) -> dict[str, dict[str, str]]:
        pass

    @classmethod
    @property
    def operator_order(cls) -> dict[str, int]:
        if cls._operator_order == None:
            if not isfile("./data/operator_order.json"):
                logger.warning(f"operator_order.json is not found.")
                _ = cls._save_operator_order()
            else:
                with open("./data/operator_order.json", 'r', encoding="utf-8-sig") as fr:
                    cls._operator_order = json.loads(fr.read())

            cls._LAST_OPERATOR_TIME = cls._operator_order.pop(
                'LAST_OPERATOR_TIME')

        return cls._operator_order

    @classmethod
    @property
    def LAST_OPERATOR_TIME(cls):
        if cls._LAST_OPERATOR_TIME == None:
            _ = cls.operator_order
        return cls._LAST_OPERATOR_TIME

    @classmethod
    @property
    def LAST_OPERATOR_TIME_TAIPEI(cls):
        return pd.Timestamp(cls.LAST_OPERATOR_TIME, unit='s').strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    @property
    @load_when_call
    def skill_map(cls) -> dict[str, str]:
        pass

    @classmethod
    @property
    @load_when_call
    def uniequip_map(cls) -> dict[str, str]:
        pass

    @classmethod
    @property
    @load_when_call
    def cultivate_info(cls) -> dict[str, ]:
        pass

    @classmethod
    @property
    @load_when_call
    def event_list(cls) -> dict[str, list[int]]:
        pass

    @classmethod
    @property
    @load_when_call
    def zone_map(cls) -> dict[str, str]:
        pass

    @classmethod
    @property
    @load_when_call
    def zone_matrix(cls) -> dict[str, dict[str, ]]:
        pass

    @classmethod
    @property
    @load_when_call
    def price(cls) -> dict[str, int]:
        pass

    @classmethod
    @property
    def formula18(cls) -> dict[str, dict[str, float]]:
        if cls._formula18 == None:
            if not isfile("./data/formula18.json"):
                logger.warning(
                    f"formula18.json is not found.")
                cls._save_formula()  # no return
            else:
                with open("./data/formula18.json", 'r', encoding="utf-8-sig") as fr:
                    cls._formula18 = json.loads(fr.read())
        return cls._formula18

    @classmethod
    @property
    def formula20(cls) -> dict[str, dict[str, float]]:
        if cls._formula20 == None:
            if not isfile("./data/formula20.json"):
                logger.warning(
                    f"formula20.json is not found.")
                cls._save_formula()  # no return
            else:
                with open("./data/formula20.json", 'r', encoding="utf-8-sig") as fr:
                    cls._formula20 = json.loads(fr.read())
        return cls._formula20

    @classmethod
    @property
    def formulaBlemishine(cls) -> dict[str, dict[str, float]]:
        if cls._formulaBlemishine == None:
            if not isfile("./data/formulaBlemishine.json"):
                logger.warning(
                    f"formulaBlemishine.json is not found.")
                cls._save_formula()  # no return
            else:
                with open("./data/formulaBlemishine.json", 'r', encoding="utf-8-sig") as fr:
                    cls._formulaBlemishine = json.loads(fr.read())
        return cls._formulaBlemishine

    @classmethod
    @property
    def formulaOPT(cls) -> dict[str, dict[str, float]]:
        pass
        # TODO: formula20 + formulaBlemishine
        # if cls._formulaOPT == None:
        #     if not isfile("./data/formulaOPT.json"):
        #         logger.warning(
        #             f"formulaOPT.json is not found.")
        #         cls._save_formula()

    @classmethod
    @property
    def target_items(cls):
        if cls._target_items == None:
            cls._target_items = {k: v for k, v in cls.item_map.items() if (
                v[-1] in "12345" and v[0].isdigit() and len(v) == 5) or "作战记录" in k or k == "赤金"}
            cls._target_items |= {"龙门币": "4001"}
        return cls._target_items

    @classmethod
    @property
    def level(cls):
        if cls._level == None:
            cls._level = {k: v[-1] for k, v in cls.item_map.items() if (
                v[-1] in "12345" and v[0].isdigit() and len(v) == 5)}
        return cls._level

    @classmethod
    def get_df_matrix(cls, formula: str | dict[str, dict[str, float]]) -> pd.DataFrame:
        match formula:
            case "18" | "fomula18":
                formula = cls.formula18
            case "20" | "fomula20":
                formula = cls.formula20
            case "Blemishine" | "fomulaBlemishine":
                formula = cls.formulaBlemishine
            case dict():
                pass
            case _:
                raise ValueError(
                    f"formula must be '18', '20', 'Blemishine' or 'formula18', 'formula20', 'formulaBlemishine'or a dict, but {formula}"
                )

        df = pd.DataFrame(columns=cls.target_items.keys())
        # from zone_matrix, TODO: check correctness
        min_times = 1000  # delta = 0.025, prob = 0.95 (single tail)
        new_dict = dict()
        dc.zone_matrix.keys()
        for zone, stage_matrix in dc.zone_matrix.items():
            for stage, matrix in stage_matrix.items():
                if "・复刻" in zone:
                    stage += "・复刻"
                elif "・永久" in zone:
                    stage += "・永久"

                stage_dict = {item: info["drop_rate"] for item, info in matrix["drop_info"].items(
                ) if item in dc.target_items and info["times"] > min_times}
                if stage_dict:
                    new_dict[stage] = stage_dict
        # from formula, TODO: check correctness
        new_dict = dict()
        for k, v in dc.formula18.items():
            new_dict[k] = v | {k: 1}
        # from __UPGRADE_ITEM_VALUE
        # TODO 0812: 建df方式
        # TODO 0813: not yet done, check correctness
        return df
    # TODO: get sanity df & @property sanity

    def generate_future_activity_materials(self,
                                           current_event_name: str = "玛莉娅・临光・复刻",
                                           sanity_per_day: int = 310,
                                           discount_rate_per_week: float = 0.95) -> dict[str, float]:  # farm

        # TODO: 算cp值 -> 刷 x 體在最多的那關，得到素材（期望值） -> 算cp值 -> 刷 x 體在最多的那關，得到素材（期望值）-> ... -> 沒體力
        # TODO: 重設 linprog 參數計算
        # TODO: 算cp值 =  MaterialPlanner.value + self.zone_matrix
        pass

    def _generate_future_activity_materials_default(self,
                                                    current_event_name: str = "玛莉娅・临光・复刻",
                                                    sanity_per_day: int = 310,
                                                    discount_rate_per_week: float = 0.95) -> dict[str, float]:  # farm

        # event_list -> stage_matrix
        current_time = self.event_list[current_event_name][0]
        BLUE_MATERIALS = {"全新装置", "RMA70-12", "研磨石", "凝胶", "炽合金", "酮凝集组", "轻锰矿", "异铁组", "扭转醇",
                          "聚酸酯组", "糖组", "固源岩组", "晶体元件", "化合切削液", "半自然溶剂"}

        # activity farm info in penguin-stats
        activity_record = dict()
        power = 1.2
        for zone, time in self.event_list.items():
            if zone in self.zone_matrix and time[0] > current_time:
                activity_record[zone] = {"start_date": time[0], "duration": "{:.1f} days".format(
                    (time[1] - time[0]) / 86400), "stage": dict()}
                if "mini" in self.zone_map[zone]:
                    continue

                # sanity_list
                sanity_list = [list(v["drop_info"].values())[0]["times"] * v["sanity"]
                               for v in self.zone_matrix[zone].values() if v["drop_info"]]

                threshold = max(0.2 * np.mean(sanity_list), 0)
                sanity_list = [t if t > threshold else 0 for t in sanity_list]
                sanity_list_ = np.array(sanity_list) ** power

                total = sanity_list_.sum()

                for k, v in self.zone_matrix[zone].items():
                    if v["drop_info"]:
                        # total snaity cost in penguin-stats
                        sanity_cost = list(v["drop_info"].values())[
                            0]["times"] * v["sanity"]
                        if sanity_cost > threshold:
                            ratio = (sanity_cost**power / total)
                            total_sanity = (sanity_per_day *
                                            (time[1] - time[0]) / 86400)
                            times = ratio * total_sanity / v["sanity"]

                            activity_record[zone]["stage"][k] = {
                                "times": times,
                                "drop": {item: times*content["quantity"]/content["times"]
                                         for item, content in v["drop_info"].items() if item in BLUE_MATERIALS},
                            }

        # material got in activities
        activity_materials = dict()
        decay_rate_per_week = 0.95
        for k, v in activity_record.items():
            week = (v["start_date"] - current_time) / 86400 / 7
            discount = decay_rate_per_week ** week
            for stage, info in v["stage"].items():
                for item, quantity in info["drop"].items():
                    if item in activity_materials:
                        activity_materials[item] += quantity*discount
                    else:
                        activity_materials[item] = quantity*discount

        self.activity_record = activity_record
        self.activity_materials = activity_materials
        return activity_materials

    # check list of operators elite (2)/ skill (6+3) / uniequip (3)
    def generate_required_materials(self, eku_map=dict(), url="", path=None):
        # TODO: calculate precise required materials
        pass

    def _generate_required_materials_default(self, discount_rate_per_week=0.98):
        required_materials = dict()

        not_in_cultivate_info = list()
        for operator, online_time in self.operator_order.items():
            week = (self.LAST_OPERATOR_TIME - online_time) / 86400 / 7
            discount = discount_rate_per_week ** week

            if operator not in self.cultivate_info:
                not_in_cultivate_info.append(operator)
                continue

            info = self.cultivate_info[operator]

            for items in info["evolve"]:  # 精1 ~ 2
                for k, v in items.items():
                    items[k] = v * discount
                required_materials = DataCollector.add(
                    required_materials, items)

            for items in info["skills"]["normal"]:  # 技1 ~ 7
                for k, v in items.items():
                    items[k] = v * discount
                required_materials = DataCollector.add(
                    required_materials, items)

            for skill in info["skills"]["elite"]:  # 一 ~ 三技
                for items in skill["cost"]:  # 專1 ~ 3
                    for k, v in items.items():
                        items[k] = v * discount
                    required_materials = DataCollector.add(
                        required_materials, items)

            for xy in info["uniequip"]:  # 模組 X,Y
                for items in xy["cost"]:  # 1 ~ 3級
                    for k, v in items.items():
                        items[k] = v * discount
                    required_materials = DataCollector.add(
                        required_materials, items)
        print(", ".join(not_in_cultivate_info), "are not in cultivate_info.")

        self.required_materials = required_materials
        return required_materials

    def calculate_zone_value(self, values):
        if isinstance(values, list):
            values = {items["name"]: items["value"]
                      for d in values for items in d["items"]}

        values |= __UPGRADE_ITEM_VALUE

        zone_value = dict()
        for zone, stages in self.zone_matrix.items():
            zone_value[zone] = dict()
            for stage_code, stage_info in stages.items():
                equivalent_sanity = 0
                for item, item_info in stage_info["drop_info"].items():
                    if item not in values:
                        # print(item, item_info)
                        continue
                    equivalent_sanity += float(values[item]) * \
                        item_info["quantity"] / item_info["times"]
                equivalent_sanity += stage_info["sanity"] * 12 * 0.0035
                cp = equivalent_sanity / \
                    stage_info["sanity"] if stage_info["sanity"] > 0 else -1
                zone_value[zone][stage_code] = (equivalent_sanity, cp)

        self.zone_value = zone_value
        self.values = values

        return zone_value

    @property
    def zone_value_filtered(self, cp=0.99):
        {zone: {stage: value for stage, value in stages.items(
        ) if value[1] > cp} for zone, stages in self.zone_value.items()}
    # item_map

    @classmethod
    def _save_item_map(cls):
        try:
            logger.info(f"Download item.json from {URL_ITEM}.")
            with requests.get(URL_ITEM) as response:
                txt = json.loads(response.text)

            item_map = dict()  # item_id2item_name | item_name2item_id
            for item in txt:
                item_id = item["itemId"].strip()
                item_name = item["name"].strip()

                item_map[item_id] = item_name
                item_map[item_name] = item_id

            # raw data
            with open("./data/raw/item.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # item_map
            with open("./data/item_map.json", 'w', encoding="utf-8") as fw:
                json.dump(item_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"item.json and item_map.json are saved.")

            cls._item_map = item_map
            return item_map
        except Exception as e:
            logger.error("Fail to update item_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_stage_map(cls):
        # TODO: update stage_map add 永久、复刻 and check dependency `zone_matrix`
        try:
            logger.info(f"Download stage.json from {URL_STAGE}.")
            with requests.get(URL_STAGE) as response:
                txt = json.loads(response.text)

            # stage_id2stage_name_list | stage_name2stage_ids_list, actually stage_code
            stage_map = dict()

            for stage in txt:
                stage_id = stage["stageId"].strip()
                stage_code = stage["code"].strip()

                stage_map[stage_id] = [stage_code]
                if stage_code in stage_map:
                    stage_map[stage_code].append(stage_id)
                else:
                    stage_map[stage_code] = [stage_id]

            # raw data
            with open("./data/raw/stage.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # stage_map
            with open("./data/stage_map.json", 'w', encoding="utf-8") as fw:
                json.dump(stage_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"stage.json and stage_map.json are saved.")

            cls._stage_map = stage_map
            return stage_map
        except Exception as e:
            logger.error("Fail to update stage_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_operator_map(cls) -> dict[str, str]:
        try:
            logger.info(f"Download operator.json from {URL_OPERATOR}.")
            with requests.get(URL_OPERATOR) as response:
                txt = json.loads(response.text)

            operator_map = dict()  # operator_id2operator_name | operator_name2operator_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                operator_map[_id] = name
                operator_map[name] = _id

            # raw data
            with open("./data/raw/operator.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # operator_map
            with open("./data/operator_map.json", 'w', encoding="utf-8") as fw:
                json.dump(operator_map, fw, indent=2, ensure_ascii=False)
            logger.success("operator.json and operator_map.json are saved.")

            cls._operator_map = operator_map
            return operator_map
        except Exception as e:
            logger.error("Fail to update operator_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_operator_order(cls) -> dict[str, str]:
        try:
            logger.info(f"Download operator_order.json from {URL_OPERATOR}.")
            dfs = pd.read_html(URL_OPERATOR_ORDER)
            operator_order = {row.干员: int(pd.to_datetime(row.国服上线时间, format='%Y年%m月%d日 %H:%M').timestamp())
                              for i, row in dfs[0].iterrows()}
            operator_order["LAST_OPERATOR_TIME"] = max(operator_order.values())

            # operator_order
            with open("./data/operator_order.json", 'w', encoding="utf-8") as fw:
                json.dump(operator_order, fw, indent=2, ensure_ascii=False)
            logger.success("operator_order.json is saved.")

            cls._operator_order = operator_order
            cls._LAST_OPERATOR_TIME = cls._operator_order.pop(
                'LAST_OPERATOR_TIME')

            return operator_order
        except Exception as e:
            logger.error("Fail to update operator_order.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_skill_map(cls) -> dict[str, str]:
        try:
            logger.info(f"Download skill.json from {URL_SKILL}.")
            with requests.get(URL_SKILL) as response:
                txt = json.loads(response.text)

            skill_map = dict()  # skill_id2skill_name | skill_name2skill_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                skill_map[_id] = name
                skill_map[name] = _id

            # raw data
            with open("./data/raw/skill.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # skill_map
            with open("./data/skill_map.json", 'w', encoding="utf-8") as fw:
                json.dump(skill_map, fw, indent=2, ensure_ascii=False)
            logger.success(f"skill.json and skill_map.json are saved.")

            cls._skill_map = skill_map
            return skill_map
        except Exception as e:
            logger.error("Fail to update skill_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_uniequip_map(cls) -> dict[str, str]:
        try:
            logger.info(f"Download uniequip.json from {URL_UNIEQUIP}.")
            with requests.get(URL_UNIEQUIP) as response:
                txt = json.loads(response.text)

            uniequip_map = dict()  # uniequip_id2uniequip_name| uniequip_name2uniequip_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                uniequip_map[_id] = name
                uniequip_map[name] = _id

            # raw data
            with open("./data/raw/uniequip.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # uniequip_map
            with open("./data/uniequip_map.json", 'w', encoding="utf-8") as fw:
                json.dump(uniequip_map, fw, indent=2, ensure_ascii=False)
            logger.success("uniequip.json and uniequip_map.json are saved.")

            cls._uniequip_map = uniequip_map
            return uniequip_map
        except Exception as e:
            logger.error("Fail to update uniequip_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_cultivate_info(cls) -> dict[str, str]:
        """
        dependency: item_map, operator_map, skill_map, uniequip_map
        """
        try:
            logger.info(f"Download cultivate.json from {URL_CULTIVATE}.")
            with requests.get(URL_CULTIVATE) as response:
                txt = json.loads(response.text)

            item_map = cls.item_map
            operator_map = cls.operator_map
            skill_map = cls.skill_map
            uniequip_map = cls.uniequip_map
            cultivate_info = {operator_map[_id]: {
                "evolve": [{item_map[item]: count for item, count in items.items()} for items in v["evolve"]],
                "skills": {
                    "normal": [{item_map[item]: count for item, count in items.items()} for items in v["skills"]["normal"]],
                    "elite": [{
                        "name": skill_map[skill["name"]],
                        "cost": [{item_map[item]: count for item, count in items.items()} for items in skill["cost"]]
                    } for skill in v["skills"]["elite"]]},
                "uniequip": [{
                    "id": uniequip_map[uniequip["id"]],
                    "cost": [{item_map[item]: count for item, count in items.items()} for items in uniequip["cost"]]
                } for uniequip in v["uniequip"]]
            } for _id, v in txt.items()}

            # raw data
            with open("./data/raw/cultivate.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # caltivate_info
            with open("./data/cultivate_info.json", 'w', encoding="utf-8") as fw:
                json.dump(cultivate_info, fw, indent=2, ensure_ascii=False)
            logger.success("cultivate.json and cultivate_info.json are saved.")

            cls._cultivate_info = cultivate_info
            return cultivate_info
        except Exception as e:
            logger.error("Fail to update cultivate_info.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_event_list(cls) -> list:
        try:
            logger.info(f"Download event.json from {URL_EVENT}.")
            with requests.get(URL_EVENT) as response:
                txt = json.loads(response.text)

            event_list = dict()
            event_list = {event["label_i18n"]["zh"].replace("·", "・"): [event["start"] // 1000, event["end"] // 1000 if event["end"] else None]
                          for event in txt if event["existence"]["CN"]["exist"]}

            # raw data
            with open("./data/raw/event.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # event_list
            with open("./data/event_list.json", 'w', encoding="utf-8") as fw:
                json.dump(event_list, fw, indent=2, ensure_ascii=False)
            logger.success("event.json and event_list.json are saved.")

            cls._event_list = event_list
            return event_list
        except Exception as e:
            logger.error("Fail to update event_list.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_zone_map(cls) -> dict[str, str]:
        try:
            logger.info(f"Download zone.json from {URL_ZONE}.")
            with requests.get(URL_ZONE) as response:
                txt = json.loads(response.text)

            zone_map = dict()  # zone_id2zone_name | zone_name2zone_ids
            for _id, name in txt.items():
                if "@" in name:
                    _id = "_".join(_id.split("_")[:3])
                    name = name.replace("@", "・永久@").split("@")[0].strip()
                elif "sre" in _id and "复刻" not in name:
                    _id = _id.strip()
                    name = name.strip().replace("·", "・") + "・复刻"
                else:
                    _id = _id.strip()
                    name = name.strip().replace("·", "・")

                zone_map[_id] = name
                zone_map[name] = _id

            # raw data
            with open("./data/raw/zone.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # zone_map
            with open("./data/zone_map.json", 'w', encoding="utf-8") as fw:
                json.dump(zone_map, fw, indent=2, ensure_ascii=False)
            logger.success(f"zone.json and zone_map.json are saved.")

            cls._zone_map = zone_map
            return zone_map
        except Exception as e:
            logger.error("Fail to update zone_map.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_zone_matrix(cls) -> dict[str, list[str]]:
        """
        dependency: item_map, stage_map, zone_map
        {
            <ZONE_ID>: {
                <STAGE_ID>: {
                    "stage_type": <STAGE_TYPE>,
                    "santity": <SANTITY>,
                    "drop_info": {
                        <ITEM_NAME>: {"times": <TIMES>, "quantity": <QUANTITY>, "std_dev": <STD_DEV>},
                        ...,
                        <ITEM_NAME>: {"times": <TIMES>, "quantity": <QUANTITY>, "std_dev": <STD_DEV>}
                    }
                }
            }
        }
        """
        try:
            item_map = cls.item_map
            stage_map = cls.stage_map
            zone_map = cls.zone_map

            zone_matrix = dict()

            # MATRIX
            logger.info(f"Download matrix.json from {URL_MATRIX_PRIVATE}.")
            with requests.get(URL_MATRIX_PRIVATE) as response:
                txt = json.loads(response.text)
            txt = txt["matrix"]

            stage_id2matrix = dict()
            for t in txt:
                stage_id = t["stageId"]
                item_name = item_map[t["itemId"]]
                times = t["times"]
                quantity = t["quantity"]
                std_dev = t["stdDev"]
                drop_rate = quantity / times  # actually expcted value of drops

                if stage_id not in stage_id2matrix:
                    stage_id2matrix[stage_id] = dict()
                stage_id2matrix[stage_id][item_name] = {
                    "drop_rate": drop_rate, "times": times, "quantity": quantity, "std_dev": std_dev}

            # STAGE
            with requests.get(URL_STAGE) as response:
                txt = json.loads(response.text)

            for stage in txt:
                stage_id = stage["stageId"]
                stage_name = stage_map[stage_id][0]
                zone_id = stage["zoneId"].replace("_zone1", "")
                zone_name = zone_map[zone_id] if zone_id in zone_map else stage["code"].split(
                    "-")[0]
                stage_type = stage["stageType"]
                sanity = stage["apCost"]

                if zone_name not in zone_matrix:
                    zone_matrix[zone_name] = dict()
                zone_matrix[zone_name][stage_name] = {
                    "stage_type": stage_type,
                    "sanity": sanity,
                    "drop_info": stage_id2matrix[stage_id] if stage_id in stage_id2matrix else dict()
                }

            # raw data
            with open("./data/raw/matrix.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # zone_matrix
            with open("./data/zone_matrix.json", 'w', encoding="utf-8") as fw:
                json.dump(zone_matrix, fw, indent=2, ensure_ascii=False)
            logger.success(f"matrix.json and zone_matrix.json are saved.")

            cls._zone_matrix = zone_matrix
            return zone_matrix
        except Exception as e:
            logger.error("Fail to update stage_matrix.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_price(cls) -> dict:
        try:
            logger.info(f"Download price.txt from {URL_PRICE}.")
            with requests.get(URL_PRICE) as response:
                txt = response.text

            price = dict(map(lambda x: (x[0], int(x[1])), [
                t.strip().split() for t in txt.strip().split("\n")]))

            # raw data
            with open("./data/raw/price.txt", 'w', encoding="utf-8") as fw:
                fw.write(txt)

            # price.json
            with open("./data/price.json", 'w', encoding="utf-8") as fw:
                json.dump(price, fw, indent=2, ensure_ascii=False)
            logger.success(f"price.txt and price.json are saved.")

            cls._price = price
            return price
        except Exception as e:
            logger.error("Fail to update price.")
            logger.exception(e)
            return dict()

    @classmethod
    def _save_formula(cls) -> dict:
        try:
            logger.info(f"Download formula.json from {URL_FORMULA}.")
            with requests.get(URL_FORMULA) as response:
                txt = json.loads(response.text)

            formula18 = {
                comp_form["name"]: DataCollector.add({"龙门币": -comp_form["goldCost"]} | {item["name"]: -item["count"] for item in comp_form["costs"]},
                                                     {item["name"]: 0.18 * item["count"] * item["weight"] / comp_form["totalWeight"] for item in comp_form["extraOutcome"]})
                for comp_form in txt}
            formula20 = {
                comp_form["name"]: DataCollector.add({"龙门币": -comp_form["goldCost"]} | {item["name"]: -item["count"] for item in comp_form["costs"]},
                                                     {item["name"]: 0.2 * item["count"] * item["weight"] / comp_form["totalWeight"] for item in comp_form["extraOutcome"]})
                for comp_form in txt}
            formulaBlemishine = {
                comp_form["name"]: DataCollector.add({"龙门币": -comp_form["goldCost"] / 2} | {item["name"]: -item["count"] for item in comp_form["costs"]},
                                                     {item["name"]: 0.14 * item["count"] * item["weight"] / comp_form["totalWeight"] for item in comp_form["extraOutcome"]})
                for comp_form in txt}
            # raw data
            with open("./data/raw/formula.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)

            # formula18.json & formula20.json & formulaBlemishine.json
            with open("./data/formula18.json", 'w', encoding="utf-8") as fw:
                json.dump(formula18, fw, indent=2, ensure_ascii=False)
            with open("./data/formula20.json", 'w', encoding="utf-8") as fw:
                json.dump(formula20, fw, indent=2, ensure_ascii=False)
            with open("./data/formulaBlemishine.json", 'w', encoding="utf-8") as fw:
                json.dump(formulaBlemishine, fw, indent=2, ensure_ascii=False)

            logger.success(
                f"formula.json, formula18.json, formula20.json, and formulaBlemishine.json are saved.")

            cls._formula18 = formula18
            cls._formula20 = formula20
            cls._formulaBlemishine = formulaBlemishine

        except Exception as e:
            logger.error("Fail to update formula.")
            logger.exception(e)
            # return dict()

    @classmethod
    def _get_current_time(cls, time: datetime.datetime | str | int | None) -> int:  # in seconds
        """Get `current time` in seconds, where `current time` is the corresponding time of CN server."""
        match time:
            case datetime.datetime():
                return int(time.timestamp())
            case str():
                candidates_ratios = [(k, SequenceMatcher(k, time).ratio())
                                     for k in cls.event_list]
                candidates_ratios = [c for c in candidates_ratios if c[1] > 0.5].sort(
                    key=lambda x: x[1], reverse=True)
                if candidates_ratios:
                    if candidates_ratios[0][1] == 1:
                        return cls.event_list[candidates_ratios[0][0]]
                    else:
                        logger.warning(
                            f"No matching event is found. Using {candidates_ratios[0][0]}. Candidates: {candidates_ratios[:3]}")
                else:
                    return pd.Timestamp(time, tz="Asia/Shanghai").value // int(1e9)

            case int():
                while time > 1e10:
                    time //= 1000
                    return time
            case None:
                time = pd.Timestamp("now", tz="Asia/Shanghai") - \
                    pd.Timedelta(days=276.3)  # about 9.21 months
                return time.value // int(1e9)
            case _:
                raise ValueError(f"{time} is not a valid time.")

    @staticmethod
    def add(d1, d2):
        d1 = d1.copy()
        for k, v in d2.items():
            if k in d1:
                d1[k] += v
            else:
                d1[k] = v
        return d1

    @staticmethod
    def sub(d1, d2):
        d1 = d1.copy()
        for k, v in d2.items():
            if k in d1:
                d1[k] -= v
            else:
                d1[k] = -v
        return d1

    @staticmethod
    def mul(d1, scalar):
        return {k: v*scalar for k, v in d1.items()}

    @staticmethod
    def replace_2dot(s: str) -> str:
        return s.replace(".", "・").replace("-", "・")
