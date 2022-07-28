import requests
import json
import numpy as np
from os.path import isfile
from loguru import logger
import pandas as pd

## useful urls from penguin-statistics (penguin-stats) and arknights-toolbox (arktools)
URL_ITEM = "https://penguin-stats.io/PenguinStats/api/v2/items"
URL_STAGE = "https://penguin-stats.io/PenguinStats/api/v2/stages"
URL_MATRIX = "https://penguin-stats.io/PenguinStats/api/v2/result/matrix?show_closed_zone=true"
URL_MATRIX_PRIVATE = "https://penguin-stats.io/PenguinStats/api/v2/_private/result/matrix/CN/global"
URL_EVENT = "https://penguin-stats.io/PenguinStats/api/v2/period"


URL_OPERATOR = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/character.json"
# URL_OPERATOR_ORDER = "https://prts.wiki/w/%E6%A8%A1%E5%9D%97:GetCharCnOnlineTime"
URL_OPERATOR_ORDER = "https://prts.wiki/w/%E5%B9%B2%E5%91%98%E4%B8%8A%E7%BA%BF%E6%97%B6%E9%97%B4%E4%B8%80%E8%A7%88"
URL_SKILL = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/skill.json"
URL_UNIEQUIP = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/uniequip.json"
URL_CULTIVATE = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/data/cultivate.json"
URL_ZONE = "https://raw.githubusercontent.com/arkntools/arknights-toolbox/master/src/locales/cn/zone.json"
URL_PRICE = "https://raw.githubusercontent.com/penguin-statistics/ArkPlanner/master/price.txt"

class DataCollector:
    def __init__(self, update = False):
        self.item_map = self.load_item_map(update)
        self.stage_map = self.load_stage_map(update)
        self.operator_map = self.load_operator_map(update)
        self.skill_map = self.load_skill_map(update)
        self.uniequip_map = self.load_uniequip_map(update)
        self.cultivate_map = self.load_cultivate_map(update)
        self.event_list = self.load_event_list(update)
        self.zone_map = self.load_zone_map(update)
        self.zone_matrix = self.load_zone_matrix(update)
        
        self.operator_order = self.load_operator_order(update)
        self.LAST_OPERATOR_TIME = self.operator_order.pop('LAST_OPERATOR_TIME')
   

    def generate_future_activity_materials(self, 
                current_event_name: str = "玛莉娅・临光・复刻",
                sanity_per_day: int = 310,
                discount_rate_per_week: float = 0.95) -> dict[str, float]: # farm
        # TODO: 算cp值 -> 刷 x 體在最多的那關，得到素材（期望值） -> 算cp值 -> 刷 x 體在最多的那關，得到素材（期望值）-> ... -> 沒體力
        # TODO: 算cp值 =  MaterialPlanner.value + self.zone_matrix
        pass
    
    
    def _generate_future_activity_materials_default(self, 
                current_event_name: str = "玛莉娅・临光・复刻",
                sanity_per_day: int = 310,
                discount_rate_per_week: float = 0.95) -> dict[str, float]: # farm
        
        ### event_list -> stage_matrix
        current_time = self.event_list[current_event_name][0]
        BLUE_MATERIALS = {"全新装置", "RMA70-12", "研磨石", "凝胶", "炽合金", "酮凝集组", "轻锰矿", "异铁组", "扭转醇",
                            "聚酸酯组", "糖组", "固源岩组", "晶体元件", "化合切削液", "半自然溶剂"}

        # activity farm info in penguin-stats
        activity_record = dict()
        power = 1.2
        for zone, time in self.event_list.items():
            if zone in self.zone_matrix and time[0] > current_time:
                activity_record[zone] = {"start_date": time[0], "duration": "{:.1f} days".format((time[1] - time[0]) / 86400), "stage": dict()}
                if "mini" in self.zone_map[zone]:
                    continue

                # sanity_list
                sanity_list = [list(v["drop_info"].values())[0]["times"] * v["sanity"]  
                               for v in self.zone_matrix[zone].values() if v["drop_info"]]


                threshold = max(0.2 * np.mean(sanity_list),0)
                sanity_list = [t if t > threshold else 0 for t in sanity_list]
                sanity_list_ = np.array(sanity_list) ** power

                total = sanity_list_.sum()


                for k, v in self.zone_matrix[zone].items():
                    if v["drop_info"]:
                        sanity_cost = list(v["drop_info"].values())[0]["times"] * v["sanity"] # total snaity cost in penguin-stats
                        if sanity_cost > threshold:
                            ratio = (sanity_cost**power / total)
                            total_sanity = (sanity_per_day * (time[1] - time[0]) / 86400)
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
    
    
    def generate_required_materials(self, eku_map = dict(), url = "", path = None): # check list of operators elite (2)/ skill (6+3) / uniequip (3)
        # TODO: calculate precise required materials
        pass
    
    
    def _generate_required_materials_default(self, discount_rate_per_week = 0.98):
        required_materials = dict()
      

        not_in_cultivate_map = list()
        for operator, online_time in self.operator_order.items():
            week = (self.LAST_OPERATOR_TIME - online_time) / 86400 / 7
            discount = discount_rate_per_week ** week

            if operator not in self.cultivate_map:
                not_in_cultivate_map.append(operator)
                continue

            info = self.cultivate_map[operator]

            for items in info["evolve"]: # 精1 ~ 2
                for k, v in items.items():
                    items[k] = v * discount
                required_materials = DataCollector.add(required_materials, items)

            for items in info["skills"]["normal"]: # 技1 ~ 7
                for k, v in items.items():
                    items[k] = v * discount
                required_materials = DataCollector.add(required_materials, items)

            for skill in info["skills"]["elite"]: # 一 ~ 三技
                for items in skill["cost"]: # 專1 ~ 3
                    for k, v in items.items():
                        items[k] = v * discount
                    required_materials = DataCollector.add(required_materials, items)

            for xy in info["uniequip"]: # 模組 X,Y
                for items in xy["cost"]: # 1 ~ 3級
                    for k, v in items.items():
                        items[k] = v * discount
                    required_materials = DataCollector.add(required_materials, items)
        print(", ".join(not_in_cultivate_map), "are not in cultivate_map.")
        
        
        self.required_materials = required_materials
        return required_materials
        
        
    def calculate_zone_value(self, values):
        if isinstance(values, list):
            values = {items["name"]: items["value"] for d in values for items in d["items"]}
            
        values |= {"基础作战记录" : 200 * 0.0035,
                    "初级作战记录" : 400 * 0.0035,
                    "中级作战记录" : 1000 * 0.0035,
                    "高级作战记录" : 2000 * 0.0035,
                    "赤金" : 400 * 0.0035,
                    "龙门币" : 0.0035}
        
        zone_value = dict()
        for zone, stages in self.zone_matrix.items():
            zone_value[zone] = dict()
            for stage_code, stage_info in stages.items():
                equivalent_sanity = 0
                for item, item_info in stage_info["drop_info"].items():
                    if item not in values:
                        # print(item, item_info)
                        continue
                    equivalent_sanity += float(values[item]) * item_info["quantity"] / item_info["times"]
                equivalent_sanity += stage_info["sanity"]  * 12 * 0.0035
                cp = equivalent_sanity / stage_info["sanity"] if stage_info["sanity"] > 0 else -1
                zone_value[zone][stage_code] = (equivalent_sanity, cp)
        
        self.zone_value = zone_value
        self.values = values
        
        return zone_value
        
        
    def update_all(self):
        self.item_map = self.load_item_map(True)
        self.tage_map = self.load_stage_map(True)
        self.operator_map = self.load_operator_map(True)
        self.skill_map = self.load_skill_map(True)
        self.uniequip_map = self.load_uniequip_map(True)
        self.cultivate_map = self.load_cultivate_map(True)
        self.event_list = self.load_event_list(True)
        self.zone_map = self.load_zone_map(True)
        self.zone_matrix = self.load_zone_matrix(True)
        
        self.operator_order = self.load_operator_order(True)
        self.LAST_OPERATOR_TIME = self.operator_order.pop('LAST_OPERATOR_TIME')
        
    @property
    def zone_value_filtered(cp = 0.99):
        {zone: {stage: value for stage, value in stages.items() if value[1] > cp} for zone,stages in self.zone_value.items() }
    # item_map
    @staticmethod
    def load_item_map(update = False): 
        if not isfile("./data/item_map.json"):
            logger.warning(f"item_map.json is not found. Download data from {URL_ITEM}")

        if update or not isfile("./data/item_map.json"):
            return DataCollector._save_item_map()
        else:
            with open("./data/item_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod        
    def _save_item_map():
        try:
            with requests.get(URL_ITEM) as response:
                txt = json.loads(response.text)

            item_map = dict() # item_id2item_name | item_name2item_id
            for item in txt:
                item_id = item["itemId"].strip()
                item_name = item["name"].strip()

                item_map[item_id] = item_name
                item_map[item_name] = item_id

            # raw data        
            with open("./data/raw/item.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"Download item.json successfully from {URL_ITEM}.")

            # item_map
            with open("./data/item_map.json", 'w', encoding="utf-8") as fw:
                json.dump(item_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"item.json and item_map.json are saved.")

            return item_map
        except Exception as e:
            logger.error("Fail to update item_map.")
            logger.error(e)
            return dict()
        
        
    # stage_map
    @staticmethod
    def load_stage_map(update = False) -> dict[str, list[str]]:
        if not isfile("./data/stage_map.json"):
            logger.warning(f"stage_map.json is not found. Download data from {URL_STAGE}")   

        if update or not isfile("./data/stage_map.json"):
            return DataCollector._save_stage_map()
        else:
            with open("./data/stage_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod
    def _save_stage_map():
        try:
            with requests.get(URL_STAGE) as response:
                txt = json.loads(response.text)

            stage_map = dict() # stage_id2stage_name_list | stage_name2stage_ids_list, actually stage_code

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
            logger.info(f"stage.json is successfully downloaded from {URL_STAGE}.")

            # stage_map
            with open("./data/stage_map.json", 'w', encoding="utf-8") as fw:
                json.dump(stage_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"stage.json and stage_map.json are saved.")

            return stage_map
        except Exception as e:
            logger.error("Fail to update stage_map.")
            logger.error(e)
            return dict()
        
        
    #  operator_map
    @staticmethod
    def load_operator_map(update = False) -> dict[str, list[str]]:
        if not isfile("./data/operator_map.json"):
            logger.warning(f"operator_map.json is not found. Download data from {URL_OPERATOR}")   

        if update or not isfile("./data/operator_map.json"):
            return DataCollector._save_operator_map()
        else:
            with open("./data/operator_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod        
    def _save_operator_map() -> dict[str, str]: 
        try:
            with requests.get(URL_OPERATOR) as response:
                txt = json.loads(response.text)

            operator_map = dict() # operator_id2operator_name | operator_name2operator_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                operator_map[_id] = name
                operator_map[name] = _id

            # raw data        
            with open("./data/raw/operator.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"operator.json is successfully downloaded from {URL_OPERATOR}.")

            # operator_map
            with open("./data/operator_map.json", 'w', encoding="utf-8") as fw:
                json.dump(operator_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"operator.json and operator_map.json are saved.")

            return operator_map
        except Exception as e:
            logger.error("Fail to update operator_map.")
            logger.error(e)
            return dict()
        
        
    #  operator_order
    @staticmethod
    def load_operator_order(update = False) -> dict[str, list[str]]:
        if not isfile("./data/operator_order.json"):
            logger.warning(f"operator_order.json is not found. Download data from {URL_OPERATOR_ORDER}")   

        if update or not isfile("./data/operator_order.json"):
            return DataCollector._save_operator_order()
        else:
            with open("./data/operator_order.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod        
    def _save_operator_order() -> dict[str, str]: 
        try:
            dfs = pd.read_html(URL_OPERATOR_ORDER)
            operator_order = {row.干员: int(pd.to_datetime(row.国服上线时间, format='%Y年%m月%d日 %H:%M').timestamp())\
                              for i, row in dfs[0].iterrows()}
            operator_order["LAST_OPERATOR_TIME"] = max(operator_order.values())
            
            # operator_order
            with open("./data/operator_order.json", 'w', encoding="utf-8") as fw:
                json.dump(operator_order, fw, indent=2, ensure_ascii=False)
            logger.info(f"operator_order.json is saved.")

            return operator_order
        except Exception as e:
            logger.error("Fail to update operator_order.")
            logger.error(e)
            return dict()    
        
        
    # skill_map 
    @staticmethod
    def load_skill_map(update = False) -> dict[str, list[str]]:
        if not isfile("./data/skill_map.json"):
            logger.warning(f"skill_map.json is not found. Download data from {URL_SKILL}")   

        if update or not isfile("./data/skill_map.json"):
            return DataCollector._save_skill_map()
        else:
            with open("./data/skill_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod
    def _save_skill_map() -> dict[str, str]: 
        try:
            with requests.get(URL_SKILL) as response:
                txt = json.loads(response.text)

            skill_map = dict() # skill_id2skill_name | skill_name2skill_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                skill_map[_id] = name
                skill_map[name] = _id

            # raw data        
            with open("./data/raw/skill.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"skill.json is successfully downloaded from {URL_SKILL}.")

            # skill_map
            with open("./data/skill_map.json", 'w', encoding="utf-8") as fw:
                json.dump(skill_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"skill.json and skill_map.json are saved.")

            return skill_map
        except Exception as e:
            logger.error("Fail to update skill_map.")
            logger.error(e)
            return dict()
        
        
    # uniequip_map
    @staticmethod
    def load_uniequip_map(update = False) -> dict[str, list[str]]:
        if not isfile("./data/uniequip_map.json"):
            logger.warning(f"uniequip_map.json is not found. Download data from {URL_UNIEQUIP}")   

        if update or not isfile("./data/uniequip_map.json"):
            return DataCollector._save_uniequip_map()
        else:
            with open("./data/uniequip_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod
    def _save_uniequip_map() -> dict[str, str]: 
        try:
            with requests.get(URL_UNIEQUIP) as response:
                txt = json.loads(response.text)

            uniequip_map = dict() # uniequip_id2uniequip_name| uniequip_name2uniequip_ids
            for _id, name in txt.items():
                _id = _id.strip()
                name = name.strip()

                uniequip_map[_id] = name
                uniequip_map[name] = _id

            # raw data        
            with open("./data/raw/uniequip.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"uniequip.json is successfully downloaded from {URL_UNIEQUIP}.")

            # uniequip_map
            with open("./data/uniequip_map.json", 'w', encoding="utf-8") as fw:
                json.dump(uniequip_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"uniequip.json and uniequip_map.json are saved.")

            return uniequip_map
        except Exception as e:
            logger.error("Fail to update uniequip_map.")
            logger.error(e)
            return dict()
    
    
    # cultivate_map
    @staticmethod
    def load_cultivate_map(update = False) -> dict[str, ]:
        if not isfile("./data/cultivate_map.json"):
            logger.warning(f"cultivate_map.json is not found. Download data from {URL_CULTIVATE}")   

        if update or not isfile("./data/cultivate_map.json"):
            return DataCollector._save_cultivate_map()
        else:
            with open("./data/cultivate_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod
    def _save_cultivate_map() -> dict[str, str]: 
        try:
            with requests.get(URL_CULTIVATE) as response:
                txt = json.loads(response.text)

            item_map = DataCollector.load_item_map()
            operator_map = DataCollector.load_operator_map()
            skill_map = DataCollector.load_skill_map()
            uniequip_map = DataCollector.load_uniequip_map()
            cultivate_map = {operator_map[_id]: {
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
            logger.info(f"cultivate.json is successfully downloaded from {URL_CULTIVATE}.")

            # operator_map
            with open("./data/cultivate_map.json", 'w', encoding="utf-8") as fw:
                json.dump(cultivate_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"cultivate.json and cultivate_map.json are saved.")

            return cultivate_map
        except Exception as e:
            logger.error("Fail to update cultivate_map.")
            logger.error(e)
            return dict()
        
        
    # event_list
    @staticmethod
    def load_event_list(update = False) -> dict:
        if not isfile("./data/event_list.json"):
            logger.warning(f"event_list.json is not found. Download data from {URL_EVENT}")   

        if update or not isfile("./data/event_list.json"):
            return DataCollector._save_event_list()
        else:
            with open("./data/event_list.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod       
    def _save_event_list() -> list: 
        try:
            with requests.get(URL_EVENT) as response:
                txt = json.loads(response.text)

            event_list = dict()
            event_list = {event["label_i18n"]["zh"].replace("·", "・"): [event["start"] // 1000, event["end"] // 1000 if event["end"] else None] 
                            for event in txt if event["existence"]["CN"]["exist"]}

            # raw data        
            with open("./data/raw/event.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"event.json is successfully downloaded from {URL_EVENT}.")

            # event_list
            with open("./data/event_list.json", 'w', encoding="utf-8") as fw:
                json.dump(event_list, fw, indent=2, ensure_ascii=False)
            logger.info(f"event.json and event_list.json are saved.")

            return event_list
        except Exception as e:
            logger.error("Fail to update event_list.")
            logger.error(e)
            return dict()

        
    # zone_map
    @staticmethod
    def load_zone_map(update = False) -> dict[str, list[str]]:
        if not isfile("./data/zone_map.json"):
            logger.warning(f"zone_map.json is not found. Download data from {URL_ZONE}")   

        if update or not isfile("./data/zone_map.json"):
            return DataCollector._save_zone_map()
        else:
            with open("./data/zone_map.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read())
            
    @staticmethod
    def _save_zone_map() -> dict[str, str]: 
        try:
            with requests.get(URL_ZONE) as response:
                txt = json.loads(response.text)

            zone_map = dict() # zone_id2zone_name | zone_name2zone_ids
            for _id, name in txt.items():
                if "@" in name:
                    _id = "_".join(_id.split("_")[:3])
                    name = name.replace("@", "・永久@").split("@")[0].strip()
                else:
                    _id = _id.strip()
                    name = name.strip().replace("·", "・")


                zone_map[_id] = name
                zone_map[name] = _id

            # raw data        
            with open("./data/raw/zone.json", 'w', encoding="utf-8") as fw:
                json.dump(txt, fw, indent=2, ensure_ascii=False)  
            logger.info(f"zone.json is successfully downloaded from {URL_ZONE}.")

            # zone_map
            with open("./data/zone_map.json", 'w', encoding="utf-8") as fw:
                json.dump(zone_map, fw, indent=2, ensure_ascii=False)
            logger.info(f"zone.json and zone_map.json are saved.")

            return zone_map
        except Exception as e:
            logger.error("Fail to update zone_map.")
            logger.error(e)
            return dict()
        
        
    # load_zone_matrix
    @staticmethod
    def load_zone_matrix(update = False) -> dict[str, list[str]]:
        if not isfile("./data/zone_matrix.json"):
            logger.warning(f"zone_matrix.json is not found. Download data from {URL_MATRIX_PRIVATE} & {URL_STAGE}")   

        if update or not isfile("./data/zone_matrix.json"):
            return DataCollector._save_zone_matrix()
        else:
            with open("./data/zone_matrix.json", 'r', encoding="utf-8-sig") as fr:
                return json.loads(fr.read()) 
            
    @staticmethod
    def _save_zone_matrix() -> dict: 
        """
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
            item_map = DataCollector.load_item_map()
            stage_map = DataCollector.load_stage_map()
            zone_map = DataCollector.load_zone_map()

            zone_matrix = dict()

            # MATRIX
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

                if stage_id not in stage_id2matrix: 
                    stage_id2matrix[stage_id] = dict()
                stage_id2matrix[stage_id][item_name] = {"times": times, "quantity": quantity, "std_dev": std_dev}

            # STAGE
            with requests.get(URL_STAGE) as response:
                txt = json.loads(response.text)

            for stage in txt:
                stage_id = stage["stageId"]
                stage_name = stage_map[stage_id][0]
                zone_id = stage["zoneId"].replace("_zone1", "")
                zone_name = zone_map[zone_id] if zone_id in zone_map else stage["code"].split("-")[0]
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
            logger.info(f"matrix.json is successfully downloaded from {URL_MATRIX}.")

            # zone_matrix
            with open("./data/zone_matrix.json", 'w', encoding="utf-8") as fw:
                json.dump(zone_matrix, fw, indent=2, ensure_ascii=False)
            logger.info(f"matrix.json and zone_matrix.json are saved.")

            return zone_matrix
        except Exception as e:
            logger.error("Fail to update stage_matrix.")
            logger.error(e)
            return dict()

    
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
        
        
        