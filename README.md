# ArknightsMaterialPlannerTW （施工中... / In progress...）
以未來視，修正[明日方舟刷素材一圖流（ArkOneGraph）](https://aog.wiki/)失真狀況（適用繁中服（台服）、國際服（日、韓、美服））。
1. 沿用**線性規劃算法**（[ArkPlanner](https://github.com/penguin-statistics/ArkPlanner)），以未來視之掉落素材/關卡/機率，解決未來不確定性問題（適用繁中服（臺服））。
    - 由於原**線性規劃算法**可能存在某些缺點，見以下影片
[[明日方舟]"高产"的副本是如何统计出来的？三种原理大揭秘](https://www.bilibili.com/video/BV1pZ4y1g7QN)，其內容大致為**線性規劃算法**僅做出對於"當前"最優的刷圖推薦，其效益會受到未來活動關卡影響。此影片中提及**綠票算法**（馬可夫決策法（？），[yituliu](https://ark.yituliu.site/)）及如何克服上述缺點：活動關卡出現的素材較高機率為綠票算法中，較不划算之素材，例如：凝膠、輕錳礦，依影片中所說，此方法能較貼近未來的狀況。

2. ArkOneGraph 是以畢業所需總素材計算。在實際情況中，越"老"的幹員，越接近畢業，因此較老幹員的素材應（部分）剔除於總所需素材之中。
---




---
資料/策略/程式碼來源：企鹅物流-数据统计（[penguin-stats](https://penguin-stats.io/)）、企鹅物流-刷图规划器（[ArkPlanner](https://github.com/penguin-statistics/ArkPlanner)）、PRTS（[PRTS](https://prts.wiki)）及 明日方舟材料获取一图流（[yituliu](https://ark.yituliu.site/)）。