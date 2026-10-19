"""Pair-by-pair reading of the 270 pairs (rule in selection_rule.md). same / different / uncertain."""
import pandas as pd
U = pd.read_csv('universe270.csv', dtype={'workflow_id':str,'board_id':str})
different = {221,249,254,266,158,186,182,224,193,199,157,195,201,159,160,194,200,235,236,232,212,214,238,
             161,162,179,180,190,191,196,172,171,192,197,239,216,248,227,150,152}
uncertain = {116, 205, 220, 229, 260}
notes = {
 116: '药剂师个人条目，与Setzer Pharmacy同址',
 205: '同一街道，门牌号250与246，同为Cub，可能是同一家门店',
 220: 'Aliveness Project的诊所服务条目，与其药房THRIVE Rx同址',
 229: 'Google地址写作Target Plaza South, 900 Nicollet Mall；Board #17626在1000 Nicollet Mall（TPS-272）',
 260: '同上',
 228: 'Google地址含900 Nicollet Mall，解析器取到了Target Plaza South，未取到门牌号',
 259: '同上',
 167: 'Target门店，店内CVS药房即Board #16714',
 25:  'Wellness Pharmacy，Google为45 10th St E，Board为45 10th St W，名称独特，判为同一家',
 205.5: '',
 222: 'Lakes Pharmacy与Lake Cedar Pharmacy LLC同址，判为同一家（法定名称与店名）',
 151: "Sam's Club门店，店内药房执照登记为Sam's West Inc.",
 150: "门牌号同为8150，但街道不同（University Ave NE与Wedgewood Lane），不同门店",
 152: '同上',
}
U['reading'] = ['不同' if p in different else '不确定' if p in uncertain else '同一家' for p in U.pid]
U['note'] = [notes.get(p, '') for p in U.pid]
U.to_csv('pair_readings_270.csv', index=False, encoding='utf-8-sig')
print(U.groupby(['src','reading']).size())
