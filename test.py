import dataloader

'''
117.0
[('34b', 19271), ('34c', 16175), (-1, 12885), ('34d', 12675), ('36c', 9436), ('32d', 7818), ('36d', 7791), ('32b', 7757), ('32c', 6907), ('36b', 6777), ('34a', 5523), ('32a', 3600), ('38d', 3020), ('38c', 1870), ('34dd', 1869), ('36a', 1430), ('32dd', 1202), ('36dd', 1018), ('34d+', 986), ('38b', 745), ('34ddd/e', 721), ('32d+', 668), ('36d+', 667), ('32ddd/e', 500), ('38d+', 449), ('38dd', 362), ('36ddd/e', 307), ('32aa', 223), ('34aa', 203), ('30d', 133), ('38ddd/e', 126), ('30c', 113), ('40d', 98), ('32f', 97), ('34f', 86), ('32g', 80), ('34g', 78), ('40dd', 78), ('30ddd/e', 70), ('36g', 65), ('30dd', 65), ('38a', 63), ('40c', 60), ('30b', 57), ('28b', 51), ('36f', 46), ('36aa', 46), ('30a', 45), ('34h', 44), ('28a', 41), ('28c', 39), ('38f', 38), ('40ddd/e', 36), ('42dd', 36), ('42c', 34), ('38g', 33), ('42d', 29), ('44dd', 28), ('28dd', 23), ('30f', 22), ('36h', 18), ('38h', 18), ('42ddd/e', 17), ('32h', 17), ('30g', 15), ('28aa', 14), ('28f', 14), ('40g', 12), ('40b', 10), ('34i', 9), ('34j', 9), ('42b', 9), ('28ddd/e', 8), ('40f', 7), ('30h', 7), ('30aa', 7), ('44d', 6), ('28d', 6), ('44c', 6), ('32i', 6), ('44ddd/e', 6), ('38aa', 5), ('32j', 5), ('44f', 5), ('42g', 5), ('36j', 4), ('42f', 4), ('40h', 4), ('30i', 4), ('46c', 3), ('28g', 3), ('38j', 3), ('44b', 3), ('38i', 3), ('36i', 3), ('40j', 2), ('42h', 1), ('46ddd/e', 1), ('28i', 1), ('44g', 1), ('28h', 1), ('48dd', 1), ('42j', 1), ('44h', 1)]
{'athletic': 30704, 'hourglass': 38819, 'petite': 15502, 'full bust': 10444, 'straight & narrow': 10315, -1: 10240, 'pear': 15579, 'apple': 3397}
[('dress', 65126), ('gown', 31094), ('sheath', 13532), ('shift', 3814), ('jumpsuit', 3604), ('top', 3438), ('maxi', 2454), ('romper', 2135), ('jacket', 1683), ('mini', 1228), ('skirt', 1078), ('sweater', 818), ('coat', 668), ('blazer', 554), ('shirtdress', 519), ('blouse', 473), ('down', 323), ('pants', 289), ('shirt', 197), ('vest', 189), ('cardigan', 173), ('frock', 143), ('culottes', 138), ('tank', 129), ('tunic', 108), ('suit', 87), ('sweatshirt', 86), ('bomber', 85), ('print', 78), ('pant', 76), ('leggings', 71), ('legging', 64), ('culotte', 63), ('cape', 62), ('midi', 41), ('trouser', 39), ('pullover', 37), ('poncho', 32), ('knit', 28), ('peacoat', 27), ('turtleneck', 22), ('kimono', 18), ('tee', 15), ('kaftan', 14), ('trench', 13), ('ballgown', 12), ('parka', 11), ('hoodie', 11), ('cami', 10), ('trousers', 10), ('blouson', 9), ('tight', 8), ('duster', 7), ('t-shirt', 7), ('skirts', 7), ('for', 7), ('skort', 5), ('combo', 5), ('henley', 5), ('overalls', 4), ('jeans', 4), ('jogger', 4), ('caftan', 3), ('overcoat', 2), ('sweatershirt', 2), ('crewneck', 1), ('sweatpants', 1)]
198.12
10.0
{'everyday': 11724, 'wedding': 40537, 'party': 25056, 'other': 10809, 'formal affair': 28243, 'work': 10638, 'vacation': 2847, 'date': 5141, -1: 5}
58
300.0
89
'''
a = [(1,2),(3,3)]
import numpy as np
f = dataloader.readTrain()
print(f[:10])
