

joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
upper_body_mask = []
for i in joints:
    upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

joints = list(range(25,55))
hands_body_mask = []
for i in joints:
    hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

joints = [0,1,2,4,5,7,8,10,11]
lower_body_mask = []
for i in joints:
    lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])