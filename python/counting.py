input = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'D', 'E' ,'E', 'E', 'E', 'E', 'E', 'E']
#B
input.count('A')
diction01 = {}
for a in set(input):
    diction01[a] = input.count(a)
diction01.items()
{k: v for k, v in sorted(diction01.items(), key=lambda item: item[1])}
sorted(diction01.items(), key=lambda item: item[1])
sorted(diction01.items(), key=lambda item: item[1])[2][0]
