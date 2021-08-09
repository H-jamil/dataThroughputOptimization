import numpy as np
import matplotlib.pyplot as plt
from operator import add

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(15, 10))

# set height of bar
# DI_memory=[16.28,16.224,16.89,19.223,18.15,17.79]
# SD_memory = [16.252,13.72,14.68, 14.226, 13.53, 15.08]
# DI_SD_memory= list(map(add, DI_memory, SD_memory))
# print(DI_SD_memory)

DI_memory=[6,8,8,8,7,7]
SD_memory = [5,5,6,6,7,7]
#DI_SD_memory= list(map(add, DI_memory, SD_memory))
DI_SD_memory = [max(DI_memory[i],SD_memory[i]) for i in range(len(SD_memory))]

print(DI_SD_memory)

# DI_memory_access=[16.28,16.224,16.89,19.223,18.15,17.79]
# SD_memory_access = [16.252,13.72,14.68, 14.226, 13.53, 15.08]
# DI_SD_memory= list(map(add, DI_memory, SD_memory))

# Set position of bar on X axis
br1 = np.arange(len(DI_memory))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, DI_memory, color=(0.2, 0.7, 0.6, 0.6), width = barWidth,
        edgecolor ='black', label ='Diversity Index')
plt.bar(br2, SD_memory, color=(0.7, 0.4, 0.6, 0.6), width = barWidth,
        edgecolor ='black', label ='Standard Deviation')
plt.bar(br3, DI_SD_memory, color=(0.2, 0.4, 0.7, 0.6), width = barWidth,
        edgecolor ='black', label ='DI+SD')

# Adding Xticks
plt.xlabel('HistoryLog', fontweight ='bold', fontsize = 15)
#plt.ylabel('Memory requirement in bytes per log', fontweight ='bold', fontsize = 15)
plt.ylabel('Worst Case # of memory access', fontweight ='bold', fontsize = 15)

plt.xticks([r + barWidth for r in range(len(DI_memory))],
        ['xsede_revised_1000','xsede_revised_2000','xsede_revised_3000','xsede_revised_4000','xsede_revised_5000','xsede_revised_6000'])

plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.savefig('memory_access.png')
plt.show()
