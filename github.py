
from os import remove, system
from itertools import product

to_divide = ['data/part1/ML/random_forest/model.pk1']



# dividing
for file in to_divide:
    system(f'split -b 90M {file} {file}')
    remove(file)
    
    
    
    
    
# joining
for file in to_divide:
    system(f'cat {file}* > {file}')
    for i in product('abcdefghijklmnoprstuvwxyz', repeat = 2):
        try:
            remove(f'{file}{"".join(i)}')
        except:
            pass
    
    
    