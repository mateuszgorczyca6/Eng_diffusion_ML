
from os import remove, system
from itertools import product

to_divide = ['data/part1/modelA/generated/task1.txt',
             'data/part1/modelA/ML/random_forest/model.pk1',
             'data/part1/modelB/ML/random_forest/model.pk1',
             'data/part1/modelC/generated/task1.txt',
             'data/part1/modelC/ML/random_forest/model.pk1']



# dividing
for file in to_divide:
    system(f'split -b 90M {file} {file}')
    remove(file)
    print(file+' --- zakończono')
    
    
    
    
    
# joining
for file in to_divide:
    system(f'cat {file}* > {file}')
    for i in product('abcdefghijklmnoprstuvwxyz', repeat = 2):
        try:
            remove(f'{file}{"".join(i)}')
        except:
            pass
    print(file+' --- zakończono')
    
    
    