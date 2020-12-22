import pandas as pd
import sys
#kolumna do załadowania
col = sys.argv[1]

tables = [[[], [], []],
          [[], [], []],
          [[], [], []]]

letters = ['A', 'B', 'C']

for model in range(3):
    for test in range(3):
        table = pd.read_csv(f'model{letters[model]}/test{letters[test]}/table.csv')
        
        tables[model][test] = list(table[col])
    
print(tables)

text = r'\begin{table}[b!]'+'\n\t'+r'\centering'+'\n\t'+r'\begin{tabular}{|>{\columncolor{Gray!5}}l|*6{c|}}\hline'+'\n\t\t'
text += r'\rowcolor{Gray!20} dane testowe z grupy: & \multicolumn{3}{c|}{dane \textbf{A}} & \multicolumn{3}{c|}{dane \textbf{B}} \\\hline'+'\n\t\t'
text += r'\rowcolor{Gray!20} dane treningowe z grupy: & A & B & C & A & B & C \\\hline'+'\n\t\t'

labels = ['TAMSD', 'regresja liniowa', 'drzewo decyzyjne', 'las losowy', 'wzmocnienie gradientowe']

for model in range(len(labels)):
    text += labels[model]+r' & '
    for test in range(2):
        for Model in range(3):
            if Model == test:
                if model == 4:
                    text += r'\textbf{'
                text += r'\cellcolor{green!50}'
            text += '%4.3f'%(tables[Model][test][model])
            if Model == test and model == 4:
                text += r'}'
            text += r' & '
    text = text[:-3]
    text += r'\\\hline'+'\n\t\t'

text += r'\rowcolor{Gray!20} dane testowe z grupy: & \multicolumn{3}{c|}{dane \textbf{C}}\\\cline{1-4}'+'\n\t\t'
text += r'\rowcolor{Gray!20} dane treningowe z grupy: & A & B & C \\\cline{1-4}'+'\n\t\t'

test = 2
for model in range(len(labels)):
    text += labels[model]+r' & '
    for Model in range(3):
        if Model == test:
            if model == 4:
                text += r'\textbf{'
            text += r'\cellcolor{green!50}'
        text += '%4.3f'%(tables[Model][test][model])
        if Model == test and model == 4:
            text += r'}'
        text += r' & '
    text = text[:-3]
    text += r'\\\cline{1-4}'+'\n\t\t'

text += r'\end{tabular}' + '\n\t' + r'\caption{Współczynniki determinacji \textbf{R2} popełniane przez modele przy estymacji wykładnika dyfuzji dla danych z poszczególnych grup. Kolorem zielonym oznaczono najlepszą estymację w danej grupie danych testowych. Pogrubieniem natomiast najlepszy model dla każdej z zielonych kolumn.}'
text += '\n\t' + r'\label{tab: R^2 dla 1}' + '\n' + r'\end{table}'

with open(f'totex_{col}.txt', 'w') as f:
    f.write(text)
