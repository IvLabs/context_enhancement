import torch
import train_translation
import barlow
import os 


# translation pretraining 
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 0')

# context enhancement
os.system('python ~/context_enhancement/context_enhancement/barlow.py --load 1') 

# tranining translation
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 1')

# testing translation
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 0')
