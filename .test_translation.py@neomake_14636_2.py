import torch import train_translation import barlow import os 


# translation pretraining 
# sweep translation 
# wandb sweep_translation.yaml 
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 0')

# context enhancement
# sweep barlow with translation encoder hyper-params 
# sweep sweep_barlow.yaml
os.system('python ~/context_enhancement/context_enhancement/barlow.py --load 1') 

# tranining translation
#train translation  with translation hyper-params
#python train_translation.py 
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 1')

# testing translation
# no need
os.system('python ~/context_enhancement/context_enhancement/train_translation.py --load 0')
