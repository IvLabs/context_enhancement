class Config: 

# Training Hyperparameters:     
    LR = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0
    ETA = 0.001
    EPOCHS = 1

# Paths: 
    LOSS_PATH = '/home/terasquid/Documents/NLP/context_enhancement/loss.txt'
    MODEL_STORE_PATH = '/home/terasquid/Documents/NLP/context_enhancement/'
    CHECKPOINT_PATH = '/home/terasquid/Documents/NLP/context_enhancement/'

# Dataset parameters: 
    LANG_PAIR = 'de-en'
    SPLIT = 'test'
    BATCH_SIZE = 8

# Tokenizer parameters: 
    TOKENIZER = 'bert-base-multilingual-cased'

# transformer hyperparameters: 
    D_MODEL = 768
    N_HEAD = 4
    DIM_FEEDFORWARD = 256
    NUM_LAYERS = 2
    
# Model Hyperparameters:
    PROJECTOR_LAYERS = '768-800'
    MBERT_OUT_SIZE = 768
    LAMBD = 0.6

# GPU parameters: 
    NUM_GPUS = 1
    NODES = 4
    WORLD_SIZE = 4 
