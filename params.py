
# Data


d_params = {
    
    'seed': 42,
    'd_path':"/datasets/UTKFace/*",
    'train_size': 16000, # training size, the test size is the rest of the data
    'tr_batch_size': 64,
    'test_batch_size': 2000,
    


}

# Model

n_params = {

    'lr': 10e-4,
    'epochs': 20,
    'architecture': "vCnn",
}