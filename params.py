
# Data


d_params = {
    
    'seed': 42,
    'd_path':"./Datasets/AgePrediction/UTKFace/*",
    'train_size': 16000,                            #23708, # training size, the test size is the rest of the data
    'tr_batch_size': 64,
    'test_batch_size': 2000,

    'd_img_mean_path':'./Datasets/AgePrediction/images_mean.csv' ,#'/datasets/images_mean.csv',
    'd_img_std_path': './Datasets/AgePrediction/images_std.csv' ,#'/datasets/images_std.csv',
    
    'd_lbl_mean_path': './Datasets/AgePrediction/labels_mean.csv', #'/datasets/labels_mean.csv',
    'd_lbl_std_path': './Datasets/AgePrediction/labels_std.csv', #'/datasets/labels_std.csv',

}

# Model

n_params = {

    'lr': 10e-4,
    'epochs': 1,
    'architecture': "vCnn",
}