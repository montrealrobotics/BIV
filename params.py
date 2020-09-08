
# Data


d_params = {

    'server_path': '/final_outps/',

     ####################################### UTKFace #######################################
    
    'd_path':"UTKFace/*",
    'test_size': 4000,
    'dataset_size': 20000,
    'tr_batch_size': 256,
    'test_batch_size': 256,
    'd_img_mean_path':'/datasets/Datasets/AgePrediction/images_mean.csv' ,
    'd_img_std_path': '/datasets/Datasets/AgePrediction/images_std.csv' ,
    
    'd_lbl_mean_path': '/datasets/Datasets/AgePrediction/labels_mean.csv', 
    'd_lbl_std_path': '/datasets/Datasets/AgePrediction/labels_std.csv', 
    ####################################### Wine Quality #########################################
    'wine_path':"/datasets/Datasets/WineQuality/wine.csv",
    'wine_train_size': 4000,                            # Dataset size = 6491 
    'wine_tr_batch_size': 256,
    'wine_test_batch_size': 256,
    'wine_features_mean_path':'/datasets/Datasets/WineQuality/features_mean.csv',
    'wine_features_std_path': '/datasets/Datasets/WineQuality/features_std.csv',
    
    'wine_lbl_mean_path': '/datasets/Datasets/WineQuality/labels_mean.csv', 
    'wine_lbl_std_path': '/datasets/Datasets/WineQuality/labels_std.csv', 

}

# Model

n_params = {

    'lr': 0.001,
    'wine_lr': 0.001,
    'epochs': 20
}