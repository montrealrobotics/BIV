
# Data

default_values = {

    "epsilon": 0.1,
    "threshold_value":1,
    "distributions_ratio": 1,
    "maximum_hetero": False,
    "hetero_scale" :1,
    "warning_messages": {"bool":":argument is not boolean.", "datatype":"datatype is not supported.", "value":"argument value is not recognized.",\
        "CustomMess_dataset":"The sizes of the train dataset ({}) and the test dataset ({}) are together higher than the full dataset ({}), making it impossible for them to be mutually exclusive."}
}


d_params = {

    'server_path': '/home/vincent/Documents/Projects/BIV/Results/',

     ####################################### UTKFace #######################################
    
    'd_path':"/home/vincent/Documents/Projects/BIV/Datasets/UTKFCropped/*",
    'test_size': 4000,
    'dataset_size': 20000,
    'tr_batch_size': 1,
    'test_batch_size': 1,
    'd_img_mean_path':'/home/vincent/Documents/Projects/BIV/Datasets/UTKFCropped_stats/images_mean.csv' ,
    'd_img_std_path': '/home/vincent/Documents/Projects/BIV/Datasets/UTKFCropped_stats/images_std.csv' ,
    
    'd_lbl_mean_path': '/home/vincent/Documents/Projects/BIV/Datasets/UTKFCropped_stats/labels_mean.csv', 
    'd_lbl_std_path': '/home/vincent/Documents/Projects/BIV/Datasets/UTKFCropped_stats/labels_std.csv', 
    ####################################### Wine Quality #########################################
    'wine_path':"/datasets/Datasets/WineQuality/wine.csv",
    'wine_test_size': 2491,
    'wine_dataset_size': 6491,
    'wine_tr_batch_size': 256,
    'wine_test_batch_size': 256,
    'wine_features_mean_path':'/datasets/Datasets/WineQuality/features_mean.csv',
    'wine_features_std_path': '/datasets/Datasets/WineQuality/features_std.csv',
    
    'wine_lbl_mean_path': '/datasets/Datasets/WineQuality/labels_mean.csv', 
    'wine_lbl_std_path': '/datasets/Datasets/WineQuality/labels_std.csv', 

    ####################################### Bike Sharing #########################################
    'bike_path':"/home/vincent/Documents/Projects/BIV/Datasets/BikeSharing",
    'bike_test_size': 3379,
    'bike_dataset_size': 17379,
    'bike_tr_batch_size': 256,
    'bike_test_batch_size': 256,
    'bike_features_mean_path':'/home/vincent/Documents/Projects/BIV/Datasets/BikeSharing/features_mean.csv',
    'bike_features_std_path': '/home/vincent/Documents/Projects/BIV/Datasets/BikeSharing/features_std.csv',
    
    'bike_lbl_mean_path': '/home/vincent/Documents/Projects/BIV/Datasets/BikeSharing/labels_mean.csv', 
    'bike_lbl_std_path': '/home/vincent/Documents/Projects/BIV/Datasets/BikeSharing/labels_std.csv', 
}

# Model

n_params = {

    'lr': 0.001,
    'wine_lr': 0.01,
    'bike_lr': 0.001,
    'utkf_epochs': 20,
    'wine_epochs': 20,
    'bike_epochs': 100
}
