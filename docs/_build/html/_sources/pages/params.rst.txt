==========
Settings
==========

Here we can find the main settings that are needed for controling the code. Below, the settings are grouped as tables, you can find the counterpart code file here. However, there are another parameters that need to be specified through the commandline, as they are considered the most important paramters in our experiments, you can find a description of them `here <https://github.com/montrealrobotics/Deep-Heteroscedastic-Regression-Using-Privileged-Information-And-Mini-batch-Statistics#2-table>`_




.. list-table:: Default Settings
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - Epsilon
     - 0.1
     - A parameter that prevents the BIV function from having high loss values.
   * - Threshold
     - 1
     - The cutoff or noise threshold value of the cutoffMSE loss.
   * - Distribution rate
     - 1
     - Probability function over noise variance distributions. This is to study the contribution effect of low and high noise variance distributions.
   * - Maximum Heteroscedasticity
     - False
     - The largest value that the heteroscedasticity (the variance of the distribution from which we sample the noise variance) could take.

..
    _ * - Log path
    _"./outputs/"
    _description


.. list-table:: UTKFace Settings
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - Path of dataset directory
     - "./Datasets/AgePrediction/UTKFace/*"
     - UTKFace directory.
   * - Dataset size
     - 23708
     - Size of the dataset.
   * - Train size
     - 16000
     - Size of the training set.
   * - Test size
     - 4000
     - Size of the testing set.
   * - Train batch size
     - 256
     - Batch size of the training set.
   * - Test batch size
     - 256
     - Batch size of the testing set.
   * - Features mean
     - "./Datasets/AgePrediction/images_mean.csv"
     - A path to the emprical mean of the training images.
   * - Features std
     - "./Datasets/AgePrediction/images_std.csv"
     - A path to the emprical standard deviation of the training set.
   * - Labels mean
     - "./Datasets/AgePrediction/labels_mean.csv"
     - A path to the emprical mean of the training labels.
   * - Labels std
     - "./Datasets/AgePrediction/labels_std.csv"
     - A path to the emprical standard deviation of the training labels.




.. list-table:: Wine Quality Settings
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - Path of dataset directory
     - "./Datasets/WineQuality/wine.csv"
     - Path of dataset directory
   * - Dataset size
     - 6491
     - Size of the dataset.
   * - Train size
     - 4000
     - Size of the training set.
   * - Test size
     - 4000
     - Size of the testing set.
   * - Train batch size
     - 64
     - Batch size of the training set.
   * - Test batch size
     - 64
     - Batch size of the testing set.
   * - Features mean
     - "./Datasets/WineQuality/features_mean.csv"
     - A path to the emprical mean of the training features.
   * - Features std
     - "./Datasets/WineQuality/features_std.csv"
     - A path to the emprical standard deviation of the training features.
   * - Labels mean
     - "./Datasets/WineQuality/labels_mean.csv"
     - A path to the emprical mean of the training labels
   * - Labels std
     - "./Datasets/WineQuality/labels_std.csv"
     - A path to the emprical standard deviation of the training labels.


.. list-table:: Bike Sharing Settings
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - Path of dataset directory
     - "./Datasets/BikeSharing/"
     - Path of dataset directory
   * - Dataset size
     - 17379
     - Size of the dataset
   * - Train size
     - 14000
     - Size of the training set.
   * - Test size
     - 3379
     - Size of the testing set.
   * - Train batch size
     - 256
     - Batch size of the training set.
   * - Test batch size
     - 256
     - Batch size of the testing set.
   * - Features mean
     - "./Datasets/BikeSharing/features_mean.csv"
     - A path to the emprical mean of the training features.
   * - Features std
     - "./Datasets/BikeSharing/features_std.csv"
     - A path to the emprical standard deviation of the training features.
   * - Labels mean
     - "./Datasets/BikeSharing/labels_mean.csv"
     - A path to the emprical mean of the training labels
   * - Labels std
     - "./Datasets/BikeSharing/labels_std.csv"
     - A path to the emprical standard deviation of the training labels.


*Note: The above paths could be modified based on your personal setup. However, if you use our default configuration, make sure to follow the above theme.


.. list-table:: Models Settings
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Value
     - Description
   * - Learning rate (UTKFace)
     - 1e-3
     - The learning rate of Resnet model that used with UTKFace dataset.
   * - Learning rate (Wine Quality)
     - 1e-2
     -  The learning rate of vanilla ann model that used with Wine Quality dataset.
   * - Learning rate (Bike Sharing)
     - 1e-3
     -  The learning rate of vanilla ann model that used with Bike Sharing dataset.
   * - Epochs (Bike Sharing)
     - 100
     - Number of training epochs for Bike Sharing dataset.
   * - Epochs (UTKFace)
     - 20
     - Number of training epochs for UTKFace dataset.
