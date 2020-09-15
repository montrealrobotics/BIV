from torch import nn
from torch import optim


class AgeModel(nn.Module):
    def __init__(self):

        """
        Description:
            The model is a CNN with 6 convolutional layers and 1 linear layer. The images has been transformed to gray before feeding them to the network,


        Args:
            :Conv 1: 
                Conv layer 
                    in_channels = 1, out_channels =10, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 10

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2


                

            :Conv 2: 
                Conv layer 
                    in_channels = 10, out_channels =20, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 20

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2, stride =2


            :Conv 3: 
                Conv layer 
                    in_channels = 20, out_channels =32, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 32

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2, stride =2


            :Conv 4: 
                Conv layer 
                    in_channels = 32, out_channels =64, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 64

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2, stride =2

            :Conv 5: 
                Conv layer 
                    in_channels = 64, out_channels =128, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 128

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2, stride =2

            :Conv 6: 
                Conv layer 
                    in_channels = 128, out_channels =256, kernel_size= 3, Stride= 1, Padding= 1

                BatchNorm2d
                    num_features= 256

                ReLU
                    a non linear function.

                MaxPool2d
                    kernel_size=2, stride =2
            :Linear: 
               
                in_features = 256*3*3 , out_features = 1        
        """
        super(AgeModel,self).__init__()



        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels = 20, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels = 32, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels = 256, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.linear = nn.Linear(in_features = 256*3*3 , out_features = 1 )

    def custom_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std=100)

    def forward(self,x):

        """
        Description:
            Predict an age using an image.

        Return:
            Batch of labels (ages)

        Return Type:
            Tensor
        
        Args:
            :x (tensor): Batch of images.
        """

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)

        return out


class WineModel(nn.Module):
    def __init__(self):
        super(WineModel, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_features = 11 , out_features = 100 ),
                    nn.BatchNorm1d(100),
                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(in_features = 100 , out_features = 50 ),
                    nn.BatchNorm1d(50),
                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(in_features = 50 , out_features = 20 ),
                    nn.BatchNorm1d(20),
                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(in_features = 20 , out_features = 10 ),
                    nn.BatchNorm1d(10),
                    nn.ReLU())
        self.layer5 = nn.Linear(in_features=10, out_features=1)
         
    

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out