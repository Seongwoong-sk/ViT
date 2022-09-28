import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


class PatchGenerator:

    # input값이 있으면 init에다가 input 값 잡아주기
    def __init__(self, patch_size):
        self.patch_size = patch_size

    # 이 함수가 호출되었을 때 어떤 작업을 할 것인지 def __call__ 함수 안에서 정의
    # 원래 데이터를 썰고 썰어서 정육면체 형태로 만든 다음에 가로세로해서 네모나게 모여 있는 것을 일렬로 배열시킨 다음에 각각이 패치를 마지막에 다 일렬로 펴서 보내주는 것

    def __call__(self, img):
        num_channels = img.size(0) # img.size의 첫 번째가 channel -> transforms은 이미지 한 장 기준으로 진행됨.
        # x.unfold(dimension, size of each slice, stride)
                           # Dimension:1(세로)로 patch_size를 16x16으로 한 번 썰고, Dimension:2(가로)로 한 번 썸 --> 3 x 16 x 16 x (16 x 16) : patch_size 크기니깐
                           # 가운데 있는 16x16를 일렬로 바꾸는 거니깐 256이 들어가게끔 reshape해서 -1 설정
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1,0,2,3) # 개수를 앞쪽으로 빼주기 위해서 256 x 3 x 16 x 16으로 만듬 -> 3x16x16이 256개가 되더라.
        num_patch = patches.size(0) # patches의 가장 첫 번째가 patch의 개수

        return patches.reshape(num_patch,-1) # Linear Projection에 들어가기 직전의 patch vector들의 모임


# 오리지널 데이터를 patch 데이터로 만드는 클래스
class Flattened2Dpatches:

    def __init__(self, patch_size=16, dataname='imagenet', img_size=256, batch_size=64):
        '''
        patch_size: patch size
        dataname: 데이터 종류
        img_size: 인풋 사이즈 어떻게 resize 시킬 건지
        batch_size: batch_size
        '''
        self.patch_size = patch_size
        self.dataname = dataname
        self.img_size = img_size
        self.batch_size = batch_size

    # WeightedRandomSampler의 weight을 만들어주는 함수
    # numpy로 만들어지니깐 후에 tensor로 만들어줌.
    def make_weights(self, labels, nclasses):
        labels = np.array(labels)
        weight_arr = np.zeros_like(labels)
        _, counts = np.unique(labels, return_counts=True)
        for cls in range(nclasses):
            weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
    
        return weight_arr 

    def patchdata(self):
        # cifar10에 대한 img mean , std
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
                                                # 직사각형이면 (250, 300) 이렇게 입력                       # padding을 붙여서 똑같은 이미지 사이즈를 cropping
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),
                                              PatchGenerator(self.patch_size)])
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean, std), PatchGenerator(self.patch_size)])

        # 이 라인에서 데이터셋 쭉 정리
        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

            # test셋을 반반으로 나눠서 val, test 사용
            evens = list(range(0, len(testset), 2)) # test셋의 데이터들의 짝수는 val
            odds = list(range(1, len(testset), 2)) # 홀수는 test셋으로
            # Subset(나누고자 하는 데이터, 각 인덱스에 해당되는 번호)
            valset = torch.utils.data.Subset(testset, evens) # 짝수 인덱스
            testset = torch.utils.data.Subset(testset, odds) # 홀수 인덱스
          
        elif self.dataname == 'imagenet':
            pass # 예시는 cifar10으로

        weights = self.make_weights(trainset.targets, len(trainset.classes))  # 가중치 계산
        weights = torch.DoubleTensor(weights) # 함수에서 numpy로 만들어지니깐  tensor로 만들어줌.
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        # 학습할 때 한 배치당 각 클래스가 동일한 개수가 들어올 수 있도록 WeightedRandomSampler 적용
        # WeightedRandomSampler는 각 클래스마다 전체 확률을 동일하게 하자는 컨셉
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler) # sampler를 통해 uniform하게 학습할 수 있다.
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

        return trainloader, valloader, testloader


### 패치가 제대로 나눠지는 지 확인하기 위해 테스트 용도로 사용 ###
### Normalize 적용 안함.이미지에 대한 일반화보다 분할 확인용   ###
##################################################################
def imshow(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig('pacth_example.png')

if __name__ == "__main__":
    print("Testing Flattened2Dpatches..")
    batch_size = 64
    patch_size = 8
    img_size = 32
    num_patches = int((img_size*img_size)/(patch_size*patch_size))
    d = Flattened2Dpatches(dataname='cifar10', img_size=img_size, patch_size=patch_size, batch_size=batch_size)
    trainloader, _, _ = d.patchdata()
    images, labels = iter(trainloader).next() # 첫 번째 batch에 대해 image와 label 불러옴
    print(images.size(), labels.size())

    sample = images.reshape(batch_size, num_patches, -1, patch_size, patch_size)[0]
    print("Sample image size: ", sample.size())
    imshow(torchvision.utils.make_grid(sample, nrow=int(img_size/patch_size)))
#####################################################################
### 패치가 제대로 나눠지는 지 확인하기 위해 테스트 용도로 사용 ###