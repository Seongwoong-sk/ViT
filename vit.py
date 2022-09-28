import patchdata # patchdata.py
import model
import test
import torch
import torch.optim as optim
import torch.nn as nn
import argparse


# Hyperparameter
# Python에서 안 열어도 terminal에서 동작 가능 argparse!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision Transformer')
    parser.add_argument('--img_size', default=32, type=int, help='image size')
    parser.add_argument('--patch_size', default=4, type=int, help='patch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--save_acc', default=50, type=int, help='val acc') # 50%가 넘지 않으면 save 안되게
    parser.add_argument('--epochs', default=501, type=int, help='training epoch')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--drop_rate', default=.1, type=float, help='drop rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay') # L2 Regularization의 Penalty
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes') # class 개수에 따라서 MLP Head 마지막 단의 node수가 정해짐.
    parser.add_argument('--latent_vec_dim', default=128, type=int, help='latent dimension') # D, image patch가 linear projection 들어갈 때 나오는 vector dimension
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads') # Attention의 head
    parser.add_argument('--num_layers', default=12, type=int, help='number of layers in transformer') # ViT의 layers 수 (반복되는)
    parser.add_argument('--dataname', default='cifar10', type=str, help='data name') # 데이터 종류
    parser.add_argument('--mode', default='train', type=str, help='train or evaluation') # 학습 모드 설정 / 파일이 크면 보통 train과 val(inference) 따로 만듦
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')
    args = parser.parse_args()
    print(args)


    latent_vec_dim = args.latent_vec_dim
    mlp_hidden_dim = int(latent_vec_dim/2) # mlp_hidden_dim = 인코더 내 MLP의 은닉층 노드수
    num_patches = int((args.img_size * args.img_size) / (args.patch_size * args.patch_size)) # patch 개수는 hw/p^2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Image Patches
    # patchdata.py의 Flattened2Dpatches class
    d = patchdata.Flattened2Dpatches(dataname=args.dataname, img_size=args.img_size, patch_size=args.patch_size,
                                     batch_size=args.batch_size)
    trainloader, valloader, testloader = d.patchdata() # patchdata 만들어서 train, val, test 불러오고
    image_patches, _ = iter(trainloader).next() # patch의 size를 알기 위해서 패치 한 덩어리 불러옴

    # Model
    vit = model.VisionTransformer(patch_vec_size=image_patches.size(2), num_patches=image_patches.size(1),
                                  latent_vec_dim=latent_vec_dim, num_heads=args.num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                  drop_rate=args.drop_rate, num_layers=args.num_layers, num_classes=args.num_classes).to(device)

    if args.pretrained == 1:
        vit.load_state_dict(torch.load('./model.pth')) # 경로도 argparse에 넣어서 불러오고 싶은 모델명을 쳐서 넣을 수 있음. 현재는 가지고 있는 pth 고정값


    if args.mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # if args.optim == 'Adam':
        optimizer = optim.Adam(vit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 논문에선 fine-tuning할 때 모멘텀 사용
        # optimizer = torch.optim.SGD(vit.parameters(), lr=args.lr, momentum=0.9)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs)

        # Train
        n = len(trainloader) # 배치에 대한 평균 loss를 구하기 위해 선언 / 배치 사이즈 X , 배치의 개수 O
        best_acc = args.save_acc
        for epoch in range(args.epochs):
            running_loss = 0
            for img, labels in trainloader: # for문이 배치의 개수만큼 돔 / Flattended Patch가 들어옴
                optimizer.zero_grad()
                
                # Flattened Patch가 모델에 들어가면 Linear Projection을 거치고, Positional Embedding 해주고, 
                # Transformer Encoder로 들어가서 12번(layer 개수) 반복하고, MLP Head 들어가서 Output이 나옴
                outputs, _ = vit(img.to(device)) 
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #scheduler.step() # Scheduling 설정 / 여기다가 설정하면 매 Batch마다 
            #scheduler.step() # Scheduling 설정 / 여기다가 설정하면 매 Epoch마다 >> Epoch 기준으로 설정 
            

            train_loss = running_loss / n # 각 평균 loss를 구해서 대략적인 loss 상황 알 수 있음.
            val_acc, val_loss = test.accuracy(valloader, vit)
            if epoch % 5 == 0:
                                                                                        # train_loss와 val_loss를 비교해가면서 Overfitting 여부 진단
                                                                                        # 학습이 끝나면 정규화를 더 강하게 할 지 / 어떤 것을 가져갈 지 판단해서 Fine-Tune
                print('[%d] train loss: %.3f, validation loss: %.3f, validation acc %.2f %%' % (epoch, train_loss, val_loss, val_acc))

            if val_acc > best_acc: # val_acc가 기준보다 크다면
                best_acc = val_acc # 그 기준이 높은 val_acc로 저장

                print('[%d] train loss: %.3f, validation acc %.2f - Save the best model' % (epoch, train_loss, val_acc))
                torch.save(vit.state_dict(), './model.pth')

    else: # args.mode가 train이 아니면 다 평가가 됨.
        test_acc, test_loss = test.accuracy(testloader, vit)
        print('test loss: %.3f, test acc %.2f %%' % (test_loss, test_acc))
