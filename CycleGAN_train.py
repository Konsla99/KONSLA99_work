import torch
from torchvision import transforms
from torch.autograd import Variable
#dataset을 불러오고 generator와 discriminator 를 불러와 학습을 진행
from dataset import DatasetFromFolder
from model import Generator, Discriminator

import utils
import argparse
import os, itertools
from logger import Logger
import numpy as np

import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='amature2master', help='input dataset')#학습 시킬 dataset의 종류
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
#generator 와 discriminato 의 parameter
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_resnet', type=int, default=9, help='number of resnet blocks in generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=550, help='number of train epochs')
parser.add_argument('--decay_epoch', type=int, default=475, help='start decaying learning rate after this number')
#0.002에서 인식률 너무 낮았음 1e-5 이하부터 loss 가 현저하게 낮아지는 것이 확인됨
parser.add_argument('--lrG', type=float, default=3e-6, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=3e-6, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')
parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')
#optimizer 수정해보기
parser.add_argument('--beta1', type=float, default=0.7, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = 'D:/CYCLE_GAN/CycleGAN-1/dataset/' + params.dataset + '/'
save_dir = params.dataset + '_results/'
model_dir = params.dataset + '_model/'

# train data를 저장할 dir존재x라면 해당 dir 생성
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data_A = DatasetFromFolder(data_dir, subfolder='trainA', transform=transform,
                                 resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_A = torch.utils.data.DataLoader(dataset=train_data_A, batch_size=params.batch_size,shuffle=True)

train_data_B = DatasetFromFolder(data_dir, subfolder='trainB', transform=transform,
                                 resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader_B = torch.utils.data.DataLoader(dataset=train_data_B,batch_size=params.batch_size, shuffle=True)

# Test data
test_data_A = DatasetFromFolder(data_dir, subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A, batch_size=params.batch_size, shuffle=False)
test_data_B = DatasetFromFolder(data_dir, subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B, batch_size=params.batch_size, shuffle=False)

# 데이터셋 크기 확인
dataset_size_A = len(test_data_A)
dataset_size_B = len(test_data_B)

# 랜덤한 인덱스 선택
random_index_A = random.randint(0, dataset_size_A - 1)
random_index_B = random.randint(0, dataset_size_B - 1)

# Get specific test images
test_real_A_data = test_data_A.__getitem__(random_index_A).unsqueeze(0)  # Convert to 4d tensor (BxNxHxW)
test_real_B_data = test_data_B.__getitem__(random_index_B).unsqueeze(0)


# Models
G_A = Generator(3, params.ngf, 3, params.num_resnet)
G_B = Generator(3, params.ngf, 3, params.num_resnet)
D_A = Discriminator(3, params.ndf, 1)
D_B = Discriminator(3, params.ndf, 1)
G_A.normal_weight_init(mean=0.0, std=0.02)
G_B.normal_weight_init(mean=0.0, std=0.02)
D_A.normal_weight_init(mean=0.0, std=0.02)
D_B.normal_weight_init(mean=0.0, std=0.02)
G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()


# Set the logger
D_A_log_dir = save_dir + 'D_A_logs'
D_B_log_dir = save_dir + 'D_B_logs'
if not os.path.exists(D_A_log_dir):
    os.mkdir(D_A_log_dir)
D_A_logger = Logger(D_A_log_dir)
if not os.path.exists(D_B_log_dir):
    os.mkdir(D_B_log_dir)
D_B_logger = Logger(D_B_log_dir)

G_A_log_dir = save_dir + 'G_A_logs'
G_B_log_dir = save_dir + 'G_B_logs'
if not os.path.exists(G_A_log_dir):
    os.mkdir(G_A_log_dir)
G_A_logger = Logger(G_A_log_dir)
if not os.path.exists(G_B_log_dir):
    os.mkdir(G_B_log_dir)
G_B_logger = Logger(G_B_log_dir)

cycle_A_log_dir = save_dir + 'cycle_A_logs'
cycle_B_log_dir = save_dir + 'cycle_B_logs'
if not os.path.exists(cycle_A_log_dir):
    os.mkdir(cycle_A_log_dir)
cycle_A_logger = Logger(cycle_A_log_dir)
if not os.path.exists(cycle_B_log_dir):
    os.mkdir(cycle_B_log_dir)
cycle_B_logger = Logger(cycle_B_log_dir)

img_log_dir = save_dir + 'img_logs'
if not os.path.exists(img_log_dir):
    os.mkdir(img_log_dir)
img_logger = Logger(img_log_dir)


# Loss function
MSE_loss = torch.nn.MSELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# optimizers
G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=params.lrG, betas=(params.beta1, params.beta2))
D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_A_avg_losses = []
D_B_avg_losses = []
G_A_avg_losses = []
G_B_avg_losses = []
cycle_A_avg_losses = []
cycle_B_avg_losses = []

# Generated image pool
num_pool = 50
fake_A_pool = utils.ImagePool(num_pool)
fake_B_pool = utils.ImagePool(num_pool)

step = 0
for epoch in range(params.num_epochs):
    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    cycle_A_losses = []
    cycle_B_losses = []

    # learning rate decay
    if (epoch + 1) > params.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= params.lrD / (params.num_epochs - params.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= params.lrG / (params.num_epochs - params.decay_epoch)

    # training
    for i, (real_A, real_B) in enumerate(zip(train_data_loader_A, train_data_loader_B)):

        # input image data
        real_A = Variable(real_A.cuda())
        real_B = Variable(real_B.cuda())

        # Train generator G
        # A -> B
        fake_B = G_A(real_A)
        D_B_fake_decision = D_B(fake_B)
        G_A_loss = MSE_loss(D_B_fake_decision, Variable(torch.ones(D_B_fake_decision.size()).cuda()))

        # forward cycle loss
        recon_A = G_B(fake_B)
        cycle_A_loss = L1_loss(recon_A, real_A) * params.lambdaA

        # B -> A
        fake_A = G_B(real_B)
        D_A_fake_decision = D_A(fake_A)
        G_B_loss = MSE_loss(D_A_fake_decision, Variable(torch.ones(D_A_fake_decision.size()).cuda()))

        # backward cycle loss
        recon_B = G_A(fake_A)
        cycle_B_loss = L1_loss(recon_B, real_B) * params.lambdaB

        # Back propagation
        G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # Train discriminator D_A
        D_A_real_decision = D_A(real_A)
        D_A_real_loss = MSE_loss(D_A_real_decision, Variable(torch.ones(D_A_real_decision.size()).cuda()))
        fake_A = fake_A_pool.query(fake_A)
        D_A_fake_decision = D_A(fake_A)
        D_A_fake_loss = MSE_loss(D_A_fake_decision, Variable(torch.zeros(D_A_fake_decision.size()).cuda()))

        # Back propagation
        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_optimizer.zero_grad()
        D_A_loss.backward()
        D_A_optimizer.step()

        # Train discriminator D_B
        D_B_real_decision = D_B(real_B)
        D_B_real_loss = MSE_loss(D_B_real_decision, Variable(torch.ones(D_B_real_decision.size()).cuda()))
        fake_B = fake_B_pool.query(fake_B)
        D_B_fake_decision = D_B(fake_B)
        D_B_fake_loss = MSE_loss(D_B_fake_decision, Variable(torch.zeros(D_B_fake_decision.size()).cuda()))

        # Back propagation
        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        D_B_optimizer.zero_grad()
        D_B_loss.backward()
        D_B_optimizer.step()

        # loss values
        D_A_losses.append(D_A_loss.item())
        D_B_losses.append(D_B_loss.item())
        G_A_losses.append(G_A_loss.item())
        G_B_losses.append(G_B_loss.item())
        cycle_A_losses.append(cycle_A_loss.item())
        cycle_B_losses.append(cycle_B_loss.item())

        print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
      % (epoch+1, params.num_epochs, i+1, len(train_data_loader_A), D_A_loss.item(), D_B_loss.item(), G_A_loss.item(), G_B_loss.item()))

        # ============ TensorBoard logging ============#
        D_A_logger.scalar_summary('losses', D_A_loss.item(), step + 1)
        D_B_logger.scalar_summary('losses', D_B_loss.item(), step + 1)
        G_A_logger.scalar_summary('losses', G_A_loss.item(), step + 1)
        G_B_logger.scalar_summary('losses', G_B_loss.item(), step + 1)
        cycle_A_logger.scalar_summary('losses', cycle_A_loss.item(), step + 1)
        cycle_B_logger.scalar_summary('losses', cycle_B_loss.item(), step + 1)
        step += 1

    D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
    D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
    G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
    G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
    cycle_A_avg_loss = torch.mean(torch.FloatTensor(cycle_A_losses))
    cycle_B_avg_loss = torch.mean(torch.FloatTensor(cycle_B_losses))

    # avg loss values for plot
    D_A_avg_losses.append(D_A_avg_loss)
    D_B_avg_losses.append(D_B_avg_loss)
    G_A_avg_losses.append(G_A_avg_loss)
    G_B_avg_losses.append(G_B_avg_loss)
    cycle_A_avg_losses.append(cycle_A_avg_loss)
    cycle_B_avg_losses.append(cycle_B_avg_loss)

    # Show result for test image
    test_real_A = Variable(test_real_A_data.cuda())
    test_fake_B = G_A(test_real_A)
    test_recon_A = G_B(test_fake_B)

    test_real_B = Variable(test_real_B_data.cuda())
    test_fake_A = G_B(test_real_B)
    test_recon_B = G_A(test_fake_A)

    utils.plot_train_result([test_real_A, test_real_B], [test_fake_B, test_fake_A], [test_recon_A, test_recon_B],
                            epoch, save=True, save_dir=save_dir)

    # log the images
   # log the images
result_AtoB = np.concatenate((utils.to_np(test_real_A), utils.to_np(test_fake_B), utils.to_np(test_recon_A)), axis=3)
result_BtoA = np.concatenate((utils.to_np(test_real_B), utils.to_np(test_fake_A), utils.to_np(test_recon_B)), axis=3)

# Convert images to uint8
result_AtoB_uint8 = ((result_AtoB + 1.0) / 2.0 * 255.0).astype(np.uint8)#적을예정
result_BtoA_uint8 = ((result_BtoA + 1.0) / 2.0 * 255.0).astype(np.uint8)

info = {
    'result_AtoB': result_AtoB_uint8.transpose(0, 2, 3, 1),  # convert to BxHxWxC
    'result_BtoA': result_BtoA_uint8.transpose(0, 2, 3, 1)
}

for tag, images in info.items():
    img_logger.image_summary(tag, images, epoch + 1)



# Plot average losses
avg_losses = []
avg_losses.append(D_A_avg_losses)
avg_losses.append(D_B_avg_losses)
avg_losses.append(G_A_avg_losses)
avg_losses.append(G_B_avg_losses)
avg_losses.append(cycle_A_avg_losses)
avg_losses.append(cycle_B_avg_losses)
utils.plot_loss(avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)
# Save trained parameters of model
torch.save(G_A.state_dict(), model_dir + 'generator_A_param.pkl')
torch.save(G_B.state_dict(), model_dir + 'generator_B_param.pkl')
torch.save(D_A.state_dict(), model_dir + 'discriminator_A_param.pkl')
torch.save(D_B.state_dict(), model_dir + 'discriminator_B_param.pkl')
