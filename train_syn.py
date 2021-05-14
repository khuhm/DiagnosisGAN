import argparse
from torch.utils.data import DataLoader
from dataset import CT
import pickle
import torch
from torch.backends import cudnn
from model import ClassNet_three, ClassNet_multi, Generic_UNet, SynNet, Discriminator, SoftDiceLoss
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
import nibabel as nib

from torch.optim import Adam
import numpy as np
import os
from utils import VisdomLinePlotter, get_roc_auc
from torch.nn.functional import softmax

cudnn.benchmark = True


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='train_syn')
    parser.add_argument('--data_dir', type=str, default='/data/ESMH/phase_synthesis/pre_reg_cropped')
    parser.add_argument('--res_dir', type=str, default='/data/ESMH/DiagnosisGAN/syn_results/full_p3_seg')
    parser.add_argument('--seg_model_path', type=str, default='/data/ESMH/segmentation_results/pretrained/model_final_checkpoint.model')
    parser.add_argument('--cls_model_path', type=str,
                        default='/data/ESMH/DiagnosisGAN/cls_results/base_p4_seg/model_90.pt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_phase', type=int, default=3)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    args = parser.parse_args()
    print(args)

    train_data_loader = DataLoader(dataset=CT(mode='train',
                                              data_dir=args.data_dir,
                                              num_phase=args.num_phase),
                                   batch_size=args.batch_size,
                                   shuffle=True)

    test_data_loader = DataLoader(dataset=CT(mode='test',
                                             data_dir=args.data_dir,
                                             num_phase=args.num_phase),
                                  batch_size=args.batch_size,
                                  shuffle=False)

    # seg model
    with open(args.seg_model_path + '.pkl', 'rb') as f:
        seg_model_data = pickle.load(f)

    seg_model = Generic_UNet(1, 32, 3, 5,
                             pool_op_kernel_sizes=seg_model_data['plans']['plans_per_stage'][0]['pool_op_kernel_sizes'],
                             conv_kernel_sizes=seg_model_data['plans']['plans_per_stage'][0]['conv_kernel_sizes'],
                             upscale_logits=False, convolutional_pooling=True, convolutional_upsampling=True)
    seg_model.cuda()
    checkpoint = torch.load(args.seg_model_path)
    seg_model.load_state_dict(checkpoint['state_dict'])

    cls_model = ClassNet_multi(num_phase=4)
    cls_model.cuda()

    # model
    net_G = SynNet()
    net_G.cuda()

    net_D = Discriminator()
    net_D.cuda()

    l1_loss = L1Loss()
    ce_loss = CrossEntropyLoss()
    mse_loss = MSELoss()
    dice_loss = SoftDiceLoss(tumor_only=True)

    # optimizer
    optimizer_G = Adam(net_G.parameters(), lr=args.lr)
    optimizer_D = Adam(net_D.parameters(), lr=args.lr)

    # load
    if args.cls_model_path is not None:
        checkpoint = torch.load(args.cls_model_path)
        cls_model.load_state_dict(checkpoint['cls_model_state_dict'])

    for param in seg_model.parameters():
        param.requires_grad = False

    for param in cls_model.parameters():
        param.requires_grad = False

    seg_model.eval()
    cls_model.eval()

    plotter = VisdomLinePlotter()
    os.makedirs(args.res_dir, exist_ok=True)

    fake_label = torch.zeros(1).cuda()
    real_label = torch.ones(1).cuda()

    for epoch in range(args.num_epochs):
        net_G.train()
        net_D.train()
        loss_avg = []

        for batch_idx, data in enumerate(train_data_loader):
            syn_in = data['syn_in'][0].cuda()
            imgs_four = data['imgs_four'][0].unsqueeze(1).cuda()
            syn_target = data['syn_target'][0].unsqueeze(1).cuda()
            seg = data['seg'][0].unsqueeze(1).cuda()
            target_indices = data['target_indices']
            label = data['label'].cuda()

            output = net_G(syn_in)

            for param in net_D.parameters():
                param.requires_grad = True

            pred_fake = net_D(output.detach())
            loss_D_fake = mse_loss(pred_fake, fake_label)
            pred_real = net_D(syn_target)
            loss_D_real = mse_loss(pred_real, real_label)
            loss_D = (loss_D_fake + loss_D_real) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            for param in net_D.parameters():
                param.requires_grad = False

            cls_emb = []
            pred_seg = []
            cnt = 0
            for i in range(4):
                if i in target_indices:
                    out_emb, out_seg = seg_model(output[[cnt]], get_seg=True)
                    cls_emb.append(out_emb)
                    pred_seg.append(out_seg)
                    cnt = cnt + 1
                else:
                    cls_emb.append(seg_model(imgs_four[[i]]))
            cls_emb = torch.cat(cls_emb, dim=1)
            pred_seg = torch.cat(pred_seg, dim=1)
            pred_G_fake = net_D(output)
            loss_G_GAN = mse_loss(pred_G_fake, real_label)
            loss_G_L1 = l1_loss(output, syn_target)

            pred_G_cls = cls_model(cls_emb, 0)
            loss_G_cls = ce_loss(pred_G_cls, label)
            loss_G_seg = dice_loss(pred_seg, seg[target_indices])
            loss_G = loss_G_GAN + loss_G_L1 + 0.1 * loss_G_seg + 0.1 * loss_G_cls
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            losses = []
            losses.append(loss_D_fake.item())
            losses.append(loss_D_real.item())
            losses.append(loss_G_GAN.item())
            losses.append(loss_G_L1.item())
            losses.append(loss_G_seg.item())
            losses.append(loss_G_cls.item())
            loss_avg.append(losses)

        loss_avg = np.mean(np.array(loss_avg), axis=0)
        plotter.plot('train', 'D_f', 'train loss', epoch, loss_avg[0])
        plotter.plot('train', 'D_r', 'train loss', epoch, loss_avg[1])
        plotter.plot('train', 'G_G', 'train loss', epoch, loss_avg[2])
        plotter.plot('train', 'G_L', 'train loss', epoch, loss_avg[3])
        plotter.plot('train', 'G_S', 'train loss', epoch, loss_avg[4])
        plotter.plot('train', 'G_C', 'train loss', epoch, loss_avg[5])

        if epoch % args.save_interval == 0:
            affine = np.eye(4)
            affine[[0, 1], [0, 1]] = -1.50669
            affine[2, 2] = 3
            result = nib.Nifti1Image(
                np.transpose(output.squeeze().detach().cpu().numpy()), affine)
            nib.save(result, os.path.join(args.res_dir, 'results.nii.gz'))

            torch.save({
                'epoch': epoch,
                'net_G_state_dict': net_G.state_dict(),
                'net_D_state_dict': net_D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, os.path.join(args.res_dir, 'model_' + str(epoch) + '.pt'))

        if epoch % args.print_interval != 0:
            continue

        net_G.eval()
        net_D.eval()
        loss_avg = []
        output_scores = np.array([]).reshape(0, 5)
        true_labels = np.array([])
        for batch_idx, data in enumerate(test_data_loader):
            syn_in = data['syn_in'][0].cuda()
            imgs_four = data['imgs_four'][0].unsqueeze(1).cuda()
            seg = data['seg'][0].unsqueeze(1).cuda()
            target_indices = np.array(data['target_indices'])
            label = data['label'].cuda()

            with torch.no_grad():
                output = net_G(syn_in)
                imgs_four[target_indices] = output

                cls_emb = []
                for i in range(4):
                    cls_emb.append(seg_model(imgs_four[[i]]))
                cls_emb = torch.cat(cls_emb, dim=1)

                # cls_emb = seg_model(imgs_four)

                pred_label = cls_model(cls_emb, 0)
                loss_cls_label = ce_loss(pred_label, label)
                output = softmax(pred_label, dim=1)

            output_scores = np.append(output_scores, output.cpu().numpy(), axis=0)
            true_labels = np.append(true_labels, label.cpu().numpy())

            losses = []
            losses.append(loss_cls_label.item())
            loss_avg.append(losses)

        auc = get_roc_auc(true_labels, output_scores)
        plotter.plot('auc', 'auc', 'auc', epoch, auc[5])
        plotter.plot('auc', 'auc_0', 'auc', epoch, auc[0])
        plotter.plot('auc', 'auc_1', 'auc', epoch, auc[1])
        plotter.plot('auc', 'auc_2', 'auc', epoch, auc[2])
        plotter.plot('auc', 'auc_3', 'auc', epoch, auc[3])
        plotter.plot('auc', 'auc_4', 'auc', epoch, auc[4])

    print('done')


if __name__ == '__main__':
    main()
