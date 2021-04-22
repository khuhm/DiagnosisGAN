import argparse
from torch.utils.data import DataLoader
from dataset import CT
import pickle
import torch
from torch.backends import cudnn
from model import ClassNet, ClassNet_multi, Generic_UNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import os
from utils import VisdomLinePlotter, get_roc_auc
from torch.nn.functional import softmax

cudnn.benchmark = True


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='train_cls')
    parser.add_argument('--data_dir', type=str, default='/data/ESMH/phase_synthesis/pre_reg_cropped')
    parser.add_argument('--res_dir', type=str, default='/data/ESMH/DiagnosisGAN/cls_results/base_p4')
    parser.add_argument('--seg_model_path', type=str, default='/data/ESMH/segmentation_results/pretrained/model_final_checkpoint.model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_phase', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=500)
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

    cls_model = ClassNet_multi(num_phase=args.num_phase)
    # cls_model = ClassNet()
    cls_model.cuda()

    ce_loss = CrossEntropyLoss()
    optimizer_cls = Adam(cls_model.parameters(), lr=args.lr)

    for param in seg_model.parameters():
        param.requires_grad = False

    seg_model.eval()

    plotter = VisdomLinePlotter()
    os.makedirs(args.res_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        cls_model.train()
        loss_avg = []

        for batch_idx, data in enumerate(train_data_loader):
            img = data['img'][0].unsqueeze(1).cuda()
            seg = data['seg'][0].unsqueeze(1).cuda()
            target_idx = data['target_idx'].cuda()
            label = data['label'].cuda()

            cls_emb = seg_model(img, seg)
            pred_label = cls_model(cls_emb, target_idx)
            loss_cls_label = ce_loss(pred_label, label)

            optimizer_cls.zero_grad()
            loss_cls_label.backward()
            optimizer_cls.step()

            losses = []
            losses.append(loss_cls_label.item())
            loss_avg.append(losses)

        loss_avg = np.mean(np.array(loss_avg), axis=0)
        plotter.plot('loss', 'tr', 'loss', epoch, loss_avg[0])

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'cls_model_state_dict': cls_model.state_dict(),
                'optimizer_cls': optimizer_cls.state_dict(),
            }, os.path.join(args.res_dir, 'model_' + str(epoch) + '.pt'))

        if epoch % args.save_interval != 0:
            continue

        cls_model.eval()
        loss_avg = []
        output_scores = np.array([]).reshape(0, 5)
        true_labels = np.array([])
        for batch_idx, data in enumerate(test_data_loader):
            img = data['img'][0].unsqueeze(1).cuda()
            seg = data['seg'][0].unsqueeze(1).cuda()
            target_idx = data['target_idx'].cuda()
            label = data['label'].cuda()

            with torch.no_grad():
                cls_emb = seg_model(img, seg)
                pred_label = cls_model(cls_emb, target_idx)
                loss_cls_label = ce_loss(pred_label, label)
                output = softmax(pred_label, dim=1)

            output_scores = np.append(output_scores, output.cpu().numpy(), axis=0)
            true_labels = np.append(true_labels, label.cpu().numpy())

            losses = []
            losses.append(loss_cls_label.item())
            loss_avg.append(losses)

        loss_avg = np.mean(np.array(loss_avg), axis=0)
        plotter.plot('loss', 'vl', 'loss', epoch, loss_avg[0])

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
