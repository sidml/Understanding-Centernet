import torch
from model import Res18UnetCenterNet
from loss import criterion
from dataset import ToyDataset
from ctdet import ctdet_decode
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, optimizer, dataloader, epoch):
    model.train()
    running_mask_loss, running_size_loss, running_offset_loss = 0, 0, 0
    for batch_idx, (img_batch, mask_batch) in enumerate(dataloader):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        # center_index = center_index.to(device)

        optimizer.zero_grad()
        output = model(img_batch)

        mask_loss, size_loss, offset_loss = criterion(output, mask_batch)

        loss = mask_loss + size_loss + offset_loss

        loss.backward()

        optimizer.step()

        running_mask_loss += mask_loss
        running_size_loss += size_loss
        running_offset_loss += offset_loss

        if batch_idx % 5 == 0:
            print(f'\r{running_mask_loss/(batch_idx+1):.3f} {running_size_loss/(batch_idx+1):.3f} {running_offset_loss/(batch_idx+1):.3f}',
                  end='', flush=True)

    print('\r', end='', flush=True)
    print(f"Epoch: {epoch} mask_loss: {running_mask_loss/(batch_idx): .3f} "
          f"size_loss: {running_size_loss/(batch_idx): .3f} offset_loss: {running_offset_loss/(batch_idx): .3f}")


@torch.no_grad()
def eval_model(model, dataloader, output_folder, epoch=0, thresh=0.25):

    for (img_batch, mask_batch) in dataloader:
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        predictions = model(img_batch)
        bboxes_raw, bboxes, scores, clses = ctdet_decode(predictions[:, 0:1], predictions[:, -4:-2, :, :],
                                                         predictions[:, -2:, :, :])
        bboxes = bboxes.long().cpu().numpy()

        for batch_idx, (im, mask, pred) in enumerate(zip(img_batch, mask_batch, predictions)):
            im = im.permute(1, 2, 0).cpu().squeeze().numpy()*255
            im = np.repeat(im[:, :, None], 3, 2)

            score_pos = []
            for score, bbox, bbox_raw in zip(scores[batch_idx], bboxes[batch_idx], bboxes_raw[batch_idx]):
                if score > thresh:
                    im = np.maximum(im, cv2.rectangle(
                        im, (bbox[2], bbox[3]), (bbox[0], bbox[1]), (0, 255, 0), 2))

                    # uncomment to visualize bbox without offset correction
                    # im = np.maximum(im, cv2.rectangle(
                    #     im, (bbox_raw[2], bbox_raw[3]), (bbox_raw[0], bbox_raw[1]), (255, 0, 0), 2))
                    score_pos.append((bbox[2]+5, bbox[3]+5, score))
                else:
                    break

            plt.subplot(2, 3, 1)
            plt.title('Image with pred bbox')
            plt.imshow(im.astype(np.int))
            for pos_x, pos_y, score in score_pos:
                plt.text(pos_x, pos_y, f'{score.item():.2}', fontsize=6, c='r')

            plt.subplot(2, 3, 2)
            plt.title('Mask')
            plt.imshow(mask[0].cpu().squeeze())

            plt.subplot(2, 3, 3)
            plt.title('Mask Prediction')
            plt.imshow(pred[0].cpu().squeeze())

            plt.subplot(2, 3, 4)
            plt.title('Width Prediction')
            plt.imshow(pred[1].cpu().squeeze())

            plt.subplot(2, 3, 5)
            plt.title('Height Prediction')
            plt.imshow(pred[3].cpu().squeeze())

            plt.subplot(2, 3, 6)
            plt.title('Width offset Prediction')
            plt.imshow(pred[4].cpu().squeeze())

            plt.suptitle(f'Epoch {epoch}')
            # plt.show()
            # print(f'{output_folder}_{epoch}.jpg')
            plt.savefig(f'{output_folder}/epoch_{epoch}.jpg')
            plt.close()
            break
        break


if __name__ == '__main__':
    import os

    model = Res18UnetCenterNet()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters())
    dataset = ToyDataset(img_shape=(128, 128))

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=8, num_workers=0)

    output_folder = './outputs'
    os.makedirs(output_folder, exist_ok=True)
    for epoch in range(20):
        train_model(model, optim, dataloader, epoch)
        eval_model(model, dataloader, output_folder, epoch=epoch)

    print()
