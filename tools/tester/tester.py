import torch
import torch.nn.functional as F
import torchmetrics as metrics

class Ai4MarsTester():

    def __init__(self, loss_fn, metric, test_loader=None, device='cpu') -> None:
        self.loss_fn = loss_fn
        self.metric = metric
        self.test_loader = test_loader
        self.device  = device
        pass

    def test_one_epoch(self, model):
        running_loss = 0.
        last_loss = 0.
        total_outputs = []
        total_labels = []

        for batch_index, batch in enumerate(self.test_loader):

            # Inputs labes pair
            inputs, labels = batch

            # Send Tensors to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = model(inputs)

            # New shape: B x W x H x C
            outputs = outputs.permute(0,2,1,3).permute(0,1,3,2)

            labels = labels.squeeze()
            labels = labels.long()

            # Accumulate inputs labels to compute metrics
            total_outputs.append(outputs)
            total_labels.append(labels)

            loss = self.loss_fn(outputs, labels)
            running_loss += loss.item()

            # Free up RAM/VRAM
            inputs.detach()
            labels.detach()
            del inputs
            del labels
            torch.cuda.empty_cache()

        # Compute average loss over all batches
        last_loss = (running_loss) / (batch_index + 1 )

        print(f"Test loss: {last_loss}\n")

        # Preprocessing for metrics computation
        if len(self.test_loader.dataset) % self.test_loader.batch_size != 0:
            total_outputs.pop(-1)
            total_labels.pop(-1)

        total_outputs = torch.stack(total_outputs, dim=0)
        total_labels = torch.stack(total_labels, dim=0)

        total_outputs = F.softmax(total_outputs, dim=-1)
        total_labels = F.one_hot(total_labels, num_classes=total_outputs.shape[-1])

        # Index metrics
        value = self.metric(total_outputs.reshape(-1), total_labels.reshape(-1))

        print(f"Metrics {self.metric}: {value.item()}")