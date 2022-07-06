import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm


## Helper functions ##

## Epoch train
def train_epoch(dataload_train, model, optimizer, criterion, device):
    epoch_loss = 0
    for batch in tqdm(dataload_train):
        x_train, y_train = batch['inp'], batch['oup']
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(model, x_train, y_train, optimizer, criterion)
        epoch_loss += loss
    return epoch_loss


# Training function
def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output


def kfold_cv(dataload_train, model, optimizer, criterion, device, epochs, K=5):
    splits = KFold(n_splits = K, shuffle=True, random_state=42)
    foldperf = {}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ConvNet()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
            test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        foldperf['fold{}'.format(fold + 1)] = history

    testl_f, tl_f, testa_f, ta_f = [], [], [], []

    for f in range(1, K + 1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)))

    return foldperf

# Model class
class neuralNet(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.hidden_layer = nn.Linear(num_feat, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        x = functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
