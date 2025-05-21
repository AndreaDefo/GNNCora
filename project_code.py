import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Determine device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads_first=8, heads_second=1, dropout=0.6):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads_first,
            concat=True,
            dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads_first,
            out_channels,
            heads=heads_second,
            concat=False,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    
class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    def get_embeddings(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x



def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    losses = []
    accuracies = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = criterion(out[mask], data.y[mask]).item()
        pred = out[mask].max(dim=1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        losses.append(loss)
        accuracies.append(acc)
    return losses, accuracies

# Load dataset and normalize features
dataset_name = 'Cora'
dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0].to(device)

# Hyperparameters
hidden_channels = 8      
heads_first = 8          
dropout = 0.6           
lr = 0.005
weight_decay = 5e-4     
heads_second = 1         
num_trials = 10
patience = 20  # early stopping patience
max_epochs = 300

# Utility to run trials
def run_experiment(ModelClass,path,**model_kwargs):
    test_accs = []
    best_test_acc = float('inf')
    for trial in range(num_trials):
        #set_seed(trial)
        model = ModelClass(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            heads_first=heads_first if ModelClass==GATNet else None,
            heads_second=heads_second if ModelClass==GATNet else None,
            dropout=dropout
        ) if ModelClass==GATNet else ModelClass(
            in_channels=dataset.num_node_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.NLLLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(max_epochs):
            train(model, data, optimizer, criterion)
            (_, val_loss, _), _ = test(model, data, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        model.load_state_dict(best_state)
        _, (_, _, test_acc) = test(model, data, criterion)
        if test_acc<best_test_acc:
            torch.save(model.state_dict(),path)
        test_accs.append(test_acc)

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    return mean_acc, std_acc


def explore_model(ModelClass,path):
    model = ModelClass(
        in_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        heads_first=heads_first if ModelClass==GATNet else None,
        heads_second=heads_second if ModelClass==GATNet else None,
        dropout=dropout
    ) if ModelClass==GATNet else ModelClass(
        in_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout
    )
   
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    model.to(device)
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index)
    
    embeddings_np = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()

    tsne = TSNE(n_components = 2, random_state = 42)
    embeddings_tsne = tsne.fit_transform(embeddings_np)
    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
   # plt.title('t-SNE Visualization of Firt Layer Output G')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.show()


# Run both GAT and GCN experiments
gat_mean, gat_std = run_experiment(GATNet,'./model_gat')
gcn_mean, gcn_std = run_experiment(GCNNet,'./model_gcn')
print(f"GAT Test Accuracy: {gat_mean:.4f} ± {gat_std:.4f}")
print(f"GCN Test Accuracy: {gcn_mean:.4f} ± {gcn_std:.4f}")

#explore_model(GATNet, './model_gat')

#explore_model(GCNNet, './model_gcn')