
# This is redundant now, but it won't be when we introduce
# Datasets...
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set learning rate - this is "eta"
lr = 0.1

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1,1)).to(device)

# Defines an SGD optimizer to update the paramters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Definese an MSE loss function
loss_fn = nn.MSELoss(reduction='mean')
