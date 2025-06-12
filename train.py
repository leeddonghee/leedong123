import numpy as np
import dezero.functions as F
import dezero.optimizers as optim
from model import ResidualMLP
from dataset import train_set
from dezero import DataLoader
import plotly.graph_objects as go  # ✅ plotly만 사용

def train_model(lr, opt_name, batch_size, epochs=10):
    model = ResidualMLP()
    train_loader = DataLoader(train_set, batch_size=batch_size)

    if opt_name == 'SGD':
        optimizer = optim.SGD(lr).setup(model)
    else:
        optimizer = optim.Adam(lr).setup(model)

    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        sum_loss = 0
        sum_acc = 0
        for x, t in train_loader:
            x = x.reshape(len(x), -1)
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data)

            pred = y.data.argmax(axis=1)
            sum_acc += np.sum(pred == t)

        avg_loss = sum_loss / len(train_loader)
        avg_acc = sum_acc / len(train_set)

        loss_list.append(avg_loss)
        acc_list.append(avg_acc)

        print(f'Epoch {epoch+1}, Loss {avg_loss:.4f}, Accuracy {avg_acc:.4f}')

    return loss_list, acc_list

if __name__ == '__main__':
    learning_rates = [0.1, 0.01]
    optimizers = ['SGD', 'Adam']
    batch_sizes = [100, 200]

    results = {}

    for lr in learning_rates:
        for opt_name in optimizers:
            for batch_size in batch_sizes:
                print(f'\nRunning experiment: lr={lr}, optimizer={opt_name}, batch_size={batch_size}')
                losses, accs = train_model(lr, opt_name, batch_size, epochs=10)
                results[(lr, opt_name, batch_size)] = (losses, accs)

    # ✅ Plotly로 정확도 시각화
    fig = go.Figure()

    for key, (losses, accs) in results.items():
        lr, opt_name, batch_size = key
        label = f'lr={lr}, opt={opt_name}, bs={batch_size}'
        fig.add_trace(go.Scatter(
            x=list(range(1, len(accs) + 1)),
            y=accs,
            mode='lines+markers',
            name=label
        ))

    fig.update_layout(
        title='Accuracy over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        legend_title='Settings',
        template='plotly_dark'
    )

    fig.show()
